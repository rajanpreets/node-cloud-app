import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rajan"
EMBEDDING_DIMENSION = 384

# Initialize Pinecone
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Initialize models
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Model initialization failed: {str(e)}")
    st.stop()

def retrieve_jobs_from_pinecone(query_text):
    try:
        query_embedding = embedding_model.encode(query_text).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="jobs"
        )
        return [match.metadata for match in results.matches if match.metadata]
    except Exception as e:
        return f"⚠️ Query error: {str(e)}"

def display_jobs_table(jobs):
    if not jobs or isinstance(jobs, str):
        st.error(jobs if jobs else "No jobs found")
        return
    
    # Create DataFrame with relevant columns
    jobs_df = pd.DataFrame([{
        "Title": job.get("Job Title", "N/A"),
        "Company": job.get("Company Name", "N/A"),
        "Location": job.get("Location", "N/A"),
        "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "N/A",
        "Link": job.get("Job Link", "#")
    } for job in jobs])
    
    # Display as interactive table with clickable links
    if not jobs_df.empty:
        st.markdown("### 🗃️ Matching Jobs")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("Apply Now"),
                "Description": "Job Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No valid job listings found")

def generate_analysis(jobs):
    if isinstance(jobs, str):
        return jobs
        
    job_texts = "\n\n".join([f"Title: {job.get('Job Title')}\nCompany: {job.get('Company Name')}\nDescription: {job.get('Job Description', '')[:300]}" 
                          for job in jobs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a career advisor. Analyze these jobs and give 3-5 brief recommendations:"),
        ("human", f"Resume content will follow this message. Here are matching jobs:\n\n{job_texts}\n\nProvide concise, actionable advice for the applicant.")
    ])
    
    try:
        return llm.invoke(prompt.format_messages()).content
    except Exception as e:
        return f"⚠️ Analysis error: {str(e)}"

def tailor_resume_interaction(resume_text, jobs):
    st.markdown("---")
    st.markdown("### ✨ Resume Tailoring")
    
    job_titles = [job.get("Job Title", "Unknown Position") for job in jobs]
    selected_title = st.selectbox("Which job would you like to tailor your resume for?", job_titles)
    
    if selected_title:
        selected_job = next(job for job in jobs if job.get("Job Title") == selected_title)
        if st.button("Generate Tailored Resume Suggestions"):
            with st.spinner("🔧 Customizing your resume..."):
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You're a professional resume writer. Tailor this resume for the specific job."),
                        ("human", f"Job Title: {selected_title}\nJob Description:\n{selected_job.get('Job Description', '')}\n\nResume:\n{resume_text}\n\nProvide specific suggestions to modify the resume. Focus on matching keywords and required skills.")
                    ])
                    response = llm.invoke(prompt.format_messages())
                    st.markdown("### 📝 Customization Suggestions")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Error generating suggestions: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title("💬 AI Career Assistant")

if 'jobs' not in st.session_state:
    st.session_state.jobs = None

with st.chat_message("assistant"):
    st.write("Hi! I'm your AI career assistant. Paste your resume below and I'll help you find relevant jobs!")

resume_text = st.chat_input("Paste your resume text here...")

if resume_text:
    with st.spinner("🔍 Analyzing your resume and searching for jobs..."):
        # Store resume in session state
        st.session_state.resume_text = resume_text
        
        # Retrieve and store jobs
        jobs = retrieve_jobs_from_pinecone(resume_text)
        if isinstance(jobs, str):
            st.error(jobs)
            st.stop()
            
        st.session_state.jobs = jobs
        
        # Display results
        st.markdown("---")
        with st.chat_message("assistant"):
            st.markdown("### 🎯 Here's what I found for you:")
            display_jobs_table(jobs)
            
            analysis = generate_analysis(jobs)
            st.markdown("---")
            st.markdown("### 📊 Career Advisor Analysis")
            st.write(analysis)
            
        # Show tailoring interface
        tailor_resume_interaction(resume_text, jobs)

# Debug section
if st.session_state.get('jobs'):
    with st.expander("🔧 Debug Information"):
        st.write("Raw job data:", st.session_state.jobs)
        st.write("Resume text length:", len(st.session_state.get('resume_text', '')))
