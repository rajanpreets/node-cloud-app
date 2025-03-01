import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
import operator

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rajan"
EMBEDDING_DIMENSION = 384

# Define State for LangGraph
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

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

# Define LangGraph nodes
def retrieve_jobs(state: AgentState):
    try:
        query_embedding = embedding_model.encode(state["resume_text"]).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="jobs"
        )
        jobs = [match.metadata for match in results.matches if match.metadata]
        return {"jobs": jobs, "history": ["Retrieved jobs from Pinecone"]}
    except Exception as e:
        return {"error": str(e)}

def generate_analysis(state: AgentState):
    job_texts = "\n\n".join([f"Title: {job.get('Job Title')}\nCompany: {job.get('Company Name')}\nDescription: {job.get('Job Description', '')[:300]}" 
                          for job in state["jobs"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a career advisor. Analyze these jobs and give 3-5 brief recommendations:"),
        ("human", f"Resume content will follow this message. Here are matching jobs:\n\n{job_texts}\n\nProvide concise, actionable advice for the applicant.")
    ])
    
    try:
        analysis = llm.invoke(prompt.format_messages()).content
        return {"current_response": analysis, "history": ["Generated career analysis"]}
    except Exception as e:
        return {"error": str(e)}

def tailor_resume(state: AgentState):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a professional resume writer. Tailor this resume for the specific job."),
            ("human", f"Job Title: {state['selected_job']['Job Title']}\nJob Description:\n{state['selected_job'].get('Job Description', '')}\n\nResume:\n{state['resume_text']}\n\nProvide specific suggestions to modify the resume. Focus on matching keywords and required skills.")
        ])
        response = llm.invoke(prompt.format_messages())
        return {"current_response": response.content, "history": ["Generated tailored resume suggestions"]}
    except Exception as e:
        return {"error": str(e)}

# Build LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_jobs", retrieve_jobs)
workflow.add_node("generate_analysis", generate_analysis)
workflow.add_node("tailor_resume", tailor_resume)

workflow.set_entry_point("retrieve_jobs")
workflow.add_edge("retrieve_jobs", "generate_analysis")
workflow.add_conditional_edges(
    "generate_analysis",
    lambda x: "tailor_resume" if x.get("selected_job") else END,
    {"tailor_resume": "tailor_resume", END: END}
)
workflow.add_edge("tailor_resume", END)
app = workflow.compile()

# Streamlit UI components
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

# Streamlit UI
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title("💬 AI Career Assistant")

# Initialize session state
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

with st.chat_message("assistant"):
    st.write("Hi! I'm your AI career assistant. Paste your resume below and I'll help you find relevant jobs!")

resume_text = st.chat_input("Paste your resume text here...")

if resume_text:
    st.session_state.agent_state["resume_text"] = resume_text
    for event in app.stream(st.session_state.agent_state):
        for key, value in event.items():
            st.session_state.agent_state.update(value)
    
    st.markdown("---")
    with st.chat_message("assistant"):
        st.markdown("### 🎯 Here's what I found for you:")
        display_jobs_table(st.session_state.agent_state["jobs"])
        
        st.markdown("---")
        st.markdown("### 📊 Career Advisor Analysis")
        st.write(st.session_state.agent_state["current_response"])

# Tailoring interface
if st.session_state.agent_state["jobs"]:
    st.markdown("---")
    st.markdown("### ✨ Resume Tailoring")
    
    job_titles = [job.get("Job Title", "Unknown Position") for job in st.session_state.agent_state["jobs"]]
    selected_title = st.selectbox("Which job would you like to tailor your resume for?", job_titles)
    
    if selected_title:
        st.session_state.agent_state["selected_job"] = next(
            job for job in st.session_state.agent_state["jobs"] 
            if job.get("Job Title") == selected_title
        )
        
        if st.button("Generate Tailored Resume Suggestions"):
            for event in app.stream(st.session_state.agent_state):
                for key, value in event.items():
                    st.session_state.agent_state.update(value)
            
            st.markdown("### 📝 Customization Suggestions")
            st.write(st.session_state.agent_state["current_response"])

# Debug section
if st.session_state.agent_state.get('jobs'):
    with st.expander("🔧 Debug Information"):
        st.write("Agent State:", st.session_state.agent_state)
        st.write("History:", st.session_state.agent_state["history"])
