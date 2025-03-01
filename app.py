import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np
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
EMBEDDING_DIMENSION = 384  # Match MiniLM-L6-v2 dimension

# Initialize Pinecone with better error handling
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Verify or create index
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            st.success(f"✅ Created new Pinecone index '{INDEX_NAME}'")
            
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Validate embedding model
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Verify embedding dimension
    test_embedding = embedding_model.encode("test")
    if len(test_embedding) != EMBEDDING_DIMENSION:
        st.error(f"❌ Embedding dimension mismatch: Expected {EMBEDDING_DIMENSION}, Got {len(test_embedding)}")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to load embedding model: {str(e)}")
    st.stop()

# Initialize LLM with validation
try:
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
    # Test LLM connection
    llm.invoke("test")
except Exception as e:
    st.error(f"❌ LLM initialization failed: {str(e)}")
    st.stop()

def retrieve_jobs_from_pinecone(query_text):
    """Robust Pinecone query function with error handling"""
    try:
        # Generate and validate embedding
        query_embedding = embedding_model.encode(query_text).tolist()
        if len(query_embedding) != EMBEDDING_DIMENSION:
            return f"⚠️ Invalid embedding dimension: {len(query_embedding)}"

        # Query Pinecone with timeout
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="jobs"  # Match your upload namespace
        )
        
        # Validate response structure
        if not results or "matches" not in results:
            return "⚠️ No results found or invalid response structure"
            
        return [match.metadata for match in results.matches if match.metadata]
    
    except Exception as e:
        return f"⚠️ Query error: {str(e)}"

def format_jobs_for_llm(jobs):
    """Safe formatting with validation"""
    if not jobs or isinstance(jobs, str):
        return jobs  # Return error messages
    
    job_texts = []
    for job in jobs:
        try:
            # Use EXACT metadata keys from Pinecone
            title = job.get("Job Title", "N/A").strip()
            company = job.get("Company Name", "N/A").strip()
            location = job.get("Location", "N/A").strip()
            link = job.get("Job Link", "#").strip()
            description = job.get("Job Description", "").strip()[:500]  # Limit length
            
            job_text = (
                f"### [{title}]({link})\n"
                f"**Company**: {company}\n"
                f"**Location**: {location}\n"
                f"**Description**: {description or 'No description available'}\n"
            )
            job_texts.append(job_text)
        except Exception as e:
            continue  # Skip invalid entries
            
    return "\n\n".join(job_texts) if job_texts else "⚠️ No valid job data found"

def generate_job_listings(jobs):
    """Improved LLM integration with validation"""
    if isinstance(jobs, str):
        return jobs  # Return existing error messages
        
    formatted_jobs = format_jobs_for_llm(jobs)
    if "⚠️" in formatted_jobs:
        return formatted_jobs
        
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful job search assistant. Analyze these job listings:"),
            ("human", f"Job Listings:\n\n{formatted_jobs}\n\nProvide concise recommendations.")
        ])
        response = llm.invoke(prompt.format_messages())
        return response.content
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="🔍 AI Job Finder", layout="wide")
st.title("🔍 AI Job Finder")
st.write("Paste your resume text to find matching jobs!")

with st.form("job_search_form"):
    resume_text = st.text_area("📄 Resume Text:", height=200)
    submitted = st.form_submit_button("🔍 Find Jobs")
    
    if submitted and resume_text.strip():
        with st.spinner("🔍 Searching for best matches..."):
            try:
                # Step 1: Retrieve jobs
                jobs = retrieve_jobs_from_pinecone(resume_text)
                
                # Step 2: Generate response
                if isinstance(jobs, str):
                    st.error(jobs)
                else:
                    response = generate_job_listings(jobs)
                    st.markdown("## 🎯 Job Recommendations")
                    st.markdown(response)
                    
                    # Show raw matches for debugging
                    with st.expander("🔧 Debug Information"):
                        st.write("### Raw Pinecone Results", jobs)
                        
            except Exception as e:
                st.error(f"❌ Critical error: {str(e)}")
                st.text(traceback.format_exc())
