import streamlit as st
import os
from pinecone import Pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rajan"  # Ensure this index exists

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"⚠️ Pinecone initialization error: {e}")

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM Model
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

# Function to normalize embeddings
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec  # Avoid division by zero

# Function to retrieve jobs using Pinecone
def retrieve_jobs_from_pinecone(resume_text):
    query_embedding = normalize_vector(embedding_model.encode(resume_text)).tolist()  # Ensure it's a flat list

    # Verify index has data
    try:
        index_stats = index.describe_index_stats()
        if index_stats.get("total_vector_count", 0) == 0:
            return "⚠️ No job data available in Pinecone."
    except Exception as e:
        return f"⚠️ Pinecone error: {e}"

    # Query Pinecone
    try:
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        jobs = [match["metadata"] for match in results.get("matches", [])]
        return jobs if jobs else "⚠️ No matching jobs found."
    except Exception as e:
        return f"⚠️ Error querying Pinecone: {e}"

# Function to generate job listing using LLM
def generate_job_listings(jobs):
    if isinstance(jobs, str):  # If it's an error message, return as-is
        return jobs
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a job search assistant. Based on the retrieved job listings, suggest the best jobs."),
        ("human", "Here are relevant job listings: {jobs}. Provide a professional summary.")
    ])
    
    try:
        response = llm.invoke({"jobs": jobs})
        return response
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# Streamlit UI
st.set_page_config(page_title="🔍 AI Job Finder", layout="wide")

st.title("🔍 AI Job Finder")
st.write("Upload your resume, and our AI will find the best job matches for you!")

# Resume Upload
uploaded_file = st.file_uploader("📄 Upload your resume (TXT format)", type=["txt"])

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")
    
    if st.button("🔍 Find My Best Job Matches"):
        jobs = retrieve_jobs_from_pinecone(resume_text)
        job_listings = generate_job_listings(jobs)

        st.write("### 🤖 AI Job Recommendations:")
        st.write(job_listings)
