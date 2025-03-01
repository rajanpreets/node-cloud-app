import streamlit as st
import os
from pinecone import Pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import PyPDF2  # For extracting text from PDFs

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rajan"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM Model
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to normalize embeddings
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

# Function to retrieve jobs from Pinecone
def retrieve_jobs_from_pinecone(resume_text):
    query_embedding = [normalize_vector(embedding_model.encode(resume_text)).tolist()]  # Convert text to vector

    # Check if index has data
    index_stats = index.describe_index_stats()
    if index_stats["total_vector_count"] == 0:
        return "⚠️ No job data available in Pinecone. Try again later."

    # Perform the query
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    jobs = [match["metadata"] for match in results.get("matches", [])]

    if not jobs:
        return "⚠️ No matching jobs found for your resume."

    return jobs

# Function to generate AI job recommendations
def generate_ai_job_recommendations(jobs):
    job_text = "\n\n".join([
        f"**{job.get('title', 'Unknown')}** at {job.get('company', 'Unknown')}\n"
        f"📍 {job.get('location', 'N/A')} | 💰 {job.get('salary', 'N/A')}\n"
        f"📝 {job.get('description', 'No Description')}"
        for job in jobs
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI job assistant. Given the following job listings, present the best opportunities in a user-friendly and engaging format."),
        ("human", "Here is the extracted resume:\n\n{resume_text}\n\nBased on this, I found the following jobs:\n\n{job_text}\n\nPlease provide a structured response.")
    ])

    return llm.invoke({"resume_text": resume_text, "job_text": job_text})

# Streamlit UI
st.set_page_config(page_title="🔍 AI Job Matcher", layout="wide")

st.title("📄 AI Resume-Based Job Finder")
st.write("Upload your resume, and AI will find the best job matches for you.")

# User uploads resume
uploaded_file = st.file_uploader("📎 Upload your resume (PDF format only):", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip():
        st.success("✅ Resume uploaded successfully. Searching for best job matches...")

        # Retrieve relevant jobs from Pinecone
        jobs = retrieve_jobs_from_pinecone(resume_text)

        if isinstance(jobs, list):
            # Generate AI-generated job recommendations
            ai_response = generate_ai_job_recommendations(jobs)
            st.write("### 🤖 AI-Generated Job Recommendations:")
            st.write(ai_response)  # Display the AI-generated output
        else:
            st.write(jobs)  # Display error message if no jobs found
    else:
        st.error("⚠️ Could not extract text from resume. Try another file.")
