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
INDEX_NAME = "rajan"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize the Pinecone index
index = pc.Index(INDEX_NAME)

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM Model
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

# Function to normalize embeddings
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Function to retrieve jobs from Pinecone
def retrieve_jobs(query):
    try:
        query_embedding = embedding_model.encode(query).tolist()  # Convert to list

        # Check if index has data
        index_stats = index.describe_index_stats()
        if index_stats["total_vector_count"] == 0:
            return None, "⚠️ No job data available in Pinecone. Try a different query."

        # Perform the query
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        if "matches" in results and results["matches"]:
            jobs = [match["metadata"] for match in results["matches"]]
            return jobs, None
        else:
            return None, "⚠️ No matching jobs found. Try refining your search query."

    except Exception as e:
        st.error(f"❌ Error in job retrieval: {str(e)}")
        return None, "⚠️ Something went wrong. Try again!"

# Function to generate LLM response based on retrieved jobs
def generate_job_listings(jobs):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful job assistant. Based on the provided job listings, create a structured and user-friendly list of job opportunities."),
        ("human", "Here are the job listings: {jobs} \n\n Please format them clearly.")
    ])
    return llm.invoke({"jobs": jobs})

# Streamlit UI
st.set_page_config(page_title="🔍 AI Job Finder", layout="wide")

st.title("🔍 AI Job Finder")
st.write("Find jobs based on your query. No need for predefined criteria!")

# User input for job search
user_query = st.text_area("💼 Enter your job-related query (skills, industry, location, etc.):", height=150)

if st.button("🔍 Find Jobs"):
    if user_query.strip():
        jobs, error_message = retrieve_jobs(user_query)

        if jobs:
            st.write("### 🤖 AI-Generated Job Listings:")
            job_listings = generate_job_listings(jobs)
            st.write(job_listings)  # Display formatted response from LLM
        else:
            st.error(error_message)
    else:
        st.error("⚠️ Please enter a job query!")

# Follow-up Question
follow_up = st.text_input("🤖 Ask a question about these jobs:")

if st.button("🤖 Get AI Answer"):
    if follow_up.strip() and 'jobs' in locals() and jobs:
        answer = generate_job_listings(jobs)  # Reuse LLM for follow-up questions
        st.write("### 🤖 AI Response:")
        st.write(answer)
    else:
        st.error("⚠️ Enter a valid question and make sure job results are displayed!")
