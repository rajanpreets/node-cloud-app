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

# Function to retrieve jobs using LLM and Pinecone
def retrieve_jobs(query):
    try:
        query_embedding = embedding_model.encode(query).tolist()  # Ensure it's a list

        # Debugging: Print embedding shape
        st.write(f"🔍 Embedding length: {len(query_embedding)}")

        # Check if index has data
        index_stats = index.describe_index_stats()
        if index_stats["total_vector_count"] == 0:
            return "⚠️ No job data available in Pinecone. Try a different query."

        # Perform the query
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        if "matches" in results and results["matches"]:
            jobs = [match["metadata"] for match in results["matches"]]
            return jobs
        else:
            return "⚠️ No matching jobs found. Try refining your search query."

    except Exception as e:
        st.error(f"❌ Error in job retrieval: {str(e)}")
        return "⚠️ Something went wrong. Try again!"

# Function to generate AI response for follow-up questions
def generate_answer(context, question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a job search assistant. Answer based on the provided job listings."),
        ("human", "Job Listings: {context} \n\n Question: {question}")
    ])
    return llm.invoke({"context": context, "question": question})

# Streamlit UI
st.set_page_config(page_title="🔍 AI Job Finder", layout="wide")

st.title("🔍 AI Job Finder")
st.write("Find jobs based on your query. No need for predefined criteria!")

# User input for job search
user_query = st.text_area("💼 Enter your job-related query (skills, industry, location, etc.):", height=150)

if st.button("🔍 Find Jobs"):
    if user_query.strip():
        jobs = retrieve_jobs(user_query)

        if isinstance(jobs, list):
            st.write("### 🔹 Top Matching Jobs:")
            for job in jobs:
                st.write(f"**{job.get('title', 'Unknown')}** at {job.get('company', 'Unknown')}")
                st.write(f"📍 {job.get('location', 'N/A')} | 💰 {job.get('salary', 'N/A')}")
                st.write(f"📝 {job.get('description', 'No Description')}")
                st.write("---")
        else:
            st.write("🤖 **AI-Generated Job Suggestions:**")
            st.write(jobs)  # LLM-generated job recommendations

    else:
        st.error("⚠️ Please enter a job query!")

# Follow-up Question
follow_up = st.text_input("🤖 Ask a question about these jobs:")

if st.button("🤖 Get AI Answer"):
    if follow_up.strip() and 'jobs' in locals() and jobs:
        answer = generate_answer(jobs, follow_up)
        st.write("### 🤖 AI Response:")
        st.write(answer)
    else:
        st.error("⚠️ Enter a valid question and make sure job results are displayed!")
