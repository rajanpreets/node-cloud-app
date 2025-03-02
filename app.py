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
import time
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login

# Load environment variables and initialize database
load_dotenv()
init_db()

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
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
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

# LangGraph nodes and workflow (keep existing implementation)
# ... [Keep all the existing LangGraph node definitions and workflow setup] ...

# Authentication UI Component
def authentication_ui():
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = get_user_by_username(username)
                if user and verify_password(user[2], password):
                    update_last_login(username)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with register_tab:
        with st.form("Register"):
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            submit = st.form_submit_button("Create Account")
            
            if submit:
                try:
                    create_user(new_username, new_password, new_email)
                    st.success("Account created! Please login")
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title("💬 AI Career Assistant")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Show authentication if not logged in
if not st.session_state.logged_in:
    authentication_ui()
    st.stop()

# Logout button
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.agent_state = {
            "resume_text": "",
            "jobs": [],
            "history": [],
            "current_response": "",
            "selected_job": None
        }
        st.rerun()
    st.write(f"Logged in as: {st.session_state.username}")

# Main Application Functionality
def main_application():
    with st.chat_message("assistant"):
        st.write("Hi! I'm your AI career assistant. Paste your resume below and I'll help you find relevant jobs!")

    resume_text = st.chat_input("Paste your resume text here...")

    if resume_text:
        st.session_state.agent_state.update({
            "resume_text": resume_text,
            "selected_job": None  # Reset selected job on new input
        })
        
        # Execute main workflow
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
    if st.session_state.agent_state.get("jobs"):
        st.markdown("---")
        st.markdown("### ✨ Resume Tailoring")
        
        job_titles = [job.get("Job Title", "Unknown Position") for job in st.session_state.agent_state["jobs"]]
        selected_title = st.selectbox("Which job would you like to tailor your resume for?", job_titles)
        
        if selected_title:
            selected_job = next(
                job for job in st.session_state.agent_state["jobs"] 
                if job.get("Job Title") == selected_title
            )
            st.session_state.agent_state["selected_job"] = selected_job
            
            if st.button("Generate Tailored Resume Suggestions"):
                # Directly invoke tailoring without re-running whole workflow
                result = tailor_resume(st.session_state.agent_state)
                st.session_state.agent_state.update(result)
                
                st.markdown("### 📝 Customization Suggestions")
                st.write(st.session_state.agent_state["current_response"])

    # Debug section
    if st.session_state.agent_state.get('jobs'):
        with st.expander("🔧 Debug Information"):
            st.write("Agent State:", st.session_state.agent_state)
            st.write("History:", st.session_state.agent_state.get("history", []))

# Run main application
main_application()
