import streamlit as st
import psycopg2
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator
import time
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {background-color: #004280; color: white;}
        .stTextInput>div>div>input {border: 1px solid #004280;}
        .stSelectbox>div>div>select {border: 1px solid #004280;}
        .header {color: #004280; font-family: 'Helvetica Neue', sans-serif;}
        .subheader {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;}
        .sidebar .sidebar-content {background-color: #e9ecef;}
    </style>
""", unsafe_allow_html=True)

# Initialize database with secrets
db_config = {
    'name': st.secrets.db.name,
    'user': st.secrets.db.user,
    'password': st.secrets.db.password,
    'host': st.secrets.db.host,
    'port': st.secrets.db.port
}
init_db(db_config)

# Define State for LangGraph
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

# Initialize Pinecone with secrets
def init_pinecone():
    try:
        pc = Pinecone(api_key=st.secrets.pinecone.api_key)
        if st.secrets.pinecone.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=st.secrets.pinecone.index_name,
                dimension=st.secrets.pinecone.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            while not pc.describe_index(st.secrets.pinecone.index_name).status['ready']:
                time.sleep(1)
        return pc.Index(st.secrets.pinecone.index_name)
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Initialize models with secrets
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets.groq.api_key)
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

# LangGraph nodes and workflow (remain unchanged from your original implementation)
# ... [Keep the same LangGraph implementation as provided] ...

# Professional UI Components
def display_jobs_table(jobs):
    if not jobs:
        st.warning("No matching positions found")
        return
    
    try:
        jobs_df = pd.DataFrame([{
            "Title": job.get("Job Title", "Not Available"),
            "Company": job.get("Company Name", "Not Available"),
            "Location": job.get("Location", "Not Available"),
            "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "Not Available",
            "Link": job.get("Job Link", "#")
        } for job in jobs])
        
        st.markdown("#### Matching Career Opportunities")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("Application Link"),
                "Description": "Position Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Data display error: {str(e)}")

# Enhanced Authentication UI
def authentication_ui():
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150x50.png?text=Company+Logo", use_column_width=True)
        
        with col2:
            tab1, tab2 = st.tabs(["Secure Sign In", "New Registration"])
            
            with tab1:
                with st.form("Secure Login"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Authenticate"):
                        handle_login(username, password)

            with tab2:
                with st.form("New Account"):
                    new_user = st.text_input("Create Username")
                    new_email = st.text_input("Email Address")
                    new_pass = st.text_input("Create Password", type="password")
                    if st.form_submit_button("Register Account"):
                        handle_registration(new_user, new_pass, new_email)

def handle_login(username, password):
    user = get_user_by_username(username)
    if user and verify_password(user[2], password):
        update_last_login(username)
        st.session_state.logged_in = True
        st.session_state.username = username
        st.rerun()
    else:
        st.error("Invalid credentials - please verify and try again")

def handle_registration(username, password, email):
    if len(password) < 8:
        st.error("Password must contain at least 8 characters")
        return
    try:
        create_user(username, password, email)
        st.success("Account created successfully. Please proceed to login")
    except Exception as e:
        st.error(f"Registration error: {str(e)}")

# Main Application UI
st.set_page_config(page_title="AI Career Optimization Platform", layout="wide")
st.markdown("<h1 class='header'>AI Career Optimization Platform</h1>", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Authentication flow
if not st.session_state.logged_in:
    authentication_ui()
    st.stop()

# Sidebar Management
with st.sidebar:
    st.markdown(f"**Active Session:** {st.session_state.username}")
    if st.button("Terminate Session"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Platform Navigation**")
    st.markdown("- Career Analysis")
    st.markdown("- Resume Optimization")
    st.markdown("- Job Market Insights")

# Main Application Functionality
def main_interface():
    with st.container():
        st.markdown("### Resume Analysis Interface")
        resume_input = st.text_area(
            "Paste professional resume content here:",
            height=200,
            placeholder="Enter resume text in plain English format..."
        )
        
        if st.button("Initiate Career Analysis"):
            if not resume_input.strip():
                st.warning("Please input valid resume content")
                return
            
            st.session_state.agent_state.update({
                "resume_text": resume_input,
                "selected_job": None
            })
            
            with st.spinner("Analyzing career profile..."):
                for event in app.stream(st.session_state.agent_state):
                    for key, value in event.items():
                        st.session_state.agent_state.update(value)
            
            display_analysis_results()

def display_analysis_results():
    st.markdown("---")
    with st.expander("Career Match Analysis Report", expanded=True):
        display_jobs_table(st.session_state.agent_state["jobs"])
        
        st.markdown("### Professional Development Recommendations")
        st.write(st.session_state.agent_state["current_response"])
    
    if st.session_state.agent_state.get("jobs"):
        st.markdown("---")
        with st.container():
            st.markdown("### Resume Customization Panel")
            selected_position = st.selectbox(
                "Select Target Position:",
                [job.get("Job Title", "Unspecified Role") for job in st.session_state.agent_state["jobs"]]
            )
            
            if st.button("Generate Position-Specific Optimization"):
                selected_job = next(
                    job for job in st.session_state.agent_state["jobs"] 
                    if job.get("Job Title") == selected_position
                )
                st.session_state.agent_state["selected_job"] = selected_job
                result = tailor_resume(st.session_state.agent_state)
                st.session_state.agent_state.update(result)
                
                with st.expander("Optimization Strategy"):
                    st.write(st.session_state.agent_state["current_response"])

# Execute main application
main_interface()
