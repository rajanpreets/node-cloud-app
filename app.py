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

# Custom professional styling
st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .stButton>button {background-color: #2c3e50; color: white; border-radius: 4px;}
        .stTextInput>div>div>input {border: 1px solid #2c3e50; border-radius: 4px;}
        .stSelectbox>div>div>select {border: 1px solid #2c3e50;}
        .header {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;}
        .sidebar .sidebar-content {background-color: #ecf0f1;}
    </style>
""", unsafe_allow_html=True)

# Initialize database with secrets
db_config = {
    'name': st.secrets.DB_NAME,
    'user': st.secrets.DB_USER,
    'password': st.secrets.DB_PASSWORD,
    'host': st.secrets.DB_HOST,
    'port': st.secrets.DB_PORT
}
init_db(db_config)

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
        pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
        if st.secrets.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=st.secrets.PINECONE_INDEX_NAME,
                dimension=int(st.secrets.EMBEDDING_DIMENSION),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=st.secrets.PINECONE_ENV)
            )
            while not pc.describe_index(st.secrets.PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
        return pc.Index(st.secrets.PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Initialize models
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets.GROQ_API_KEY)
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

# ... [Keep the rest of your original LangGraph nodes and workflow] ...

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
        
        st.markdown("#### Career Opportunity Matches")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("Apply Now"),
                "Description": "Position Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Data display error: {str(e)}")

# Authentication and main application logic
# ... [Keep your existing authentication and main application logic] ...

# Database initialization and management
# ... [Keep your existing database initialization code] ...
