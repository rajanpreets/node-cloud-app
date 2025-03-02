import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login

# Initialize database before other operations
init_db()

# Define State for LangGraph
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

# Initialize Pinecone with Streamlit secrets
def init_pinecone():
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index_name = st.secrets["PINECONE_INDEX_NAME"]
        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=st.secrets["EMBEDDING_DIMENSION"],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Initialize models
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(
        model="llama3-8b-8192", 
        temperature=0, 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"❌ Model initialization failed: {str(e)}")
    st.stop()

# ... [Keep all your existing LangGraph nodes and workflow code the same] ...

# Authentication state management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Authentication UI components
def login_form():
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            user = get_user_by_username(username)
            if user and verify_password(user[2], password):
                st.session_state.authenticated = True
                st.session_state.current_user = user
                update_last_login(username)
                st.rerun()
            else:
                st.error("Invalid credentials")

def register_form():
    with st.form("Register"):
        username = st.text_input("Username", max_chars=50)
        password = st.text_input("Password", type="password")
        email = st.text_input("Email", max_chars=100)
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if get_user_by_username(username):
                st.error("Username already exists")
            else:
                try:
                    create_user(username, password, email)
                    st.success("Registration successful! Please login")
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")

def logout():
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }
    st.rerun()

# Main app logic
if not st.session_state.authenticated:
    st.title("🔒 AI Career Assistant")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login_form()
    with tab2:
        register_form()
    st.stop()

# Main application UI
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title(f"💬 AI Career Assistant - Welcome {st.session_state.current_user[1]}")

# Logout button
st.sidebar.button("Logout", on_click=logout)
st.sidebar.markdown(f"""
**User Info**
- Username: {st.session_state.current_user[1]}
- Email: {st.session_state.current_user[3]}
- Last Login: {st.session_state.current_user[5]}
""")

# ... [Keep your existing Streamlit UI components the same] ...
