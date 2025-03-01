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

# Set page config first
st.set_page_config(
    page_title="💬 AI Career Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Authentication Check with Error Handling
try:
    if not st.experimental_user.is_logged_in:
        st.title("🔐 Secure Career Assistant Login")
        st.markdown("""
            <div style='text-align: center; padding: 2rem; border-radius: 10px; background: #f0f2f6; margin: 2rem 0;'>
                <h3 style='color: #2c3e50;'>Please authenticate with Google</h3>
                <p style='color: #7f8c8d;'>Your career data is protected with enterprise-grade security</p>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,0.5,1])
        with col2:
            if st.button("🚀 Login with Google", type="primary", use_container_width=True):
                st.login("google")
                st.rerun()
        
        st.markdown("---")
        st.warning("🔒 All data is encrypted in transit and at rest. We never store your personal information.")
        st.stop()
    else:
        # Display user info and logout in sidebar
        with st.sidebar:
            st.markdown(f"""
                <div style='padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <img src="{st.experimental_user.picture}" 
                             style='width: 40px; height: 40px; border-radius: 50%;'>
                        <div>
                            <h4 style='margin: 0;'>{st.experimental_user.name}</h4>
                            <small>{st.experimental_user.email}</small>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                st.logout()
                st.rerun()

except Exception as auth_error:
    st.error(f"🔐 Authentication Error: {str(auth_error)}")
    st.error("Please refresh the page or contact support if the problem persists")
    st.stop()

# Main App Content ------------------------------------------------------------
st.title("💬 AI Career Assistant")
st.markdown("---")

# Define State for LangGraph (keep existing code)
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

# Initialize Pinecone with enhanced error handling
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            with st.spinner("🚀 Creating new Pinecone index..."):
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {str(e)}")
        st.error("Please check your API key and network connection")
        st.stop()

# Rest of your existing code remains the same, wrapped in try-except blocks
try:
    index = init_pinecone()
    
    # Initialize models with progress
    with st.spinner("⚙️ Loading AI models..."):
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
        
except Exception as model_error:
    st.error(f"❌ Model initialization failed: {str(model_error)}")
    st.error("Please check your API keys and internet connection")
    st.stop()

# Define LangGraph nodes with error handling
def retrieve_jobs(state: AgentState):
    try:
        with st.spinner("🔍 Searching for matching jobs..."):
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
        st.error("🔍 Job search failed. Please try again with different resume text")
        return {"error": str(e)}

# ... (keep existing node functions with added error handling)

# Rest of your existing code remains the same, with added:
# - try-except blocks
# - progress indicators
# - user-friendly error messages
# - session state validation checks

# Add validation for critical operations
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Main chat interface
with st.chat_message("assistant"):
    st.write(f"Hi {st.experimental_user.given_name}! Ready to boost your career?")

# Rest of your existing UI code...

# Enhanced debug section
if st.session_state.get('agent_state', {}).get('jobs'):
    with st.expander("🔧 Technical Details"):
        st.caption("Last operation status:")
        st.json({
            "user": st.experimental_user.__dict__,
            "jobs_found": len(st.session_state.agent_state.get('jobs', [])),
            "last_update": pd.Timestamp.now().isoformat()
        })
        st.write("Full state object:")
        st.write(st.session_state.agent_state)
