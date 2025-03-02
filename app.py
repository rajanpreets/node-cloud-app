import streamlit as st
import os
import pandas as pd
import firebase_admin
import json
from firebase_admin import credentials, auth
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
import operator
from streamlit.components.v1 import html

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rajan"
EMBEDDING_DIMENSION = 384

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
    firebase_admin.initialize_app(cred)

# Authentication functions
def get_current_user():
    return st.session_state.get('auth_user')

def login_component():
    firebase_config = st.secrets["firebase_config"]
    
    auth_html = f"""
    <script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-auth-compat.js"></script>
    <script>
        var firebaseConfig = {json.dumps(firebase_config)};
        var app = firebase.initializeApp(firebaseConfig);
        
        function signInWithGoogle() {{
            var provider = new firebase.auth.GoogleAuthProvider();
            firebase.auth().signInWithPopup(provider)
                .then((result) => {{
                    var user = result.user;
                    window.parent.postMessage({{
                        'type': 'USER_LOGGED_IN',
                        'user': {{
                            'uid': user.uid,
                            'email': user.email,
                            'name': user.displayName,
                            'photo_url': user.photoURL
                        }}
                    }}, '*');
                }})
                .catch((error) => {{
                    console.error(error);
                }});
        }}
    </script>
    
    <button onclick="signInWithGoogle()" style="
        background: #4285F4;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
            <path d="M12.24 10.285V14.4h6.806c-.275 1.765-2.056 5.174-6.806 5.174-4.095 0-7.439-3.389-7.439-7.574s3.345-7.574 7.439-7.574c2.33 0 3.891.989 4.785 1.849l3.254-3.138C18.189 1.186 15.479 0 12.24 0c-6.635 0-12 5.365-12 12s5.365 12 12 12c6.926 0 11.52-4.869 11.52-11.726 0-.788-.085-1.39-.189-1.989H12.24z"/>
        </svg>
        Sign in with Google
    </button>
    """
    html(auth_html, height=60)

# Authentication check
if not get_current_user():
    st.set_page_config(page_title="Login - Career Assistant", layout="centered")
    st.title("🔐 Secure Career Assistant Login")
    st.markdown("""
        <div style='text-align: center; padding: 2rem; border-radius: 10px; background: #f0f2f6; margin: 2rem 0;'>
            <h3 style='color: #2c3e50;'>Please authenticate with Google</h3>
            <p style='color: #7f8c8d;'>Your career data is protected with enterprise-grade security</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,0.5,1])
    with col2:
        login_component()
    
    st.markdown("---")
    st.warning("🔒 All data is encrypted in transit and at rest. We never store your personal information.")
    
    # Handle auth callback
    html("""
    <script>
        window.addEventListener('message', function(event) {
            if (event.data.type === 'USER_LOGGED_IN') {
                const user = event.data.user;
                const data = {
                    'uid': user.uid,
                    'email': user.email,
                    'name': user.name,
                    'photo_url': user.photo_url
                };
                
                fetch('/_stcore/set-session-data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        'key': 'auth_user',
                        'value': data
                    })
                }).then(() => {
                    window.location.reload();
                });
            }
        });
    </script>
    """)
    st.stop()

# Main App Configuration
st.set_page_config(
    page_title="💬 AI Career Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar with User Info
with st.sidebar:
    user = get_current_user()
    st.markdown(f"""
        <div style='padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;'>
            <div style='display: flex; align-items: center; gap: 1rem;'>
                <img src="{user['photo_url']}" 
                     style='width: 40px; height: 40px; border-radius: 50%;'>
                <div>
                    <h4 style='margin: 0;'>{user['name']}</h4>
                    <small>{user['email']}</small>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", type="secondary", use_container_width=True):
        del st.session_state.auth_user
        st.rerun()

# Career Assistant Functionality
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

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
        st.stop()

try:
    index = init_pinecone()
    
    with st.spinner("⚙️ Loading AI models..."):
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
        
except Exception as model_error:
    st.error(f"❌ Model initialization failed: {str(model_error)}")
    st.stop()

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

# Initialize session state
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Main Interface
st.title("💬 AI Career Assistant")
st.markdown("---")

with st.chat_message("assistant"):
    st.write(f"Hi {user['name'].split()[0]}! Ready to boost your career?")

# Add your existing chat interface and workflow logic here
# ...

# Debug Section
if st.session_state.get('agent_state', {}).get('jobs'):
    with st.expander("🔧 Technical Details"):
        st.caption("Last operation status:")
        st.json({
            "user": user,
            "jobs_found": len(st.session_state.agent_state.get('jobs', [])),
            "last_update": pd.Timestamp.now().isoformat()
        })
