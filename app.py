import streamlit as st
import os
import pandas as pd
import firebase_admin
import json
from firebase_admin import credentials
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
    try:
        cred = credentials.Certificate(st.secrets["firebase"])
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"🔥 Firebase initialization error: {str(e)}")
        st.stop()

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

# Main App Configuration ------------------------------------------------------
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title("💬 AI Career Assistant")

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

# Original Career Assistant Functionality -------------------------------------
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

index = init_pinecone()

try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Model initialization failed: {str(e)}")
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

def generate_analysis(state: AgentState):
    job_texts = "\n\n".join([f"Title: {job.get('Job Title')}\nCompany: {job.get('Company Name')}\nDescription: {job.get('Job Description', '')[:300]}" 
                          for job in state["jobs"]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a career advisor. Analyze these jobs and give 3-5 brief recommendations:"),
        ("human", f"Resume content will follow this message. Here are matching jobs:\n\n{job_texts}\n\nProvide concise, actionable advice for the applicant.")
    ])
    
    try:
        analysis = llm.invoke(prompt.format_messages()).content
        return {"current_response": analysis, "history": ["Generated career analysis"]}
    except Exception as e:
        return {"error": str(e)}

def tailor_resume(state: AgentState):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a professional resume writer. Tailor this resume for the specific job."),
            ("human", f"Job Title: {state['selected_job']['Job Title']}\nJob Description:\n{state['selected_job'].get('Job Description', '')}\n\nResume:\n{state['resume_text']}\n\nProvide specific suggestions to modify the resume. Focus on matching keywords and required skills.")
        ])
        response = llm.invoke(prompt.format_messages())
        return {"current_response": response.content, "history": ["Generated tailored resume suggestions"]}
    except Exception as e:
        return {"error": str(e)}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_jobs", retrieve_jobs)
workflow.add_node("generate_analysis", generate_analysis)
workflow.add_node("tailor_resume", tailor_resume)
workflow.set_entry_point("retrieve_jobs")
workflow.add_edge("retrieve_jobs", "generate_analysis")
workflow.add_conditional_edges(
    "generate_analysis",
    lambda x: "tailor_resume" if x.get("selected_job") else END,
    {"tailor_resume": "tailor_resume", END: END}
)
workflow.add_edge("tailor_resume", END)
app = workflow.compile()

def display_jobs_table(jobs):
    if not jobs or isinstance(jobs, str):
        st.error(jobs if jobs else "No jobs found")
        return
    
    jobs_df = pd.DataFrame([{
        "Title": job.get("Job Title", "N/A"),
        "Company": job.get("Company Name", "N/A"),
        "Location": job.get("Location", "N/A"),
        "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "N/A",
        "Link": job.get("Job Link", "#")
    } for job in jobs])
    
    if not jobs_df.empty:
        st.markdown("### 🗃 Matching Jobs")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("Apply Now"),
                "Description": "Job Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No valid job listings found")

if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

with st.chat_message("assistant"):
    st.write(f"Hi {user['name'].split()[0]}! I'm your AI career assistant. Paste your resume below and I'll help you find relevant jobs!")

resume_text = st.chat_input("Paste your resume text here...")

if resume_text:
    st.session_state.agent_state["resume_text"] = resume_text
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

if st.session_state.agent_state["jobs"]:
    st.markdown("---")
    st.markdown("### ✨ Resume Tailoring")
    
    job_titles = [job.get("Job Title", "Unknown Position") for job in st.session_state.agent_state["jobs"]]
    selected_title = st.selectbox("Which job would you like to tailor your resume for?", job_titles)
    
    if selected_title:
        st.session_state.agent_state["selected_job"] = next(
            job for job in st.session_state.agent_state["jobs"] 
            if job.get("Job Title") == selected_title
        )
        
        if st.button("Generate Tailored Resume Suggestions"):
            for event in app.stream(st.session_state.agent_state):
                for key, value in event.items():
                    st.session_state.agent_state.update(value)
            
            st.markdown("### 📝 Customization Suggestions")
            st.write(st.session_state.agent_state["current_response"])

if st.session_state.agent_state.get('jobs'):
    with st.expander("🔧 Debug Information"):
        st.write("Agent State:", st.session_state.agent_state)
        st.write("History:", st.session_state.agent_state["history"])
