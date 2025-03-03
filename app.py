import asyncio
import sys
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
from contextlib import contextmanager
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login

# Asyncio event loop configuration
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Custom professional styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {background-color: #2c3e50; color: white; border-radius: 4px;}
        .stTextInput>div>div>input {border: 1px solid #2c3e50; border-radius: 4px;}
        .stSelectbox>div>div>select {border: 1px solid #2c3e50;}
        .header {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;}
        .sidebar .sidebar-content {background-color: #ecf0f1;}
        .dataframe {box-shadow: 0 1px 3px rgba(0,0,0,0.12);}
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

# LangGraph nodes
def retrieve_jobs(state: AgentState):
    try:
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
        return {"error": str(e), "history": ["Job retrieval failed"]}

def generate_analysis(state: AgentState):
    if not state.get("jobs"):
        return {"current_response": "No jobs found for analysis", "history": ["Skipped analysis"]}
    
    job_texts = "\n\n".join([
        f"Title: {job.get('Job Title')}\nCompany: {job.get('Company Name')}\nDescription: {job.get('Job Description', '')[:300]}"
        for job in state["jobs"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze these job opportunities and provide career development recommendations:"),
        ("human", f"Resume content:\n{state['resume_text']}\n\nMatching positions:\n{job_texts}")
    ])
    
    try:
        analysis = llm.invoke(prompt.format_messages()).content
        return {"current_response": analysis, "history": ["Generated career analysis"]}
    except Exception as e:
        return {"error": str(e), "history": ["Analysis generation failed"]}

def tailor_resume(state: AgentState):
    if not state.get("selected_job"):
        return {"current_response": "No job selected for tailoring", "history": ["Skipped tailoring"]}
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide resume customization suggestions for this position:"),
            ("human", f"Position: {state['selected_job']['Job Title']}\nRequirements:\n{state['selected_job'].get('Job Description', '')}\n\nResume Content:\n{state['resume_text']}")
        ])
        response = llm.invoke(prompt.format_messages())
        return {"current_response": response.content, "history": ["Generated resume suggestions"]}
    except Exception as e:
        return {"error": str(e), "history": ["Tailoring failed"]}

# Build workflow
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

# Context manager for event loop
@contextmanager
def st_redirect():
    orig_loop = asyncio.get_event_loop()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield
    finally:
        asyncio.set_event_loop(orig_loop)

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
                "Link": st.column_config.LinkColumn("Application Link"),
                "Description": "Position Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Data display error: {str(e)}")

# Authentication UI
def authentication_ui():
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/150x50.png?text=CareerAI", use_column_width=True)
        
        with col2:
            tab1, tab2 = st.tabs(["Secure Sign In", "New Registration"])
            
            with tab1:
                with st.form("Secure Login"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Authenticate"):
                        user = get_user_by_username(username)
                        if user and verify_password(user[2], password):
                            update_last_login(username)
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

            with tab2:
                with st.form("New Account"):
                    new_user = st.text_input("Create Username")
                    new_email = st.text_input("Email Address")
                    new_pass = st.text_input("Create Password", type="password")
                    if st.form_submit_button("Register"):
                        try:
                            create_user(new_user, new_pass, new_email)
                            st.success("Account created successfully")
                        except Exception as e:
                            st.error(str(e))

# Main Application
def main_interface():
    st.set_page_config(page_title="AI Career Platform", layout="wide")
    st.markdown("<h1 class='header'>Professional Career Optimization Platform</h1>", unsafe_allow_html=True)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.agent_state = {
            "resume_text": "",
            "jobs": [],
            "history": [],
            "current_response": "",
            "selected_job": None
        }

    if not st.session_state.logged_in:
        authentication_ui()
        st.stop()

    with st.sidebar:
        st.markdown(f"**Active Session:** {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Navigation**")
        st.markdown("- Career Analysis\n- Resume Optimization\n- Market Insights")

    with st.container():
        st.markdown("### Professional Profile Analysis")
        resume_input = st.text_area(
            "Input Resume Content:",
            height=250,
            placeholder="Paste your professional resume text here..."
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
    with st.expander("Career Match Report", expanded=True):
        display_jobs_table(st.session_state.agent_state["jobs"])
        
        st.markdown("### Professional Development Recommendations")
        st.write(st.session_state.agent_state["current_response"])
    
    if st.session_state.agent_state.get("jobs"):
        st.markdown("---")
        with st.container():
            st.markdown("### Resume Customization")
            selected_position = st.selectbox(
                "Select Target Position:",
                [job.get("Job Title", "Unspecified Role") for job in st.session_state.agent_state["jobs"]]
            )
            
            if st.button("Generate Optimization Strategy"):
                selected_job = next(
                    job for job in st.session_state.agent_state["jobs"] 
                    if job.get("Job Title") == selected_position
                )
                st.session_state.agent_state["selected_job"] = selected_job
                result = tailor_resume(st.session_state.agent_state)
                st.session_state.agent_state.update(result)
                
                with st.expander("Customization Recommendations"):
                    st.write(st.session_state.agent_state["current_response"])

if __name__ == "__main__":
    with st_redirect():
        main_interface()
