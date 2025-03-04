import streamlit as st
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator
import time
from database import init_db, get_user_by_username, verify_password, update_last_login

# Set page config
st.set_page_config(
    page_title="AI Career Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove Streamlit UI elements
st.markdown("""
<style>
    /* Hide header, footer, and menu */
    header { visibility: hidden; }
    .stApp > header { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    /* Hide deploy button */
    .stDeployButton { display: none; }
    /* Hide 'Manage app' button */
    [data-testid="manage-app-button"] { display: none; }
    /* Hide GitHub icon */
    [data-testid="stHeader"] [data-testid="stDecoration"] { display: none; }
    /* Hide three-dot menu */
    [data-testid="stActionButton"] { display: none; }
    /* Hide pen symbol */
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Rest of your code...

# Professional CSS styling
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a2833;
        transform: translateY(-1px);
    }
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 10px;
        font-size: 14px;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown h1 {
        color: #2c3e50;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 0.5rem;
    }
    .stMarkdown h2 {
        color: #34495e;
        margin-top: 1.5rem;
    }
    .stSidebar {
        background-color: #f8f9fa;
        padding: 20px;
        border-right: 1px solid #e0e0e0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# Get secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
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
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

index = init_pinecone()

# Initialize models
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

# LangGraph nodes and workflow (unchanged)
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
        return {"error": str(e), "history": ["Analysis generation failed"]}

def tailor_resume(state: AgentState):
    if not state.get("selected_job"):
        return {"current_response": "No job selected for tailoring", "history": ["Skipped tailoring"]}
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a professional resume writer. Tailor this resume for the specific job."),
            ("human", f"Job Title: {state['selected_job']['Job Title']}\nJob Description:\n{state['selected_job'].get('Job Description', '')}\n\nResume:\n{state['resume_text']}\n\nProvide specific suggestions to modify the resume. Focus on matching keywords and required skills.")
        ])
        response = llm.invoke(prompt.format_messages())
        return {"current_response": response.content, "history": ["Generated tailored resume suggestions"]}
    except Exception as e:
        return {"error": str(e), "history": ["Tailoring failed"]}

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

# Streamlit UI components
def display_jobs_table(jobs):
    if not jobs:
        st.warning("No matching positions found")
        return
    
    try:
        jobs_df = pd.DataFrame([{
            "Title": job.get("Job Title", "Not Available"),
            "Company": job.get("Company Name", "Not Available"),
            "Location": job.get("Location", "Not Specified"),
            "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "Description not available",
            "Link": job.get("Job Link", "#")
        } for job in jobs])
        
        st.markdown("### Matching Opportunities")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("View Position"),
                "Description": "Position Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error displaying positions: {str(e)}")

# Authentication UI
def authentication_ui():
    with st.container():
        st.markdown("## System Access")
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate")
            
            if submit:
                user = get_user_by_username(username)
                if user and verify_password(user[2], password):
                    update_last_login(username)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Authentication failed")

# Main UI
st.title("AI Career Platform")

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
    st.write(f"Active session: {st.session_state.username}")

# Main Application Functionality
def main_application():
    with st.form("resume_analysis"):
        resume_text = st.text_area("Input professional summary or resume content:", 
                                 height=250,
                                 placeholder="Paste your resume text here...")
        submitted = st.form_submit_button("Analyze Profile")
        
    if submitted and resume_text:
        st.session_state.agent_state.update({
            "resume_text": resume_text,
            "selected_job": None
        })
        
        # Execute workflow
        for event in app.stream(st.session_state.agent_state):
            for key, value in event.items():
                st.session_state.agent_state.update(value)
        
        st.markdown("---")
        with st.container():
            st.markdown("### Career Strategy Analysis")
            st.write(st.session_state.agent_state["current_response"])

    # Tailoring interface
    if st.session_state.agent_state.get("jobs"):
        st.markdown("---")
        display_jobs_table(st.session_state.agent_state["jobs"])
        
        st.markdown("---")
        with st.container():
            st.markdown("### Resume Optimization")
            
            job_titles = [job.get("Job Title", "Unspecified Position") for job in st.session_state.agent_state["jobs"]]
            selected_title = st.selectbox("Select target position:", job_titles)
            
            if selected_title:
                selected_job = next(
                    job for job in st.session_state.agent_state["jobs"] 
                    if job.get("Job Title") == selected_title
                )
                st.session_state.agent_state["selected_job"] = selected_job
                
                if st.button("Generate Customization Recommendations"):
                    result = tailor_resume(st.session_state.agent_state)
                    st.session_state.agent_state.update(result)
                    
                    st.markdown("### Professional Enhancement Suggestions")
                    st.write(st.session_state.agent_state["current_response"])

# Run main application
main_application()
