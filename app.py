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

# Custom CSS for a professional look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4a4e69;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 8px;
    }
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    .stSidebar {
        background-color: #f8f9fa;
        padding: 20px;
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

# LangGraph nodes and workflow
def retrieve_jobs(state: AgentState):
    try:
        query_embedding = embedding_model.encode(state["resume_text"]).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="jobs"
        )
        jobs = [{
            'Job Title': match.metadata.get('title') or 'N/A',
            'Company Name': match.metadata.get('company') or 'N/A',
            'Location': match.metadata.get('location') or 'N/A',
            'Job Description': match.metadata.get('description') or '',
            'Job Link': match.metadata.get('link') or '#'
        } for match in results.matches if match.metadata]
        return {"jobs": jobs, "history": ["Retrieved jobs from Pinecone"]}
    except Exception as e:
        return {"error": str(e), "history": ["Job retrieval failed"]}

def generate_analysis(state: AgentState):
    if not state.get("jobs"):
        return {"current_response": "No jobs found for analysis", "history": ["Skipped analysis"]}
    
    job_texts = []
    for job in state["jobs"]:
        description = (job.get('Job Description') or '')[:300]
        job_texts.append(
            f"Title: {job.get('Job Title', 'N/A')}\n"
            f"Company: {job.get('Company Name', 'N/A')}\n"
            f"Location: {job.get('Location', 'N/A')}\n"
            f"Description: {description}"
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a career advisor. Analyze these jobs and give 3-5 brief recommendations:"),
        ("human", f"Resume content will follow this message. Here are matching jobs:\n\n{'\n\n'.join(job_texts)}\n\nProvide concise, actionable advice for the applicant.")
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
        job_desc = state['selected_job'].get('Job Description', '')
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a professional resume writer. Tailor this resume for the specific job."),
            ("human", f"Job Title: {state['selected_job']['Job Title']}\nJob Description:\n{job_desc}\n\nResume:\n{state['resume_text']}\n\nProvide specific suggestions to modify the resume. Focus on matching keywords and required skills.")
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
        st.warning("No matching jobs found")
        return
    
    try:
        jobs_df = pd.DataFrame([{
            "Title": job.get("Job Title", "N/A"),
            "Company": job.get("Company Name", "N/A"),
            "Location": job.get("Location", "N/A"),
            "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "N/A",
            "Link": job.get("Job Link", "#")
        } for job in jobs])
        
        st.markdown("### Best Matching Jobs")
        st.dataframe(
            jobs_df,
            column_config={
                "Link": st.column_config.LinkColumn("Apply Now"),
                "Description": "Job Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error displaying jobs: {str(e)}")

# Authentication UI
def authentication_ui():
    with st.container():
        st.markdown("## Login")
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

# Main UI
st.title("AI Career Assistant")

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
    with st.form("resume_form"):
        resume_text = st.text_area("Paste your resume text here and press Enter to analyze:", 
                                 height=200,
                                 help="Press Enter to submit after pasting your resume")
        submitted = st.form_submit_button("Analyze Resume")
        
    if submitted and resume_text:
        st.session_state.agent_state.update({
            "resume_text": resume_text.strip(),
            "selected_job": None
        })
        
        # Execute workflow
        for event in app.stream(st.session_state.agent_state):
            for key, value in event.items():
                st.session_state.agent_state.update(value)
        
        st.markdown("---")
        st.markdown("### Career Advisor Analysis")
        st.write(st.session_state.agent_state["current_response"])

        # Display job matches
        if st.session_state.agent_state.get("jobs"):
            st.markdown("---")
            display_jobs_table(st.session_state.agent_state["jobs"])
            
            # Tailoring interface
            st.markdown("---")
            st.markdown("### Resume Tailoring")
            
            job_titles = [job.get("Job Title", "Unknown Position") for job in st.session_state.agent_state["jobs"]]
            selected_title = st.selectbox("Select a job to tailor your resume for:", job_titles)
            
            if selected_title:
                selected_job = next(
                    (job for job in st.session_state.agent_state["jobs"] 
                    if job.get("Job Title") == selected_title),
                    None
                )
                if selected_job:
                    st.session_state.agent_state["selected_job"] = selected_job
                    
                    if st.button("Generate Tailored Suggestions"):
                        result = tailor_resume(st.session_state.agent_state)
                        st.session_state.agent_state.update(result)
                        
                        st.markdown("### Customization Suggestions")
                        st.write(st.session_state.agent_state["current_response"])

# Run main application
main_application()
