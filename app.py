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
import logging
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login, verify_payment, is_subscription_active
from paypal_client import PayPalClient

# Initialize first
st.set_page_config(
    page_title="💼 Premium Career Assistant",
    layout="wide",
    page_icon="💼"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
init_db()
paypal = PayPalClient()

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .payment-button {
        background-color: #003087 !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        display: block;
        margin: 1rem auto;
        width: fit-content;
    }
    .subscription-status {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .subscription-active {
        background-color: #e8f5e9;
        border: 1px solid #2e7d32;
    }
    .subscription-inactive {
        background-color: #ffebee;
        border: 1px solid #c62828;
    }
    .stDataFrame { 
        border: 1px solid #dee2e6; 
        border-radius: 4px; 
    }
</style>
""", unsafe_allow_html=True)

def handle_payment_callback():
    """Handle PayPal payment redirects"""
    params = st.query_params
    if 'token' in params:
        try:
            order_id = params['token']
            if paypal.verify_payment(order_id):
                # Complete registration after successful payment
                if 'registration_data' in st.session_state:
                    user_data = st.session_state.registration_data
                    try:
                        create_user(
                            user_data['username'],
                            user_data['password'],
                            user_data['email'],
                            order_id
                        )
                        if verify_payment(order_id):
                            st.success("✅ Payment verified! Account activated.")
                            st.session_state.registration_success = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Account activation failed: {str(e)}")
                else:
                    st.error("❌ Registration data missing")
            else:
                st.error("❌ Payment verification failed")
        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            st.error("❌ Payment processing failed")
    elif 'cancelled' in params:
        st.warning("⚠️ Payment cancelled")

def authentication_ui():
    """Authentication interface with payment handling"""
    handle_payment_callback()
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                try:
                    user = get_user_by_username(username)
                    if user and verify_password(user[2], password):
                        if is_subscription_active(username):
                            update_last_login(username)
                            st.session_state.update({
                                "logged_in": True,
                                "username": username
                            })
                            st.rerun()
                        else:
                            st.error("Subscription expired or inactive")
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    logger.error(f"Login error: {str(e)}")
                    st.error("Login failed")

    with register_tab:
        with st.form("Register"):
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            if st.form_submit_button("Create Account"):
                try:
                    # Create PayPal order first
                    order = paypal.create_order(10.00)
                    if not order:
                        raise ValueError("Failed to create payment order")
                    
                    # Store registration data temporarily
                    st.session_state.registration_data = {
                        'username': new_username,
                        'email': new_email,
                        'password': new_password
                    }
                    
                    # Get PayPal approval URL
                    approval_url = next(
                        link['href'] for link in order['links'] 
                        if link['rel'] == 'approve'
                    )
                    
                    # Show payment button
                    st.markdown(
                        f"<a href='{approval_url}' class='payment-button' target='_blank'>"
                        "🔒 Complete Secure $10 Payment via PayPal</a>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    logger.error(f"Registration error: {str(e)}")
                    st.error(f"Registration failed: {str(e)}")

def display_jobs_table(jobs):
    """Display jobs in a dataframe"""
    if not jobs:
        st.warning("No jobs found")
        return
    
    try:
        jobs_df = pd.DataFrame([{
            "Title": job.get("Job Title", "N/A"),
            "Company": job.get("Company Name", "N/A"),
            "Location": job.get("Location", "N/A"),
            "Description": (job.get("Job Description", "")[:150] + "...") if job.get("Job Description") else "N/A",
            "Link": job.get("Job Link", "#")
        } for job in jobs])
        
        st.markdown("### 📌 Matching Jobs")
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

def main_application():
    """Main application interface"""
    # Initialize AI services
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = "rajan"
    
    # Ensure index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    index = pc.Index(index_name)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

    # Define LangGraph State
    class AgentState(TypedDict):
        resume_text: str
        jobs: List[dict]
        history: Annotated[List[str], operator.add]
        current_response: str
        selected_job: dict

    # Define LangGraph nodes
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

    # Build LangGraph workflow
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

    # Main application UI
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
                result = tailor_resume(st.session_state.agent_state)
                st.session_state.agent_state.update(result)
                
                st.markdown("### 📝 Customization Suggestions")
                st.write(st.session_state.agent_state["current_response"])

    # Debug section
    if st.session_state.agent_state.get('jobs'):
        with st.expander("🔧 Debug Information"):
            st.write("Agent State:", st.session_state.agent_state)
            st.write("History:", st.session_state.agent_state.get("history", []))

def main():
    """Main application flow"""
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            "logged_in": False,
            "agent_state": {
                "resume_text": "",
                "jobs": [],
                "history": [],
                "current_response": "",
                "selected_job": None
            }
        })

    if not st.session_state.logged_in:
        authentication_ui()
        st.stop()

    st.title("💼 Premium Career Assistant")
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        st.write(f"Welcome, {st.session_state.username}")
    
    main_application()

if __name__ == "__main__":
    main()
