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
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login

# Custom professional styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        background-color: #2b2d42; 
        color: white; 
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4a4e69;
        transform: scale(1.05);
    }
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 0.5rem;
    }
    .stDataFrame {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-radius: 8px;
    }
    .header { color: #2b2d42; }
    .subheader { color: #4a4e69; }
    .success { color: #2a9d8f; }
    .error { color: #e76f51; }
</style>
""", unsafe_allow_html=True)

class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

def init_pinecone():
    try:
        pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
        index_name = "career-index"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        return pc.Index(index_name)
    except Exception as e:
        st.error("Search service temporarily unavailable. Please try again later.")
        st.stop()

def authentication_ui():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1 class='header'>Career Analytics Platform</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Sign In", "Register"])

        with tab1:
            with st.form("Login"):
                st.markdown("<h3 class='subheader'>Secure Sign In</h3>", unsafe_allow_html=True)
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
                        st.error("Invalid credentials", icon="⚠️")

        with tab2:
            with st.form("Register"):
                st.markdown("<h3 class='subheader'>New Account</h3>", unsafe_allow_html=True)
                new_username = st.text_input("Username")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Create Account")

                if submit:
                    try:
                        create_user(new_username, new_password, new_email)
                        st.success("Account created successfully. Please sign in.")
                    except Exception as e:
                        st.error(str(e))

def main_interface():
    st.markdown("<h2 class='header'>Career Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # Service initialization
    index = init_pinecone()
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets.GROQ_API_KEY)

    # Resume input
    resume_text = st.text_area("Paste your professional resume:", height=250)
    
    if st.button("Analyze Career Opportunities"):
        with st.spinner("Processing your resume..."):
            try:
                # Query processing
                query_embedding = embedding_model.encode(resume_text).tolist()
                results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                
                # Display results
                jobs = [match.metadata for match in results.matches if match.metadata]
                if jobs:
                    df = pd.DataFrame([{
                        "Position": j.get("Job Title"),
                        "Organization": j.get("Company Name"),
                        "Location": j.get("Location"),
                        "Summary": (j.get("Job Description", "")[:150] + "...")
                    } for j in jobs])
                    
                    st.markdown("<h3 class='subheader'>Recommended Opportunities</h3>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    
                    # Generate analysis
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Provide professional career analysis based on resume and opportunities:"),
                        ("human", f"Resume: {resume_text}\n\nOpportunities: {jobs}")
                    ])
                    analysis = llm.invoke(prompt.format_messages()).content
                    st.markdown("<h3 class='subheader'>Strategic Career Analysis</h3>", unsafe_allow_html=True)
                    st.write(analysis)
                else:
                    st.info("No matching opportunities found")

            except Exception as e:
                st.error("Analysis service currently unavailable. Please try again later.")

def main():
    st.set_page_config(page_title="Career Analytics Platform", layout="wide")
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        authentication_ui()
    else:
        with st.sidebar:
            st.markdown(f"<div class='subheader'>Welcome back, {st.session_state.username}</div>", 
                       unsafe_allow_html=True)
            if st.button("Sign Out"):
                st.session_state.clear()
                st.rerun()
        main_interface()

if __name__ == "__main__":
    main()
