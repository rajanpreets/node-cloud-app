# app.py (Streamlit Main Application)
import streamlit as st
import psycopg2
import pandas as pd
import bcrypt
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator
import time
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        background-color: #4a4e69; 
        color: white; 
        border-radius: 4px; 
        padding: 0.5rem 1rem;
    }
    .stTextInput input, .stTextArea textarea {
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }
    .stDataFrame { 
        border: 1px solid #dee2e6; 
        border-radius: 4px; 
    }
    .header { color: #2b2d42; }
    .subheader { color: #4a4e69; }
    .success { color: #2a9d8f; }
    .error { color: #e76f51; }
</style>
""", unsafe_allow_html=True)

# Database Configuration
@contextmanager
def get_db_connection():
    """Get database connection from Streamlit secrets"""
    try:
        conn = psycopg2.connect(
            dbname=st.secrets.db.name,
            user=st.secrets.db.user,
            password=st.secrets.db.password,
            host=st.secrets.db.host,
            port=st.secrets.db.port
        )
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        st.error("Database connection error. Please try again later.")
        st.stop()

def init_db():
    """Initialize database tables"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(100) NOT NULL,
                        email VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id VARCHAR(40) PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        expires_at TIMESTAMP NOT NULL
                    );
                """)
                conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        st.error("Database initialization error. Please contact support.")
        st.stop()

# Authentication Functions
def create_user(username: str, password: str, email: str) -> int:
    """Create new user with validation"""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
        
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                cur.execute("""
                    INSERT INTO users (username, password_hash, email)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (username, password_hash, email))
                user_id = cur.fetchone()[0]
                conn.commit()
                return user_id
    except psycopg2.IntegrityError:
        raise ValueError("Username already exists")
    except Exception as e:
        logger.error(f"User creation failed: {str(e)}")
        raise RuntimeError("Registration failed")

def get_user_by_username(username: str):
    """Retrieve user by username"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, password_hash, email 
                    FROM users 
                    WHERE username = %s
                """, (username,))
                return cur.fetchone()
    except Exception as e:
        logger.error(f"User retrieval failed: {str(e)}")
        return None

# Application State Management
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

# Pinecone Configuration
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
        logger.error(f"Pinecone initialization failed: {str(e)}")
        st.error("Search service unavailable. Please try again later.")
        st.stop()

# Streamlit UI Components
def authentication_ui():
    """Professional authentication interface"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("<h2 class='header'>Career Analytics Platform</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Sign In", "Register"])

        with tab1:
            with st.form("Login"):
                st.markdown("<h3 class='subheader'>Account Login</h3>", unsafe_allow_html=True)
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In")

                if submit:
                    user = get_user_by_username(username)
                    if user and bcrypt.checkpw(password.encode(), user[2].encode()):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials", icon="⚠️")

        with tab2:
            with st.form("Register"):
                st.markdown("<h3 class='subheader'>Create Account</h3>", unsafe_allow_html=True)
                new_username = st.text_input("Username")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Register")

                if submit:
                    try:
                        create_user(new_username, new_password, new_email)
                        st.success("Account created successfully. Please sign in.")
                    except Exception as e:
                        st.error(str(e))

def main_interface():
    """Main application interface"""
    st.markdown("<h1 class='header'>Career Analytics Platform</h1>", unsafe_allow_html=True)
    
    # Initialize services
    index = init_pinecone()
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets.GROQ_API_KEY)

    # Application workflow
    resume_text = st.text_area("Paste your resume text:", height=200)
    
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            try:
                # Pinecone query
                query_embedding = embedding_model.encode(resume_text).tolist()
                results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                
                # Display results
                jobs = [match.metadata for match in results.matches if match.metadata]
                if jobs:
                    df = pd.DataFrame([{
                        "Title": j.get("Job Title"),
                        "Company": j.get("Company Name"),
                        "Location": j.get("Location"),
                        "Description": j.get("Job Description", "")[:100] + "..."
                    } for j in jobs])
                    
                    st.markdown("<h3 class='subheader'>Recommended Positions</h3>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No matching positions found")
                
                # Generate analysis
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Provide concise career analysis based on resume and job matches:"),
                    ("human", f"Resume: {resume_text}\n\nJobs: {jobs}")
                ])
                analysis = llm.invoke(prompt.format_messages()).content
                st.markdown("<h3 class='subheader'>Career Analysis</h3>", unsafe_allow_html=True)
                st.write(analysis)

            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                st.error("Analysis failed. Please try again.")

# Main App Execution
def main():
    st.set_page_config(page_title="Career Analytics Platform", layout="wide")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        init_db()

    if not st.session_state.logged_in:
        authentication_ui()
    else:
        with st.sidebar:
            st.markdown(f"<p class='subheader'>Welcome, {st.session_state.username}</p>", unsafe_allow_html=True)
            if st.button("Sign Out"):
                st.session_state.clear()
                st.rerun()
        main_interface()

if __name__ == "__main__":
    main()
