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
from jose import JWTError, jwt

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")
OIDC_ISSUER = os.getenv("OIDC_ISSUER")
INDEX_NAME = "rajan"
EMBEDDING_DIMENSION = 384

# Configure OIDC
oidc_config = {
    "client_id": OIDC_CLIENT_ID,
    "client_secret": OIDC_CLIENT_SECRET,
    "authorization_params": {"scope": "openid profile email"},
    "issuer": OIDC_ISSUER,
    "token_endpoint_auth_method": "client_secret_post"
}

def validate_token(token: str):
    try:
        jwks_url = f"{OIDC_ISSUER}/.well-known/jwks.json"
        jwks = requests.get(jwks_url).json()
        header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        if rsa_key:
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=OIDC_CLIENT_ID,
                issuer=OIDC_ISSUER
            )
            return payload
        return None
    except JWTError:
        return None

# Authentication UI
if 'auth' not in st.session_state:
    st.session_state.auth = None

with st.sidebar:
    if st.session_state.auth:
        if st.button("Logout"):
            st.session_state.auth = None
            st.session_state.agent_state = None
            st.rerun()
    else:
        if st.login(config=oidc_config):
            token = st.experimental_get_query_params().get("token")[0]
            user_info = validate_token(token)
            if user_info:
                st.session_state.auth = {
                    "token": token,
                    "user_info": user_info
                }
                st.rerun()

# Restrict access to authenticated users
if not st.session_state.auth:
    st.warning("Please login to access the Career Assistant")
    st.stop()

# Define State for LangGraph (remaining code same as before)
class AgentState(TypedDict):
    resume_text: str
    jobs: List[dict]
    history: Annotated[List[str], operator.add]
    current_response: str
    selected_job: dict

# Initialize Pinecone and models (keep existing implementation)

# LangGraph workflow setup (keep existing implementation)

# Streamlit UI components (keep existing display_jobs_table implementation)

# Modified UI with user greeting
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title(f"💬 AI Career Assistant · Welcome {st.session_state.auth['user_info'].get('name', 'User')}!")

# Initialize session state
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "resume_text": "",
        "jobs": [],
        "history": [],
        "current_response": "",
        "selected_job": None
    }

# Existing chat interface (keep implementation the same)
# ...
# [Rest of your existing code remains unchanged]
