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
from datetime import datetime
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
    .payment-button:hover {
        background-color: #001f5e !important;
        color: white !important;
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
</style>
""", unsafe_allow_html=True)

# Payment verification handler
def handle_payment_callback():
    query_params = st.experimental_get_query_params()
    if 'token' in query_params:
        try:
            order_id = query_params['token'][0]
            payment_data = paypal.capture_payment(order_id)
            
            if payment_data and payment_data.get('status') == 'COMPLETED':
                payment_id = payment_data['id']
                if verify_payment(payment_id):
                    st.success("✅ Payment verified! You can now login.")
                    st.balloons()
                else:
                    st.error("❌ Payment verification failed. Contact support.")
            else:
                st.error("❌ Payment not completed. Please try again.")
                
        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            st.error("❌ Payment processing failed. Please try again.")

# Enhanced authentication UI
def authentication_ui():
    handle_payment_callback()
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = get_user_by_username(username)
                if user and verify_password(user[2], password):
                    if is_subscription_active(username):
                        update_last_login(username)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Subscription expired or inactive. Please renew.")
                else:
                    st.error("Invalid credentials")

    with register_tab:
        with st.form("Register"):
            new_username = st.text_input("New Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("New Password", type="password", key="reg_password")
            submit = st.form_submit_button("Create Account")
            
            if submit:
                try:
                    # Create PayPal order
                    order = paypal.create_order(10.00)
                    if not order:
                        raise Exception("Failed to create payment order")
                    
                    approval_url = next(
                        (link['href'] for link in order['links'] 
                        if link['rel'] == 'approve'), None
                    )
                    
                    if approval_url:
                        # Store registration data temporarily
                        st.session_state.registration_data = {
                            'username': new_username,
                            'email': new_email,
                            'password': new_password,
                            'payment_id': order['id']
                        }
                        
                        # Show payment button
                        st.markdown(
                            f"<a href='{approval_url}' class='payment-button' target='_blank'>"
                            "🔒 Complete Secure $10 Payment via PayPal</a>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Failed to get payment approval URL")
                        
                except Exception as e:
                    logger.error(f"Registration error: {str(e)}")
                    st.error(f"Registration failed: {str(e)}")

# Main application UI components
def display_subscription_status():
    user = get_user_by_username(st.session_state.username)
    if user and user[5]:  # subscription_end
        end_date = user[5].strftime("%Y-%m-%d")
        status_class = "subscription-active" if datetime.now() < user[5] else "subscription-inactive"
        status_text = f"Active until {end_date}" if datetime.now() < user[5] else "Expired"
        
        st.markdown(
            f"<div class='subscription-status {status_class}'>"
            f"Subscription Status: {status_text}</div>",
            unsafe_allow_html=True
        )

def main_application_interface():
    # Pinecone and model initialization
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    INDEX_NAME = "rajan"
    EMBEDDING_DIMENSION = 384

    class AgentState(TypedDict):
        resume_text: str
        jobs: List[dict]
        history: Annotated[List[str], operator.add]
        current_response: str
        selected_job: dict

    def init_pinecone():
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

    index = init_pinecone()
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

    # Rest of your existing LangGraph workflow and UI components
    # ... [Keep original application logic here] ...

# Main application flow
def main():
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

    # Main application
    st.title("💼 Premium Career Assistant")
    display_subscription_status()
    
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        st.write(f"Welcome, {st.session_state.username}")
        if st.button("Check Subscription Status"):
            st.experimental_rerun()
            
    main_application_interface()

if __name__ == "__main__":
    main()
