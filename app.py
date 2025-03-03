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
</style>
""", unsafe_allow_html=True)

def handle_payment_callback():
    """Handle PayPal payment redirects"""
    params = st.query_params
    if 'token' in params:
        try:
            order_id = params['token']
            if paypal.verify_payment(order_id) and verify_payment(order_id):
                st.success("✅ Payment verified! Account activated.")
                st.session_state.registration_success = True
                st.rerun()
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
                    order = paypal.create_order(10.00)
                    if not order:
                        raise ValueError("Failed to create payment order")
                    
                    approval_url = next(
                        link['href'] for link in order['links'] 
                        if link['rel'] == 'approve'
                    )
                    
                    st.session_state.registration_data = {
                        'username': new_username,
                        'email': new_email,
                        'password': new_password,
                        'payment_id': order['id']
                    }
                    
                    st.markdown(
                        f"<a href='{approval_url}' class='payment-button' target='_blank'>"
                        "🔒 Complete Secure $10 Payment via PayPal</a>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    logger.error(f"Registration error: {str(e)}")
                    st.error(f"Registration failed: {str(e)}")

def main_application():
    """Main application interface"""
    # Initialize services
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("rajan")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    # Existing application logic
    # ... [Keep your original application workflow here] ...

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
