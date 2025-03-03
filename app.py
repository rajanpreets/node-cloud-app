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

# Custom CSS (keep your existing styles)

def handle_payment_callback():
    """Handle PayPal payment redirects"""
    params = st.query_params
    if 'token' in params:
        try:
            order_id = params['token']
            if paypal.verify_payment(order_id) and verify_payment(order_id):
                st.success("✅ Payment verified! Account activated.")
                # Automatically log in the user
                if 'registration_data' in st.session_state:
                    user_data = st.session_state.registration_data
                    try:
                        user = get_user_by_username(user_data['username'])
                        if user and is_subscription_active(user_data['username']):
                            st.session_state.update({
                                "logged_in": True,
                                "username": user_data['username']
                            })
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Auto-login failed: {str(e)}")
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
                    
                    # Create user in database with pending status
                    create_user(
                        new_username,
                        new_password,
                        new_email,
                        order['id']  # Use PayPal order ID as payment_id
                    )
                    
                    # Store registration data in session
                    st.session_state.registration_data = {
                        'username': new_username,
                        'email': new_email
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

# Rest of your application code (main_application and other functions)

if __name__ == "__main__":
    main()
