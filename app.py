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
from database import init_db, create_user, get_user_by_username, verify_password, update_last_login, get_user_subscription
from paypal import create_subscription, verify_subscription

# Initialize database
init_db()

# Set page config
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .subscription-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .active { background-color: #e8f5e9; border: 1px solid #4caf50; }
    .inactive { background-color: #ffebee; border: 1px solid #f44336; }
</style>
""", unsafe_allow_html=True)

# [Keep all your existing Pinecone, LangGraph, and state management code here]

# Modified Authentication UI
def authentication_ui():
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = get_user_by_username(username)
                if user and verify_password(user[2], password):
                    subscription = get_user_subscription(user[0])
                    if not subscription or subscription['status'] != 'ACTIVE':
                        st.session_state.temp_user = user
                        st.session_state.requires_subscription = True
                    else:
                        update_last_login(username)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                else:
                    st.error("Invalid credentials")

    with register_tab:
        with st.form("Register"):
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            submit = st.form_submit_button("Create Account")
            
            if submit:
                try:
                    create_user(new_username, new_password, new_email)
                    st.success("Account created! Please login")
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")

# Payment UI Component
def payment_ui():
    st.markdown("## 🔒 Premium Subscription Required")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("""
            ### Unlock Full Access
            Get unlimited job matches and career advice for just $9.69/month
            """)
    with col2:
        if st.button("✨ Subscribe Now"):
            if create_subscription(st.session_state.temp_user[0], st.session_state.temp_user[3]):
                st.session_state.logged_in = True
                st.session_state.username = st.session_state.temp_user[1]
                st.rerun()
            else:
                st.error("Failed to create subscription. Please try again.")

# Subscription Status Component
def subscription_status():
    user = get_user_by_username(st.session_state.username)
    subscription = get_user_subscription(user[0])
    
    status_class = "active" if subscription and subscription['status'] == 'ACTIVE' else "inactive"
    status_text = "Active" if subscription else "Inactive"
    
    st.markdown(f"""
    <div class="subscription-status {status_class}">
        <h4>Subscription Status: {status_text}</h4>
        {f"<p>Plan: Premium ($9.69/month)<br>Next Billing Date: {subscription['end_date'].strftime('%Y-%m-%d') if subscription else ''}</p>" if subscription else ""}
    </div>
    """, unsafe_allow_html=True)

# Modified Main Application Flow
def main_application():
    # Add subscription status to sidebar
    with st.sidebar:
        subscription_status()
        if st.button("Manage Subscription"):
            st.session_state.show_subscription_management = True

    # [Keep your existing main application code here]

# Application Entry Point
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.requires_subscription = False

if not st.session_state.logged_in:
    authentication_ui()
    if st.session_state.get('requires_subscription'):
        payment_ui()
    st.stop()

# Check subscription status on every load
user = get_user_by_username(st.session_state.username)
subscription = get_user_subscription(user[0])
if not subscription or subscription['status'] != 'ACTIVE':
    st.session_state.requires_subscription = True
    st.session_state.temp_user = user
    payment_ui()
    st.stop()

main_application()
