import streamlit as st

# Load secrets from Streamlit Cloud
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PAYPAL_CLIENT_ID = st.secrets["PAYPAL_CLIENT_ID"]
PAYPAL_CLIENT_SECRET = st.secrets["PAYPAL_CLIENT_SECRET"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
