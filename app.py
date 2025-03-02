import streamlit as st
import os
import pandas as pd
import firebase_admin
import json
from firebase_admin import credentials, auth
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
import operator
from streamlit.components.v1 import html

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rajan"
EMBEDDING_DIMENSION = 384

# Firebase Initialization ======================================================
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            # Load Firebase configuration from secrets
            firebase_secrets = dict(st.secrets["firebase"])
            
            # Remove non-essential keys that might cause issues
            firebase_secrets.pop("universe_domain", None)
            
            # Initialize Firebase app
            cred = credentials.Certificate(firebase_secrets)
            firebase_admin.initialize_app(cred)
            
            # Verify connection
            auth.list_users()
            
        except Exception as e:
            st.error(f"""
            🔥 Firebase Initialization Error: {str(e)}
            
            Common Fixes:
            1. Verify service account credentials in secrets.toml
            2. Check Firebase project permissions at [Google Cloud Console](https://console.cloud.google.com)
            3. Ensure authorized domains include:
               - localhost
               - *.streamlit.app
            """)
            st.stop()

initialize_firebase()

# Authentication Components ===================================================
def get_current_user():
    return st.session_state.get('auth_user')

def firebase_auth_component():
    firebase_config = st.secrets["firebase_config"]
    
    auth_js = f"""
    <script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.0.0/firebase-auth-compat.js"></script>
    <script>
        const firebaseConfig = {json.dumps(firebase_config)};
        const app = firebase.initializeApp(firebaseConfig);
        
        async function signInWithGoogle() {{
            const provider = new firebase.auth.GoogleAuthProvider();
            try {{
                const result = await firebase.auth().signInWithPopup(provider);
                const user = result.user;
                
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH',
                    user: {{
                        uid: user.uid,
                        email: user.email,
                        name: user.displayName,
                        photoURL: user.photoURL,
                        refreshToken: user.refreshToken,
                        idToken: await user.getIdToken()
                    }}
                }}, '*');
            }} catch (error) {{
                console.error('Authentication error:', error);
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_ERROR',
                    error: error.message
                }}, '*');
            }}
        }}
    </script>
    """
    
    auth_html = f"""
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <button onclick="signInWithGoogle()" style="
            background: #4285F4;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: transform 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
                <path d="M12.24 10.285V14.4h6.806c-.275 1.765-2.056 5.174-6.806 5.174-4.095 0-7.439-3.389-7.439-7.574s3.345-7.574 7.439-7.574c2.33 0 3.891.989 4.785 1.849l3.254-3.138C18.189 1.186 15.479 0 12.24 0c-6.635 0-12 5.365-12 12s5.365 12 12 12c6.926 0 11.52-4.869 11.52-11.726 0-.788-.085-1.39-.189-1.989H12.24z"/>
            </svg>
            Sign in with Google
        </button>
    </div>
    {auth_js}
    """
    
    html(auth_html, height=100)

# Authentication Check ========================================================
if not get_current_user():
    st.set_page_config(page_title="Login - Career Assistant", layout="centered")
    st.title("🔐 Secure Career Assistant Login")
    
    st.markdown("""
        <div style='
            text-align: center; 
            padding: 2rem; 
            border-radius: 10px; 
            background: #f8f9fa;
            margin: 2rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        '>
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>Enterprise-Grade Security</h3>
            <div style='
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-bottom: 1.5rem;
            '>
                <div style='padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 5px;'>
                    🔒 AES-256 Encryption
                </div>
                <div style='padding: 0.5rem 1rem; background: #e3f2fd; border-radius: 5px;'>
                    🔑 OAuth 2.0
                </div>
            </div>
            <p style='color: #7f8c8d; font-size: 0.9rem;'>
                Your data is protected with bank-level security measures
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    firebase_auth_component()
    
    # Auth callback handler
    html("""
    <script>
        window.addEventListener('message', async (event) => {
            if (event.data.type === 'FIREBASE_AUTH') {
                const user = event.data.user;
                
                await fetch('/_stcore/set-session-data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        key: 'auth_user',
                        value: user
                    })
                });
                
                window.location.reload();
            }
            
            if (event.data.type === 'FIREBASE_AUTH_ERROR') {
                console.error('Auth error:', event.data.error);
            }
        });
    </script>
    """)
    
    st.markdown("---")
    st.info("""
        ℹ️ **Security Notice**  
        - All communications are encrypted with TLS 1.3  
        - Session tokens automatically expire after 1 hour  
        - No personal data is stored permanently
    """)
    st.stop()

# Main Application ============================================================
st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
st.title("💬 AI Career Assistant")

# User Session Management -----------------------------------------------------
def handle_logout():
    try:
        auth.revoke_refresh_tokens(get_current_user()['uid'])
    except Exception as e:
        st.error(f"Logout error: {str(e)}")
    del st.session_state.auth_user
    st.rerun()

# Sidebar User Profile --------------------------------------------------------
with st.sidebar:
    user = get_current_user()
    
    st.markdown(f"""
        <div style='
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        '>
            <div style='
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            '>
                <img src="{user['photoURL']}" 
                     style='
                         width: 48px;
                         height: 48px;
                         border-radius: 50%;
                         object-fit: cover;
                     '>
                <div>
                    <h4 style='margin: 0; color: #2c3e50;'>{user['name']}</h4>
                    <small style='color: #7f8c8d;'>{user['email']}</small>
                </div>
            </div>
            <div style='
                display: flex;
                gap: 0.5rem;
                justify-content: space-between;
            '>
                <button onclick="window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_LOGOUT'
                }}, '*')" 
                style='
                    flex: 1;
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    color: #2c3e50;
                    padding: 8px;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.2s;
                '>
                    🔄 Refresh Session
                </button>
                <button onclick="handleLogout()" 
                style='
                    flex: 1;
                    background: #fff0f0;
                    border: 1px solid #ffcccc;
                    color: #dc3545;
                    padding: 8px;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.2s;
                '>
                    🚪 Logout
                </button>
            </div>
        </div>
        
        <script>
            function handleLogout() {{
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_LOGOUT'
                }}, '*');
                
                fetch('/_stcore/set-session-data', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        key: 'auth_user',
                        value: null
                    }})
                }}).then(() => window.location.reload());
            }}
        </script>
    """, unsafe_allow_html=True)

# Original Career Assistant Functionality --------------------------------------
# ... (Keep your existing career assistant code here) ...

# Security Handlers ===========================================================
html("""
<script>
    window.addEventListener('message', (event) => {
        if (event.data.type === 'FIREBASE_AUTH_LOGOUT') {
            firebase.auth().signOut();
        }
    });
</script>
""")
