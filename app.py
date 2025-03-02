import streamlit as st
import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
from streamlit.components.v1 import html

# Load environment variables
load_dotenv()

# Firebase Initialization ======================================================
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            firebase_secrets = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"],
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
            }
            
            cred = credentials.Certificate(firebase_secrets)
            firebase_admin.initialize_app(cred)
            auth.list_users()
            
        except Exception as e:
            st.error(f"Firebase Initialization Error: {str(e)}")
            st.stop()

initialize_firebase()

# Authentication Components ===================================================
def get_current_user():
    return st.session_state.get('auth_user')

def firebase_auth_component():
    firebase_config = {
        "apiKey": st.secrets["firebase_config"]["apiKey"],
        "authDomain": st.secrets["firebase_config"]["authDomain"],
        "projectId": st.secrets["firebase_config"]["projectId"],
        "storageBucket": st.secrets["firebase_config"]["storageBucket"],
        "messagingSenderId": st.secrets["firebase_config"]["messagingSenderId"],
        "appId": st.secrets["firebase_config"]["appId"],
        "measurementId": st.secrets["firebase_config"]["measurementId"]
    }
    
    auth_js = f"""
    <!-- Firebase Auth Scripts -->
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

# Rest of your application code (career assistant functionality) #
# ... (keep your existing career assistant implementation here) ... #

# Authentication Check ========================================================
if not get_current_user():
    st.set_page_config(page_title="Login - Career Assistant", layout="centered")
    st.title("🔐 Secure Career Assistant Login")
    firebase_auth_component()
    st.stop()
else:
    st.set_page_config(page_title="💬 AI Career Assistant", layout="wide")
    st.title("💬 AI Career Assistant")
    # Main application content #
