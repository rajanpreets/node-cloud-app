import psycopg2
import bcrypt
import streamlit as st
from contextlib import contextmanager
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Get database connection using Streamlit secrets"""
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
        raise

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
                conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

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

def get_user_by_username(username: str) -> Optional[Tuple]:
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

def verify_password(stored_hash: str, password: str) -> bool:
    """Securely verify password"""
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def update_last_login(username: str):
    """Update user's last login timestamp"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET last_login = CURRENT_TIMESTAMP
                    WHERE username = %s
                """, (username,))
                conn.commit()
    except Exception as e:
        logger.error(f"Login update failed: {str(e)}")
        raise
