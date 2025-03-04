import psycopg2
from contextlib import contextmanager
import bcrypt
import streamlit as st
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Get database connection using Streamlit secrets"""
    try:
        conn = psycopg2.connect(
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"]
        )
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
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
        logger.error(f"Init failed: {str(e)}")
        raise

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
        logger.error(f"Get user failed: {str(e)}")
        return None

def verify_password(stored_hash: str, password: str) -> bool:
    """Securely verify password"""
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception as e:
        logger.error(f"Password verify failed: {str(e)}")
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
        logger.error(f"Update last login failed: {str(e)}")
        raise
