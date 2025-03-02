import psycopg2
from contextlib import contextmanager
import bcrypt
import streamlit as st
import logging

# Configure logging
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Get database connection with proper error handling"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", st.secrets.get("DB_NAME")),
            user=os.getenv("DB_USER", st.secrets.get("DB_USER")),
            password=os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD")),
            host=os.getenv("DB_HOST", st.secrets.get("DB_HOST")),
            port=os.getenv("DB_PORT", st.secrets.get("DB_PORT", 5432))
        )
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        st.error("Database connection error. Please try again later.")
        raise

def init_db():
    """Initialize database with proper error handling"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create users table
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
                # Create jobs table (example additional table)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_jobs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        job_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        st.error("Database initialization failed. Please check your credentials.")

def create_user(username, password, email):
    """Create user with proper error handling"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                cur.execute("""
                    INSERT INTO users (username, password_hash, email)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (username, password_hash, email))
                conn.commit()
                return cur.fetchone()[0]
    except psycopg2.IntegrityError:
        raise ValueError("Username already exists")
    except Exception as e:
        logger.error(f"User creation failed: {str(e)}")
        raise RuntimeError("User creation failed")

def get_user_by_username(username):
    """Get user by username with proper error handling"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                return cur.fetchone()
    except Exception as e:
        logger.error(f"User retrieval failed: {str(e)}")
        return None

def verify_password(stored_hash, password):
    """Verify password with bcrypt"""
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def update_last_login(username):
    """Update last login timestamp"""
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
        logger.error(f"Last login update failed: {str(e)}")
