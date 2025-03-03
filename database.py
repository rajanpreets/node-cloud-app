import psycopg2
from contextlib import contextmanager
import bcrypt
import streamlit as st
import logging
from datetime import datetime, timedelta
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
                        last_login TIMESTAMP,
                        payment_status VARCHAR(20) DEFAULT 'pending',
                        payment_id VARCHAR(50) UNIQUE,
                        subscription_end TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        payment_id VARCHAR(50) UNIQUE NOT NULL,
                        amount DECIMAL(10,2) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def create_user(username: str, password: str, email: str, payment_id: str) -> int:
    """Create new user with validation"""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
        
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                cur.execute("""
                    INSERT INTO users (username, password_hash, email, payment_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (username, password_hash, email, payment_id))
                user_id = cur.fetchone()[0]
                conn.commit()
                return user_id
    except psycopg2.IntegrityError as e:
        raise ValueError("Username or payment already exists") from e
    except Exception as e:
        logger.error(f"User creation failed: {str(e)}")
        raise RuntimeError("Registration failed") from e

def get_user_by_username(username: str) -> Optional[Tuple]:
    """Retrieve user by username"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, password_hash, email, payment_status, subscription_end
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

def verify_payment(payment_id: str) -> bool:
    """Verify and activate payment"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET payment_status = 'active',
                        subscription_end = %s
                    WHERE payment_id = %s
                    RETURNING id
                """, (datetime.now() + timedelta(days=365), payment_id))
                user_id = cur.fetchone()
                
                if user_id:
                    cur.execute("""
                        INSERT INTO payments 
                        (user_id, payment_id, amount, status)
                        VALUES (%s, %s, %s, %s)
                    """, (user_id[0], payment_id, 10.00, 'completed'))
                    conn.commit()
                    return True
                return False
    except Exception as e:
        logger.error(f"Payment verification failed: {str(e)}")
        raise

def is_subscription_active(username: str) -> bool:
    """Check if user has active subscription"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT subscription_end 
                    FROM users 
                    WHERE username = %s 
                    AND payment_status = 'active'
                    AND subscription_end > CURRENT_TIMESTAMP
                """, (username,))
                return bool(cur.fetchone())
    except Exception as e:
        logger.error(f"Subscription check failed: {str(e)}")
        return False
