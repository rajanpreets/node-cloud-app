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
        logger.error(f"Init failed: {str(e)}")
        raise

def create_user(username: str, password: str, email: str, payment_id: str) -> int:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
        
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
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
                conn.rollback()
                if "users_payment_id_key" in str(e):
                    raise ValueError("Payment already used")
                raise ValueError("Username already exists")
            except Exception as e:
                conn.rollback()
                logger.error(f"Create user failed: {str(e)}")
                raise RuntimeError("Registration failed")

def get_user_by_username(username: str) -> Optional[Tuple]:
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
        logger.error(f"Get user failed: {str(e)}")
        return None

def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception as e:
        logger.error(f"Password verify failed: {str(e)}")
        return False

def update_last_login(username: str):
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

def verify_payment(payment_id: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Update user status
                cur.execute("""
                    UPDATE users
                    SET payment_status = 'active',
                        subscription_end = %s
                    WHERE payment_id = %s
                    RETURNING id
                """, (datetime.now() + timedelta(days=365), payment_id))
                user_id = cur.fetchone()
                
                if user_id:
                    # Record payment
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
