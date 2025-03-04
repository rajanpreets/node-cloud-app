import psycopg2
from contextlib import contextmanager
import bcrypt
import streamlit as st
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
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

def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Users table
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
            # Subscriptions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) UNIQUE,
                    subscription_id VARCHAR(255) UNIQUE,
                    plan_id VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP,
                    payer_email VARCHAR(100)
                );
            """)
            conn.commit()

def create_user(username: str, password: str, email: str) -> int:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
        
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
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
                conn.rollback()
                raise ValueError("Username already exists")

def get_user_by_username(username: str) -> Optional[Tuple]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, username, password_hash, email 
                FROM users 
                WHERE username = %s
            """, (username,))
            return cur.fetchone()

def verify_password(stored_hash: str, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), stored_hash.encode())

def update_last_login(username: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE username = %s
            """, (username,))
            conn.commit()

def store_subscription(user_id: int, subscription_data: Dict):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO subscriptions 
                (user_id, subscription_id, plan_id, status, start_date, payer_email)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET subscription_id = EXCLUDED.subscription_id,
                    plan_id = EXCLUDED.plan_id,
                    status = EXCLUDED.status,
                    start_date = EXCLUDED.start_date,
                    payer_email = EXCLUDED.payer_email
            """, (
                user_id,
                subscription_data['id'],
                subscription_data['plan_id'],
                subscription_data['status'],
                datetime.fromisoformat(subscription_data['start_time']),
                subscription_data['subscriber']['email_address']
            ))
            conn.commit()

def get_user_subscription(user_id: int) -> Optional[Dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT subscription_id, plan_id, status, start_date, end_date, payer_email
                FROM subscriptions
                WHERE user_id = %s
            """, (user_id,))
            result = cur.fetchone()
            if result:
                return {
                    "subscription_id": result[0],
                    "plan_id": result[1],
                    "status": result[2],
                    "start_date": result[3],
                    "end_date": result[4],
                    "payer_email": result[5]
                }
            return None
