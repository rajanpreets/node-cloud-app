import psycopg2
from contextlib import contextmanager
import bcrypt
import streamlit as st

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        host=st.secrets["DB_HOST"],
        port=st.secrets["DB_PORT"]
    )
    try:
        yield conn
    finally:
        conn.close()

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
                        last_login TIMESTAMP
                    );
                """)
                conn.commit()
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")

def create_user(username, password, email):
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

def get_user_by_username(username):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cur.fetchone()

def verify_password(stored_hash, password):
    return bcrypt.checkpw(password.encode(), stored_hash.encode())

def update_last_login(username):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE username = %s
            """, (username,))
            conn.commit()
