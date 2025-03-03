import psycopg2
from contextlib import contextmanager
import bcrypt
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
_db_config = None

def init_db(config: dict):
    global _db_config
    _db_config = config
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
                    )""")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id VARCHAR(40) PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        expires_at TIMESTAMP NOT NULL
                    )""")
                conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

@contextmanager
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=_db_config['name'],
            user=_db_config['user'],
            password=_db_config['password'],
            host=_db_config['host'],
            port=_db_config['port']
        )
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        raise

def create_user(username: str, password: str, email: str) -> int:
    if len(password) < 8:
        raise ValueError("Password must contain 8+ characters")
        
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                cur.execute("""
                    INSERT INTO users (username, password_hash, email)
                    VALUES (%s, %s, %s)
                    RETURNING id""", (username, pw_hash, email))
                user_id = cur.fetchone()[0]
                conn.commit()
                return user_id
    except psycopg2.IntegrityError:
        raise ValueError("Username already exists")
    except Exception as e:
        logger.error(f"User creation error: {str(e)}")
        raise RuntimeError("Account creation failed")

def get_user_by_username(username: str) -> Optional[Tuple]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, password_hash, email 
                    FROM users 
                    WHERE username = %s""", (username,))
                return cur.fetchone()
    except Exception as e:
        logger.error(f"User fetch error: {str(e)}")
        return None

def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def update_last_login(username: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE username = %s""", (username,))
                conn.commit()
    except Exception as e:
        logger.error(f"Login update failed: {str(e)}")
        raise
