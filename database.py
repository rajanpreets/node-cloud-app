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

# ... [Keep the rest of your database functions unchanged] ...
