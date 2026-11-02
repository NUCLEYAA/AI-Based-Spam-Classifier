import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "spam_history.db"

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                date_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_prediction(message, prediction, confidence):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (message_text, prediction, confidence_score) VALUES (?, ?, ?)",
            (message, prediction, confidence)
        )

def get_history(limit=200):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM messages ORDER BY message_id DESC LIMIT ?", (limit,)
        ).fetchall()

def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM messages")
