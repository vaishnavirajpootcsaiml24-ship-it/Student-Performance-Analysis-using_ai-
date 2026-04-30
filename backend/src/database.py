import sqlite3
import os

# Database path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get(
    'DB_PATH',
    os.path.join(os.path.dirname(BASE_DIR), 'database.db')
)

def init_db():
    """Initialize the database with all required tables.
    Uses IF NOT EXISTS to safely handle repeated calls.
    Runs column migrations for schema upgrades.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            institution TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # History table
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT NOT NULL,
            confidence REAL,
            features_json TEXT NOT NULL,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')

    # ── Schema Migrations ─────────────────────────────────────────────────
    # Safely add columns that may not exist in older databases
    _safe_add_column(c, 'users', 'institution', 'TEXT DEFAULT \'\'')
    _safe_add_column(c, 'users', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')

    conn.commit()
    conn.close()


def _safe_add_column(cursor, table, column, col_type):
    """Add a column to a table only if it doesn't already exist."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists


def get_db():
    """Return a database connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
