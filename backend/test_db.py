import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.database import get_db, init_db
import json

def test_db():
    init_db()
    db = get_db()
    # Test user insertion
    try:
        db.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)', 
                   ('test@test.com', 'Tester', 'hashed_pass'))
        db.commit()
        print("User insertion test passed.")
    except Exception as e:
        print(f"User insertion test failed or user exists: {e}")
    
    # Test history insertion
    try:
        db.execute('INSERT INTO history (user_email, prediction, confidence, features_json) VALUES (?, ?, ?, ?)',
                   ('test@test.com', 'A', 0.95, json.dumps({"test": 1})))
        db.commit()
        print("History insertion test passed.")
    except Exception as e:
        print(f"History insertion test failed: {e}")
    
    # Test fetch
    history = db.execute('SELECT * FROM history WHERE user_email = ?', ('test@test.com',)).fetchall()
    if len(history) > 0:
        print(f"History fetch test passed. Found {len(history)} entries.")
    else:
        print("History fetch test failed.")
    
    db.close()

if __name__ == "__main__":
    test_db()
