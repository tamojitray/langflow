import sqlite3
import os

db_path = r'd:\Work\langflow\langflow\src\backend\base\langflow\langflow.db'
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
try:
    cur.execute('SELECT name, type, value, user_id FROM variable')
    rows = cur.fetchall()
    print(f"Total variables: {len(rows)}")
    for name, type_, value, user_id in rows:
        val_preview = "Secret" if type_ == "Generic (Secret)" else (value[:20] if value else "None")
        print(f"Name: {name}, Type: {type_}, Value: {val_preview}, UserID: {user_id}")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
