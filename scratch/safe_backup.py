import sqlite3
import os

src = r'd:\Work\langflow\langflow\src\backend\base\langflow\langflow.db'
dst = r'C:\Users\tamoj\AppData\Local\langflow\langflow\langflow.db'

try:
    # Connect to the source database
    src_conn = sqlite3.connect(src)
    # Connect to the destination database
    dst_conn = sqlite3.connect(dst)
    
    # Use the backup API
    with dst_conn:
        src_conn.backup(dst_conn)
    
    print("Backup successful!")
    src_conn.close()
    dst_conn.close()
except Exception as e:
    print(f"Error during backup: {e}")
    exit(1)
