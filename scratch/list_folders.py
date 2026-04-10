import sqlite3
import json
import os

db_path = r'd:\Work\langflow\langflow\src\backend\base\langflow\langflow.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('SELECT flow.name, folder.name FROM flow JOIN folder ON flow.folder_id = folder.id')
rows = cur.fetchall()
for flow_name, folder_name in rows:
    print(f"Flow: {flow_name}, Folder: {folder_name}")
conn.close()
