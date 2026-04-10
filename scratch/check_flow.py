import sqlite3
import json
import os

db_path = r'd:\Work\langflow\langflow\src\backend\base\langflow\langflow.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('SELECT name, data FROM flow WHERE name LIKE "%astraDB%"')
rows = cur.fetchall()
for name, data_str in rows:
    print(f"Flow: {name}")
    data = json.loads(data_str)
    # Search for nodes with "token" or "api_key" fields
    for node in data.get('nodes', []):
        node_data = node.get('data', {}).get('node', {})
        template = node_data.get('template', {})
        for field_name, field_data in template.items():
            if any(k in field_name.lower() for k in ['token', 'api_key', 'secret']):
                val = field_data.get('value')
                print(f"  Field: {field_name}, Value: {val}")
conn.close()
