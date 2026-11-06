
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

database_url = os.environ["DATABASE_URL"]

with psycopg.connect(database_url) as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database(), 1")
        database, status = cursor.fetchone()

print("Connected to Neon PostgreSQL!")
print("Database:", database)
print("Connection test:", status)
