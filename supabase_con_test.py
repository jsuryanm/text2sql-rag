import os 
import psycopg2 #python postgres db adapter
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not found in .env")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT NOW();"
    )

    print("Connected. Time:", cur.fetchone()[0])
    conn.close()

except Exception as e:
    print("Failed to connect:",e)