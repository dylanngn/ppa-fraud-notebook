import polars as pl
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI environment variable not set")

try:
    print("Testing connection...")
    df = pl.read_database_uri("SELECT 1", DB_URI, engine="connectorx")
    print("Connection successful!")
    print(df)
except Exception as e:
    print(f"Connection failed: {e}")
