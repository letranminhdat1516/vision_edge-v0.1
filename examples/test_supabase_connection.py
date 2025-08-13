"""
Test PostgreSQL connection to Supabase using individual parameters
"""
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT", "5432")
DBNAME = os.getenv("DB_NAME", "postgres")

print("Connection parameters:")
print(f"  User: {USER}")
print(f"  Password: {'*' * len(PASSWORD) if PASSWORD else None}")
print(f"  Host: {HOST}")
print(f"  Port: {PORT}")
print(f"  Database: {DBNAME}")
print()

# Connect to the database
try:
    print("🔄 Attempting connection...")
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        database=DBNAME,
        connect_timeout=10
    )
    print("✅ Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    print("🔄 Testing query...")
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print(f"✅ Current Time: {result[0]}")
    
    # Test table existence
    print("🔄 Checking tables...")
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('event_detections', 'alerts', 'snapshots')
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    
    if tables:
        print(f"✅ Found healthcare tables: {[t[0] for t in tables]}")
    else:
        print("⚠️ No healthcare tables found - may need to run schema")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("✅ Connection closed successfully")

except Exception as e:
    print(f"❌ Failed to connect: {e}")
    print()
    print("Troubleshooting:")
    print("  1. Check if all environment variables are set in .env")
    print("  2. Verify Supabase project is active")
    print("  3. Check network connectivity")
    print("  4. Ensure password is correct")
