"""
Check snapshots table structure
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_snapshots_table():
    """Check snapshots table structure"""
    try:
        conn = psycopg2.connect(
            host="aws-1-ap-southeast-1.pooler.supabase.com",
            port=5432,
            database="postgres",
            user="postgres.undznprwlqjpnxqsgyiv",
            password=os.getenv("DB_PASSWORD", "12345678")
        )
        
        cur = conn.cursor()
        
        # Check snapshots table structure  
        print("🏗️ Checking snapshots table structure...")
        cur.execute("""
            SELECT column_name, data_type, column_default, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'snapshots'
            ORDER BY ordinal_position;
        """)
        
        columns = cur.fetchall()
        for col in columns:
            print(f"   {col[0]}: {col[1]} (default: {col[2]}, nullable: {col[3]})")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_snapshots_table()
