"""
Check Database Schema and Enum Values
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_enum_values():
    """Check enum values in database"""
    try:
        conn = psycopg2.connect(
            host="aws-1-ap-southeast-1.pooler.supabase.com",
            port=5432,
            database="postgres",
            user="postgres.undznprwlqjpnxqsgyiv",
            password=os.getenv("DB_PASSWORD", "12345678")
        )
        
        cur = conn.cursor()
        
        # Check enum types
        print("🔍 Checking enum types...")
        cur.execute("""
            SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder)
            FROM pg_type t 
            JOIN pg_enum e ON t.oid = e.enumtypid 
            WHERE t.typname LIKE '%enum%'
            GROUP BY t.typname
            ORDER BY t.typname;
        """)
        
        enums = cur.fetchall()
        for enum_name, enum_values in enums:
            print(f"📊 {enum_name}: {enum_values}")
        
        # Check event_detections table structure  
        print("\n🏗️ Checking event_detections table structure...")
        cur.execute("""
            SELECT column_name, data_type, column_default, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'event_detections'
            ORDER BY ordinal_position;
        """)
        
        columns = cur.fetchall()
        for col in columns:
            print(f"   {col[0]}: {col[1]} (default: {col[2]}, nullable: {col[3]})")
        
        # Check existing events
        print("\n📋 Sample existing events:")
        cur.execute("SELECT event_type, COUNT(*) FROM event_detections GROUP BY event_type;")
        event_counts = cur.fetchall()
        
        if event_counts:
            for event_type, count in event_counts:
                print(f"   {event_type}: {count} events")
        else:
            print("   No events found")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_enum_values()
