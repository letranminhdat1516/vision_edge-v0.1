"""
Check existing records for foreign keys
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_existing_records():
    """Check existing camera, room, user records"""
    try:
        conn = psycopg2.connect(
            host="aws-1-ap-southeast-1.pooler.supabase.com",
            port=5432,
            database="postgres",
            user="postgres.undznprwlqjpnxqsgyiv",
            password=os.getenv("DB_PASSWORD", "12345678")
        )
        
        cur = conn.cursor()
        
        # Check cameras
        print("📷 Checking cameras...")
        cur.execute("SELECT camera_id, camera_name FROM cameras LIMIT 3;")
        cameras = cur.fetchall()
        
        if cameras:
            for camera in cameras:
                print(f"   Camera: {camera[0]} - {camera[1]}")
        else:
            print("   No cameras found")
        
        # Check rooms
        print("\n🏠 Checking rooms...")
        cur.execute("SELECT room_id, room_name FROM rooms LIMIT 3;")
        rooms = cur.fetchall()
        
        if rooms:
            for room in rooms:
                print(f"   Room: {room[0]} - {room[1]}")
        else:
            print("   No rooms found")
        
        # Check users
        print("\n👤 Checking users...")
        cur.execute("SELECT user_id, email FROM users LIMIT 3;")
        users = cur.fetchall()
        
        if users:
            for user in users:
                print(f"   User: {user[0]} - {user[1]}")
        else:
            print("   No users found")
            
        cur.close()
        conn.close()
        
        return cameras, rooms, users
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], [], []

if __name__ == "__main__":
    check_existing_records()
