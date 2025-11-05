"""
Test PostgreSQL LISTEN/NOTIFY directly
"""

import psycopg
import os
from dotenv import load_dotenv
import time
import threading

load_dotenv()

# Build database URL - USE DIRECT CONNECTION (port 5432) not pooler (6543)
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', '')
db_host = os.getenv('DB_HOST', 'localhost')
db_name = os.getenv('DB_NAME', 'postgres')

# IMPORTANT: Use port 5432 (direct) for LISTEN/NOTIFY, NOT 6543 (pooler)
database_url = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

print("=" * 80)
print("🧪 TESTING POSTGRESQL LISTEN/NOTIFY")
print("=" * 80)

# Listener thread
def listener():
    print("\n🎧 Starting LISTENER...")
    try:
        conn = psycopg.connect(database_url, autocommit=True)
        print(f"✅ Connected to: {db_host}:5432/{db_name}")
        
        with conn.cursor() as cur:
            cur.execute("LISTEN system_alarm_channel;")
            print("✅ Listening on 'system_alarm_channel'")
            print("⏳ Waiting for notifications...\n")
            
            for i in range(30):  # Wait up to 30 seconds
                gen = conn.notifies(timeout=1.0)
                for notify in gen:
                    print("=" * 80)
                    print("🔔 NOTIFICATION RECEIVED!")
                    print(f"   Channel: {notify.channel}")
                    print(f"   Payload: {notify.payload}")
                    print("=" * 80)
                time.sleep(0.1)
        
        conn.close()
        print("\n🛑 Listener stopped")
        
    except Exception as e:
        print(f"❌ Listener error: {e}")
        import traceback
        traceback.print_exc()

# Start listener in background
listener_thread = threading.Thread(target=listener, daemon=True)
listener_thread.start()

# Wait for listener to connect
time.sleep(2)

# Sender
print("\n📤 SENDING TEST NOTIFICATION...")
print("-" * 80)

try:
    conn = psycopg.connect(database_url, autocommit=True)
    print("✅ Sender connected")
    
    with conn.cursor() as cur:
        # Send test notification
        test_payload = '{"test": "Hello from test!", "timestamp": "' + str(time.time()) + '"}'
        print(f"   Payload: {test_payload}")
        
        cur.execute(
            "SELECT pg_notify('system_alarm_channel', %s);",
            (test_payload,)
        )
        print("✅ Notification sent!")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Sender error: {e}")
    import traceback
    traceback.print_exc()

print("\n⏳ Waiting for listener to receive (5 seconds)...")
time.sleep(5)

print("\n✅ Test completed!")
