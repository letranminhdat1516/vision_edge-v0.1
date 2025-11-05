"""
Complete Alarm System Test
Tests the entire flow: main.py handler + trigger_alarm_test.py
"""

import os
import time
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

print("=" * 80)
print("🧪 COMPLETE ALARM SYSTEM DIAGNOSTIC")
print("=" * 80)

# 1. Check database connection
print("\n1️⃣ Testing Database Connection...")
try:
    from service.postgresql_healthcare_service import PostgreSQLHealthcareService
    db_service = PostgreSQLHealthcareService()
    conn = db_service.get_connection()
    if conn:
        print("   ✅ PostgreSQL connection OK (pooler)")
        db_service.return_connection(conn)
    else:
        print("   ❌ PostgreSQL connection FAILED")
        exit(1)
except Exception as e:
    print(f"   ❌ Database error: {e}")
    exit(1)

# 2. Check direct connection for LISTEN/NOTIFY
print("\n2️⃣ Testing Direct Connection (port 5432)...")
try:
    import psycopg
    
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_name = os.getenv('DB_NAME', 'postgres')
    
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    
    conn = psycopg.connect(database_url, autocommit=True)
    print(f"   ✅ Direct connection OK: {db_host}:5432")
    conn.close()
except Exception as e:
    print(f"   ❌ Direct connection FAILED: {e}")
    print("   💡 This is required for LISTEN/NOTIFY!")

# 3. Check trigger exists
print("\n3️⃣ Checking Database Trigger...")
try:
    db_service = PostgreSQLHealthcareService()
    conn = db_service.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT trigger_name, event_manipulation
        FROM information_schema.triggers
        WHERE trigger_name LIKE '%alarm%'
          AND event_object_table = 'event_detections'
    """)
    
    triggers = cursor.fetchall()
    if triggers:
        print("   ✅ Alarm trigger found:")
        for trig in triggers:
            name = trig['trigger_name'] if isinstance(trig, dict) else trig[0]
            event = trig['event_manipulation'] if isinstance(trig, dict) else trig[1]
            print(f"      - {name} on {event}")
    else:
        print("   ❌ No alarm triggers found!")
        print("   💡 Run: database/create_emergency_alarm_trigger.sql")
    
    db_service.return_connection(conn)
except Exception as e:
    print(f"   ❌ Trigger check failed: {e}")

# 4. Check trigger function and channel
print("\n4️⃣ Checking Trigger Function...")
try:
    db_service = PostgreSQLHealthcareService()
    conn = db_service.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT pg_get_functiondef(oid) as definition
        FROM pg_proc
        WHERE proname = 'notify_alarm_trigger'
    """)
    
    func = cursor.fetchone()
    if func:
        definition = func['definition'] if isinstance(func, dict) else func[0]
        
        if 'system_alarm_channel' in definition:
            print("   ✅ Function sends to 'system_alarm_channel'")
        elif 'emergency_alarm_channel' in definition:
            print("   ⚠️  Function sends to 'emergency_alarm_channel'")
            print("   💡 Handler listens to 'system_alarm_channel' - MISMATCH!")
        else:
            print("   ⚠️  Unknown channel in function")
        
        if 'ALARM_ACTIVATED' in definition:
            print("   ✅ Function triggers on ALARM_ACTIVATED")
        else:
            print("   ❌ Function may not trigger on ALARM_ACTIVATED")
    else:
        print("   ❌ Function 'notify_alarm_trigger' not found!")
    
    db_service.return_connection(conn)
except Exception as e:
    print(f"   ❌ Function check failed: {e}")

# 5. Check audio system
print("\n5️⃣ Checking Audio System...")
try:
    from infrastructure.services.audio_alert_service import audio_alert_service
    
    status = audio_alert_service.get_status()
    
    if status['enabled']:
        print("   ✅ Audio system enabled")
        print(f"      Backend: {status['audio_backend']}")
        print(f"      Devices: {status['available_devices']}")
    else:
        print("   ❌ Audio system disabled!")
        print("   💡 No sound will play even if handler works!")
except Exception as e:
    print(f"   ❌ Audio check failed: {e}")

# 6. Test LISTEN/NOTIFY
print("\n6️⃣ Testing LISTEN/NOTIFY (10 seconds)...")
try:
    import psycopg
    import threading
    
    received = []
    listener_ready = []
    
    def listener():
        try:
            conn = psycopg.connect(database_url, autocommit=True)
            with conn.cursor() as cur:
                cur.execute("LISTEN system_alarm_channel;")
                listener_ready.append(True)
                print("      Listener connected and waiting...")
                
                for i in range(100):  # 10 seconds
                    gen = conn.notifies(timeout=0.1)
                    for notify in gen:
                        print(f"      📩 Received: {notify.channel} - {notify.payload}")
                        received.append(notify)
                        return
            conn.close()
        except Exception as e:
            print(f"      ❌ Listener error: {e}")
    
    # Start listener
    thread = threading.Thread(target=listener, daemon=True)
    thread.start()
    
    # Wait for listener to be ready
    for _ in range(50):  # 5 seconds max
        if listener_ready:
            break
        time.sleep(0.1)
    
    if not listener_ready:
        print("   ❌ Listener failed to start!")
    else:
        time.sleep(0.5)  # Extra safety
        
        # Send test notification
        print("      Sending test notification...")
        conn = psycopg.connect(database_url, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_notify('system_alarm_channel', 'test_message');")
        conn.close()
        print("      Notification sent, waiting...")
        
        thread.join(timeout=5)
        
        if received:
            print("   ✅ LISTEN/NOTIFY working!")
            print(f"      Received: {len(received)} notification(s)")
        else:
            print("   ❌ LISTEN/NOTIFY not working!")
            print("   💡 Notification sent but not received!")
            print("   💡 This means handler WON'T receive alarm triggers!")
        
except Exception as e:
    print(f"   ❌ LISTEN/NOTIFY test failed: {e}")
    import traceback
    traceback.print_exc()

# 7. Check handler is imported correctly
print("\n7️⃣ Checking Handler Import...")
try:
    from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
    print("   ✅ Handler imported successfully")
    print(f"      Channel: {emergency_alarm_handler.channel_name}")
    print(f"      Running: {emergency_alarm_handler.is_running}")
except Exception as e:
    print(f"   ❌ Handler import failed: {e}")

# Summary
print("\n" + "=" * 80)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 80)
print("\n💡 For alarm system to work, ALL must be ✅:")
print("   1. Database connection")
print("   2. Direct connection (port 5432)")
print("   3. Database trigger exists")
print("   4. Trigger sends to correct channel")
print("   5. Audio system enabled")
print("   6. LISTEN/NOTIFY working")
print("   7. Handler imported")
print("\n💡 If main.py is running:")
print("   - Run: python trigger_alarm_test.py")
print("   - Select an event")
print("   - Watch main.py terminal for handler logs")
print("=" * 80)
