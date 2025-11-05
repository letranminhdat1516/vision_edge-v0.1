"""
Quick Test - UPDATE event với handler running
"""

import asyncio
import threading
from dotenv import load_dotenv
import os
import logging

# Setup logging để thấy handler logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

print("=" * 80)
print("🧪 QUICK ALARM TEST - Handler + UPDATE")
print("=" * 80)

# Import
from infrastructure.services.audio_alert_service import audio_alert_service
from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

print("\n💾 Connecting...")
db_service = PostgreSQLHealthcareService()
emergency_alarm_handler.set_postgresql_service(db_service)

# Check audio
audio_status = audio_alert_service.get_status()
if not audio_status['enabled']:
    print("❌ Audio disabled!")
    exit(1)

print(f"✅ Audio: {audio_status['available_devices']} devices")

# Start handler
print("\n🎧 Starting handler...")

def run_handler():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(emergency_alarm_handler.start_listening())

handler_thread = threading.Thread(target=run_handler, daemon=True)
handler_thread.start()

import time
time.sleep(3)  # Wait for handler to connect

print("✅ Handler listening on 'system_alarm_channel'")

# Get user
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')

# Find event
print(f"\n🔍 Finding events for user {user_id}...")
conn = db_service.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT event_id, event_type, lifecycle_state
    FROM event_detections
    WHERE user_id = %s
      AND lifecycle_state != 'ALARM_ACTIVATED'
      AND detected_at > NOW() - INTERVAL '24 hours'
    ORDER BY detected_at DESC
    LIMIT 10
""", (user_id,))

events = cursor.fetchall()

if not events:
    print("❌ No events found!")
    db_service.return_connection(conn)
    exit(1)

print(f"\n📋 Found {len(events)} event(s):")
for i, ev in enumerate(events, 1):
    if isinstance(ev, dict):
        ev_id = str(ev['event_id'])[:8]
        ev_type = ev['event_type']
        state = ev['lifecycle_state']
    else:
        ev_id = str(ev[0])[:8]
        ev_type = ev[1]
        state = ev[2]
    print(f"   {i}. {ev_id}... | {ev_type:20s} | {state}")

choice = input("\n👉 Select event to trigger alarm (1-10): ").strip()

try:
    idx = int(choice) - 1
    if idx < 0 or idx >= len(events):
        print("❌ Invalid!")
        exit(1)
    
    selected = events[idx]
    event_id = selected['event_id'] if isinstance(selected, dict) else selected[0]
    
    print("\n" + "=" * 80)
    print("🔥 TRIGGERING ALARM...")
    print("=" * 80)
    print(f"   Event ID: {event_id}")
    print(f"\n   1. UPDATE lifecycle_state → ALARM_ACTIVATED")
    print(f"   2. Trigger fires → NOTIFY to 'system_alarm_channel'")
    print(f"   3. Handler receives → Processes")
    print(f"   4. 🔊 ALARM PLAYS!")
    print("=" * 80)
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        exit(0)
    
    print("\n⚡ Executing UPDATE...")
    cursor.execute("""
        UPDATE event_detections
        SET lifecycle_state = 'ALARM_ACTIVATED', last_action_at = NOW()
        WHERE event_id = %s
    """, (event_id,))
    
    conn.commit()
    print("✅ UPDATE executed!")
    
    print("\n⏳ Waiting for handler to process (3 seconds)...")
    print("   Watch for '🔔 NOTIFICATION RECEIVED' above!")
    
    time.sleep(20)
    
    # Check if processed
    cursor.execute("""
        SELECT lifecycle_state, notes
        FROM event_detections
        WHERE event_id = %s
    """, (event_id,))
    
    result = cursor.fetchone()
    if result:
        if isinstance(result, dict):
            state = result['lifecycle_state']
            notes = result.get('notes', '')
        else:
            state, notes = result
        
        print("\n📊 Result:")
        print(f"   State: {state}")
        if notes and 'ALARM ACTIVATED' in notes:
            print("   ✅ ✅ ✅ ALARM WAS TRIGGERED! ✅ ✅ ✅")
            print(f"   Notes: {notes[:150]}")
        elif state == 'ACKED':
            print("   ✅ ✅ ✅ ALARM WAS TRIGGERED! ✅ ✅ ✅")
        else:
            print("   ⚠️  Check handler logs above")
            print("   Handler should show: 🔔 NOTIFICATION RECEIVED")
    
    db_service.return_connection(conn)
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
    
    # Keep running for a bit
    print("\n💡 Handler still running in background...")
    print("   Press Ctrl+C to exit")
    
    while True:
        time.sleep(1)
    
except ValueError:
    print("❌ Invalid input!")
except KeyboardInterrupt:
    print("\n\n🛑 Stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
