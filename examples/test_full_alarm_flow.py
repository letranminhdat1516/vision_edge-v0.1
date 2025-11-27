"""
Test Full Alarm Flow: Trigger → Play → Auto-Stop
Requires main.py running in another terminal
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("🔊 FULL ALARM TEST - TRIGGER → PLAY → AUTO-STOP")
print("=" * 80)

from service.postgresql_healthcare_service import PostgreSQLHealthcareService

db_service = PostgreSQLHealthcareService()
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')

print("\n⚠️ IMPORTANT: Make sure main.py is running!")
print("   Terminal 1: python src/main.py")
print("   Terminal 2: This test")

input("\nPress Enter when main.py is running...")

def create_and_trigger_alarm():
    """Create event and trigger alarm"""
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        
        import uuid
        event_id = str(uuid.uuid4())
        snapshot_id = str(uuid.uuid4())
        
        # Create snapshot
        cursor.execute("""
            INSERT INTO snapshots (
                snapshot_id, user_id, camera_id,
                capture_type, captured_at, is_processed
            ) VALUES (
                %s, %s,
                (SELECT camera_id FROM cameras WHERE user_id = %s LIMIT 1),
                'manual', NOW(), false
            )
        """, (snapshot_id, user_id, user_id))
        
        # Create event with NOTIFIED first (like real flow)
        cursor.execute("""
            INSERT INTO event_detections (
                event_id, user_id, camera_id, event_type,
                snapshot_id, confidence_score, status, lifecycle_state,
                event_description, detected_at, created_at,
                confirmation_state, verification_status,
                escalation_count, is_canceled, notification_attempts
            ) VALUES (
                %s, %s, 
                (SELECT camera_id FROM cameras WHERE user_id = %s LIMIT 1),
                'fall', %s, 0.85, 'danger', 'NOTIFIED',
                'Test full alarm flow', NOW(), NOW(),
                'DETECTED', 'PENDING', 0, false, 0
            )
        """, (event_id, user_id, user_id, snapshot_id))
        
        conn.commit()
        print(f"\n✅ Created event: {event_id[:8]}... (NOTIFIED)")
        
        # Wait a bit
        time.sleep(1)
        
        # Now TRIGGER alarm by updating to ALARM_ACTIVATED
        print(f"\n🔊 Triggering alarm...")
        cursor.execute("""
            UPDATE event_detections
            SET lifecycle_state = 'ALARM_ACTIVATED',
                last_action_at = NOW()
            WHERE event_id = %s
        """, (event_id,))
        
        conn.commit()
        cursor.close()
        db_service.return_connection(conn)
        
        print(f"   ✅ Event updated: NOTIFIED → ALARM_ACTIVATED")
        print(f"   🔔 Database trigger should fire!")
        print(f"   🔊 Alarm should start playing in main.py!")
        
        return event_id
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

def wait_for_alarm_playing():
    """Wait for alarm to start"""
    print(f"\n⏳ Waiting 3 seconds for alarm to start...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print(f"   🔊 Alarm should be playing now!")

def simulate_two_people():
    """Simulate 2 people detected - should stop alarm"""
    print(f"\n" + "=" * 80)
    print(f"🧪 TEST: Simulating 2 people detected")
    print(f"=" * 80)
    
    print(f"\n💡 In real system:")
    print(f"   - YOLO detects 2 people in frame")
    print(f"   - main.py calls: emergency_alarm_handler.resolve_active_alarms()")
    print(f"   - Database: ALARM_ACTIVATED → RESOLVED")
    print(f"   - Alarm stops!")
    
    print(f"\n⏳ We'll resolve the alarm manually...")
    
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        
        # Resolve all ALARM_ACTIVATED events
        cursor.execute("""
            UPDATE event_detections
            SET 
                lifecycle_state = 'RESOLVED',
                last_action_at = NOW(),
                notes = COALESCE(notes, '') || '\nTest: Help arrived - 2 people detected'
            WHERE lifecycle_state = 'ALARM_ACTIVATED'
            RETURNING event_id
        """)
        
        resolved = cursor.fetchall()
        conn.commit()
        cursor.close()
        db_service.return_connection(conn)
        
        if resolved:
            print(f"\n   ✅ Resolved {len(resolved)} event(s)")
            print(f"   🔔 Stop trigger should fire!")
            print(f"   🔇 Alarm should stop in main.py!")
        else:
            print(f"\n   ⚠️ No ALARM_ACTIVATED events found")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")

# Run test
print("\n" + "=" * 80)
print("STEP 1: CREATE & TRIGGER ALARM")
print("=" * 80)

event_id = create_and_trigger_alarm()

if event_id:
    wait_for_alarm_playing()
    
    heard = input("\n👂 Do you hear the alarm? (y/n): ").strip().lower()
    
    if heard == 'y':
        print("\n✅ Great! Alarm is working!")
        
        print("\n" + "=" * 80)
        print("STEP 2: AUTO-STOP TEST")
        print("=" * 80)
        
        simulate_two_people()
        
        print(f"\n⏳ Waiting 2 seconds for alarm to stop...")
        time.sleep(2)
        
        stopped = input("\n🔇 Did the alarm stop? (y/n): ").strip().lower()
        
        if stopped == 'y':
            print("\n🎉 SUCCESS! Full alarm flow works!")
            print("\n✅ Tested:")
            print("   1. Create event (NOTIFIED)")
            print("   2. Trigger alarm (ALARM_ACTIVATED)")
            print("   3. Alarm plays")
            print("   4. Auto-stop when resolved (RESOLVED)")
        else:
            print("\n❌ Alarm didn't stop")
            print("   Check main.py logs for errors")
    else:
        print("\n❌ Alarm not playing!")
        print("\n💡 Troubleshooting:")
        print("   1. Is main.py running?")
        print("   2. Check main.py logs for:")
        print("      '🔔 NOTIFICATION RECEIVED'")
        print("      '🔊 ALARM PLAYING'")
        print("   3. Audio device connected?")

print("\n" + "=" * 80)
