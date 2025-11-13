"""
Quick Test - Alarm Stop Conditions
Test các điều kiện dừng alarm
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

from service.postgresql_healthcare_service import PostgreSQLHealthcareService

print("=" * 80)
print("🧪 ALARM STOP CONDITIONS TEST")
print("=" * 80)
print("\n⚠️  Prerequisites:")
print("   1. main.py is RUNNING")
print("   2. Database trigger 'alarm_stop_trigger' is created")
print("   3. Alarm is currently playing")
print("=" * 80)

# Get user ID
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')
print(f"\n👤 User ID: {user_id}")

# Connect to database
print("\n💾 Connecting to PostgreSQL...")
db_service = PostgreSQLHealthcareService()
conn = db_service.get_connection()
cursor = conn.cursor()

# Find ALARM_ACTIVATED events
print(f"\n🔍 Finding events with ALARM_ACTIVATED state...")
cursor.execute("""
    SELECT 
        event_id, 
        event_type, 
        lifecycle_state,
        event_description,
        confidence_score,
        detected_at
    FROM event_detections
    WHERE user_id = %s
      AND lifecycle_state = 'ALARM_ACTIVATED'
    ORDER BY detected_at DESC
    LIMIT 10
""", (user_id,))

events = cursor.fetchall()

if not events:
    print("❌ No events with ALARM_ACTIVATED state found!")
    print("\n💡 To test alarm stop:")
    print("   1. Run: python examples/trigger_alarm_test.py")
    print("   2. Select an event to trigger alarm")
    print("   3. Then run this script again to stop alarm")
    db_service.return_connection(conn)
    exit(1)

# Display events
print(f"\n📋 Found {len(events)} event(s) with ALARM_ACTIVATED:")
print("-" * 80)

for i, event in enumerate(events, 1):
    if isinstance(event, dict):
        event_id = str(event['event_id'])[:8]
        event_type = event['event_type']
        description = event.get('event_description', 'No description')[:60]
        confidence = event.get('confidence_score', 0)
    else:
        event_id = str(event[0])[:8]
        event_type = event[1]
        description = (event[3] if event[3] else 'No description')[:60]
        confidence = float(event[4]) if len(event) > 4 and event[4] else 0
    
    print(f"{i:2d}. {event_id}... | {event_type:20s} | ALARM_ACTIVATED")
    print(f"    📝 {description}...")
    print(f"    📊 Confidence: {confidence:.1%}")
    print("-" * 80)

# Get user selection
print("\n👉 Select event to STOP ALARM (1-10) or 'q' to quit: ", end='')
choice = input().strip()

if choice.lower() == 'q':
    print("👋 Cancelled")
    db_service.return_connection(conn)
    exit(0)

try:
    idx = int(choice) - 1
    if idx < 0 or idx >= len(events):
        print("❌ Invalid selection!")
        db_service.return_connection(conn)
        exit(1)
    
    selected = events[idx]
    event_id = selected['event_id'] if isinstance(selected, dict) else selected[0]
    event_type = selected['event_type'] if isinstance(selected, dict) else selected[1]
    
    print("\n" + "=" * 80)
    print("🔇 STOPPING ALARM")
    print("=" * 80)
    print(f"📌 Event ID: {event_id}")
    print(f"📌 Event Type: {event_type}")
    print(f"📌 Action: UPDATE lifecycle_state → DISMISSED")
    print("=" * 80)
    
    print("\n⚡ What happens:")
    print("   1. This script: UPDATE event_detections")
    print("   2. Database trigger 'alarm_stop_trigger' fires")
    print("   3. NOTIFY sent to 'system_alarm_stop_channel'")
    print("   4. main.py handler receives notification")
    print("   5. 🔇 Alarm STOPS!")
    print("\n   💡 Watch main.py terminal for handler logs!")
    print("=" * 80)
    
    confirm = input("\n✅ Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        db_service.return_connection(conn)
        exit(0)
    
    # Execute UPDATE to DISMISSED
    print("\n⚡ Executing UPDATE...")
    cursor.execute("""
        UPDATE event_detections
        SET 
            lifecycle_state = 'DISMISSED',
            dismissed_at = NOW(),
            is_canceled = TRUE,
            notes = COALESCE(notes, '') || ' | Alarm dismissed via test script',
            last_action_at = NOW()
        WHERE event_id = %s
    """, (event_id,))
    
    conn.commit()
    print("✅ UPDATE executed!")
    print("\n📡 Notification sent to database trigger")
    print("🔇 Check main.py terminal - alarm should STOP now!")
    
    db_service.return_connection(conn)
    
    print("\n" + "=" * 80)
    print("✅ Done! Alarm should have stopped in main.py")
    print("=" * 80)
    
    print("\n🔍 Verify in database:")
    print("   SELECT event_id, lifecycle_state, dismissed_at")
    print("   FROM event_detections")
    print(f"   WHERE event_id = '{event_id}';")
    
except ValueError:
    print("❌ Invalid input - must be a number!")
    db_service.return_connection(conn)
except KeyboardInterrupt:
    print("\n\n👋 Interrupted")
    db_service.return_connection(conn)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    db_service.return_connection(conn)
