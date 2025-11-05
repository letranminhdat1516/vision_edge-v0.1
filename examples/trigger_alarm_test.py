"""
Standalone Alarm Trigger Test
Update an event to trigger alarm while main.py is running
"""

import os
from dotenv import load_dotenv
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

load_dotenv()

print("=" * 80)
print("🧪 STANDALONE ALARM TRIGGER TEST")
print("=" * 80)
print("\n⚠️  Make sure main.py is RUNNING before using this!")
print("   main.py includes the alarm handler that will play the sound")
print("=" * 80)

# Get user ID
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')
print(f"\n👤 User ID: {user_id}")

# Connect to database
print("\n💾 Connecting to PostgreSQL...")
db_service = PostgreSQLHealthcareService()
conn = db_service.get_connection()
cursor = conn.cursor()

# Find recent events
print(f"\n🔍 Finding recent events (last 24 hours)...")
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
      AND lifecycle_state != 'ALARM_ACTIVATED'
      AND detected_at > NOW() - INTERVAL '24 hours'
    ORDER BY detected_at DESC
    LIMIT 15
""", (user_id,))

events = cursor.fetchall()

if not events:
    print("❌ No suitable events found!")
    print("\n💡 To create test events:")
    print("   1. Run main.py")
    print("   2. Press 'e' key to create random test events")
    print("   3. Then run this script again")
    db_service.return_connection(conn)
    exit(1)

# Display events
print(f"\n📋 Found {len(events)} event(s):")
print("-" * 80)

for i, event in enumerate(events, 1):
    if isinstance(event, dict):
        event_id = str(event['event_id'])[:8]
        event_type = event['event_type']
        state = event['lifecycle_state']
        description = event.get('event_description', 'No description')[:60]
        confidence = event.get('confidence_score', 0)
    else:
        event_id = str(event[0])[:8]
        event_type = event[1]
        state = event[2]
        description = (event[3] if event[3] else 'No description')[:60]
        confidence = event[4] if len(event) > 4 and event[4] else 0
    
    print(f"{i:2d}. {event_id}... | {event_type:20s} | {state}")
    print(f"    📝 {description}...")
    print(f"    📊 Confidence: {confidence:.1%}")
    print("-" * 80)

# Get user selection
print("\n👉 Select event number (1-15) or 'q' to quit: ", end='')
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
    print("🔥 TRIGGERING ALARM")
    print("=" * 80)
    print(f"📌 Event ID: {event_id}")
    print(f"📌 Event Type: {event_type}")
    print(f"📌 Action: UPDATE lifecycle_state → ALARM_ACTIVATED")
    print("=" * 80)
    
    print("\n⚡ What happens:")
    print("   1. This script: UPDATE event_detections")
    print("   2. Database trigger fires")
    print("   3. NOTIFY sent to 'system_alarm_channel'")
    print("   4. main.py handler receives notification")
    print("   5. 🔊 Alarm plays!")
    print("\n   💡 Watch main.py terminal for handler logs!")
    print("=" * 80)
    
    confirm = input("\n✅ Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        db_service.return_connection(conn)
        exit(0)
    
    # Execute UPDATE
    print("\n⚡ Executing UPDATE...")
    cursor.execute("""
        UPDATE event_detections
        SET 
            lifecycle_state = 'ALARM_ACTIVATED',
            last_action_at = NOW()
        WHERE event_id = %s
    """, (event_id,))
    
    conn.commit()
    print("✅ UPDATE executed!")
    print("\n📡 Notification sent to database trigger")
    print("🔊 Check main.py terminal - alarm should play now!")
    
    db_service.return_connection(conn)
    
    print("\n" + "=" * 80)
    print("✅ Done! Alarm should be playing in main.py")
    print("=" * 80)
    
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
