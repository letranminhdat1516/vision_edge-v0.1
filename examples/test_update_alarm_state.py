"""
Simple Test - Update Event to ALARM_ACTIVATED (Test Trigger)
Chỉ test trigger bằng cách UPDATE lifecycle_state
"""

import uuid
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🧪 SIMPLE TRIGGER TEST - UPDATE TO ALARM_ACTIVATED")
print("=" * 80)

# Import database service only
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

print("\n💾 Connecting to database...")
db_service = PostgreSQLHealthcareService()
print("✅ Database connected!")

user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')
print(f"\n👤 User ID: {user_id}")

def list_recent_events():
    """Show recent events that can be updated"""
    print("\n🔍 Finding recent events...")
    
    conn = db_service.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            event_id, 
            event_type, 
            event_description, 
            lifecycle_state,
            status,
            detected_at
        FROM event_detections
        WHERE user_id = %s
          AND lifecycle_state != 'ALARM_ACTIVATED'
          AND detected_at > NOW() - INTERVAL '24 hours'
        ORDER BY detected_at DESC
        LIMIT 20
    """, (user_id,))
    
    events = cursor.fetchall()
    db_service.return_connection(conn)
    
    if not events:
        print("❌ No events found in last 24 hours!")
        print("\n💡 Create an event first:")
        print("   Option 1: Run camera system to detect fall/seizure")
        print("   Option 2: Insert manual_emergency event in database")
        return None
    
    print(f"\n📋 Found {len(events)} event(s):")
    print("-" * 80)
    
    for i, event in enumerate(events, 1):
        if isinstance(event, dict):
            ev_id = event['event_id']
            ev_type = event['event_type']
            desc = event.get('event_description', 'No description')
            lifecycle = event['lifecycle_state']
            status = event.get('status', 'N/A')
            detected = event['detected_at']
        else:
            ev_id, ev_type, desc, lifecycle, status, detected = event
            desc = desc or 'No description'
        
        # Truncate description
        desc_short = desc[:60] + "..." if len(desc) > 60 else desc
        
        print(f"{i:2d}. Event ID: {str(ev_id)[:8]}...")
        print(f"    Type: {ev_type:20s} | State: {lifecycle:15s} | Status: {status}")
        print(f"    Time: {detected}")
        print(f"    Desc: {desc_short}")
        print("-" * 80)
    
    return events

def update_to_alarm_activated(event_id):
    """Update event lifecycle_state to ALARM_ACTIVATED"""
    print("\n🔄 UPDATING EVENT TO ALARM_ACTIVATED...")
    print(f"   Event ID: {event_id}")
    print("-" * 80)
    
    conn = db_service.get_connection()
    cursor = conn.cursor()
    
    try:
        # Get current state
        cursor.execute("""
            SELECT lifecycle_state, event_type, event_description
            FROM event_detections
            WHERE event_id = %s
        """, (event_id,))
        
        current = cursor.fetchone()
        if not current:
            print("❌ Event not found!")
            return False
        
        if isinstance(current, dict):
            old_state = current['lifecycle_state']
            ev_type = current['event_type']
            desc = current.get('event_description', '')
        else:
            old_state, ev_type, desc = current
        
        print(f"\n📊 Current State:")
        print(f"   Lifecycle: {old_state}")
        print(f"   Type: {ev_type}")
        print(f"   Description: {desc[:100]}")
        
        if old_state == 'ALARM_ACTIVATED':
            print("\n⚠️  Already in ALARM_ACTIVATED state!")
            print("   Choose another event or change it to other state first")
            return False
        
        print("\n⚡ Executing UPDATE...")
        print("   This will trigger PostgreSQL TRIGGER (if exists)")
        print("   Trigger should send NOTIFY to 'emergency_alarm_channel'")
        
        # UPDATE - This will fire the trigger!
        cursor.execute("""
            UPDATE event_detections
            SET 
                lifecycle_state = 'ALARM_ACTIVATED',
                last_action_at = NOW()
            WHERE event_id = %s
        """, (event_id,))
        
        conn.commit()
        
        print("\n✅ UPDATE EXECUTED!")
        print("-" * 80)
        print("📡 Expected flow:")
        print("   1. ✅ UPDATE statement executed")
        print("   2. 🔥 PostgreSQL TRIGGER fired (if trigger exists)")
        print("   3. 📢 NOTIFY sent to 'emergency_alarm_channel'")
        print("   4. 🎧 Handler receives notification (if listening)")
        print("   5. 🔊 Alarm plays via Bluetooth speaker")
        print("-" * 80)
        
        # Verify update
        cursor.execute("""
            SELECT lifecycle_state, last_action_at
            FROM event_detections
            WHERE event_id = %s
        """, (event_id,))
        
        result = cursor.fetchone()
        if result:
            if isinstance(result, dict):
                new_state = result['lifecycle_state']
                updated_at = result['last_action_at']
            else:
                new_state, updated_at = result
            
            print(f"\n📊 Verified in database:")
            print(f"   New lifecycle_state: {new_state}")
            print(f"   Updated at: {updated_at}")
            
            if new_state == 'ALARM_ACTIVATED':
                print("\n✅ ✅ ✅ STATE CHANGED SUCCESSFULLY! ✅ ✅ ✅")
            else:
                print(f"\n⚠️  Unexpected state: {new_state}")
        
        db_service.return_connection(conn)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        db_service.return_connection(conn)
        return False

def check_trigger_exists():
    """Check if trigger exists in database"""
    print("\n🔍 Checking if trigger exists...")
    
    conn = db_service.get_connection()
    cursor = conn.cursor()
    
    # Simple check - just look for triggers on event_detections
    cursor.execute("""
        SELECT 
            trigger_name, 
            event_manipulation,
            action_timing
        FROM information_schema.triggers
        WHERE event_object_table = 'event_detections'
          AND trigger_schema = 'public'
    """)
    
    triggers = cursor.fetchall()
    
    if triggers:
        print(f"✅ Found {len(triggers)} trigger(s) on event_detections!")
        
        for trigger in triggers:
            if isinstance(trigger, dict):
                name = trigger['trigger_name']
                event = trigger['event_manipulation']
                timing = trigger['action_timing']
            else:
                name, event, timing = trigger
            
            print(f"\n   🔧 Trigger: {name}")
            print(f"      Events: {event}")
            print(f"      Timing: {timing}")
            
            # Check specific trigger
            if 'notify_alarm' in name.lower() or 'emergency' in name.lower():
                print(f"      ✅ This is an alarm trigger!")
        
        db_service.return_connection(conn)
        return True
    else:
        print("❌ Trigger NOT FOUND!")
        db_service.return_connection(conn)
        return False

def main():
    print("\n" + "=" * 80)
    print("📋 MENU")
    print("=" * 80)
    print("1. 📋 List recent events")
    print("2. 🔄 Update event to ALARM_ACTIVATED (trigger test)")
    print("3. 🔍 Check if trigger exists in database")
    print("4. ❌ Exit")
    print("=" * 80)
    
    choice = input("\n👉 Enter choice (1-4): ").strip()
    
    if choice == '1':
        events = list_recent_events()
        if events:
            print("\n💡 Use option 2 to update one of these events")
        
    elif choice == '2':
        events = list_recent_events()
        if not events:
            print("\n❌ No events available to update!")
            return
        
        print("\n" + "=" * 80)
        event_choice = input("👉 Enter event number to update (or 0 to cancel): ").strip()
        
        try:
            idx = int(event_choice) - 1
            if idx < 0 or idx >= len(events):
                print("❌ Invalid choice!")
                return
            
            selected = events[idx]
            selected_id = selected['event_id'] if isinstance(selected, dict) else selected[0]
            
            # Confirm
            print("\n⚠️  This will UPDATE the event to ALARM_ACTIVATED")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                update_to_alarm_activated(selected_id)
                
                print("\n💡 NEXT STEPS:")
                print("   1. Check if alarm played (if handler is running)")
                print("   2. Check handler logs for NOTIFICATION RECEIVED")
                print("   3. If no alarm, check:")
                print("      - Is trigger created in database? (option 3)")
                print("      - Is handler running and connected?")
                print("      - Check handler logs for errors")
            else:
                print("❌ Cancelled")
                
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == '3':
        check_trigger_exists()
        
        print("\n💡 If trigger doesn't exist:")
        print("   File: database/create_emergency_alarm_trigger.sql")
        print("   Run this SQL in Supabase Dashboard → SQL Editor")
    
    elif choice == '4':
        print("\n👋 Exiting...")
        return False
    
    else:
        print("\n❌ Invalid choice!")
    
    return True

if __name__ == "__main__":
    try:
        # Check trigger first
        print("\n" + "=" * 80)
        has_trigger = check_trigger_exists()
        
        if not has_trigger:
            print("\n⚠️  WARNING: Trigger not found!")
            print("   Updates will work, but NO notification will be sent")
            print("   Handler will NOT receive events")
            cont = input("\nContinue anyway? (y/n): ").strip().lower()
            if cont != 'y':
                print("❌ Exiting. Create trigger first!")
                exit(0)
        
        # Main loop
        while True:
            if not main():
                break
            
            cont = input("\n⏸️  Press Enter to continue (or 'q' to quit): ").strip()
            if cont.lower() == 'q':
                break
        
        print("\n" + "=" * 80)
        print("👋 Test completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
