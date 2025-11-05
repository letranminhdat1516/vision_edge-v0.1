"""
Check all triggers in database
"""

from dotenv import load_dotenv
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

load_dotenv()

print("=" * 80)
print("🔍 CHECKING ALL TRIGGERS IN DATABASE")
print("=" * 80)

db_service = PostgreSQLHealthcareService()

print("\n💾 Connecting to database...")
conn = db_service.get_connection()
cursor = conn.cursor()

# Get all triggers
print("\n📋 Listing all triggers...")
cursor.execute("""
    SELECT 
        trigger_name,
        event_object_table,
        event_manipulation,
        action_timing,
        action_statement
    FROM information_schema.triggers
    WHERE trigger_schema = 'public'
    ORDER BY event_object_table, trigger_name
""")

triggers = cursor.fetchall()

if not triggers:
    print("❌ No triggers found in database!")
else:
    print(f"\n✅ Found {len(triggers)} trigger(s):")
    print("=" * 80)
    
    for trigger in triggers:
        if isinstance(trigger, dict):
            name = trigger['trigger_name']
            table = trigger['event_object_table']
            event = trigger['event_manipulation']
            timing = trigger['action_timing']
            statement = trigger['action_statement']
        else:
            name, table, event, timing, statement = trigger
        
        print(f"\n🔧 Trigger: {name}")
        print(f"   Table: {table}")
        print(f"   Event: {event}")
        print(f"   Timing: {timing}")
        print(f"   Statement: {statement[:100]}...")
        print("-" * 80)

# Check specifically for emergency_alarm_trigger
print("\n🎯 Checking for 'emergency_alarm_trigger' on event_detections...")
cursor.execute("""
    SELECT 
        trigger_name,
        event_manipulation,
        action_statement
    FROM information_schema.triggers
    WHERE trigger_name = 'emergency_alarm_trigger'
      AND event_object_table = 'event_detections'
""")

emergency_trigger = cursor.fetchone()

if emergency_trigger:
    print("✅ emergency_alarm_trigger EXISTS!")
    if isinstance(emergency_trigger, dict):
        print(f"   Name: {emergency_trigger['trigger_name']}")
        print(f"   Events: {emergency_trigger['event_manipulation']}")
    else:
        print(f"   Name: {emergency_trigger[0]}")
        print(f"   Events: {emergency_trigger[1]}")
else:
    print("❌ emergency_alarm_trigger NOT FOUND!")
    print("\n💡 This trigger is needed for emergency alarm system")
    print("   File: database/create_emergency_alarm_trigger.sql")

# Check what your existing trigger does
print("\n" + "=" * 80)
print("🔍 ANALYZING EXISTING TRIGGERS ON event_detections...")
print("=" * 80)

cursor.execute("""
    SELECT 
        trigger_name,
        event_manipulation,
        action_timing,
        action_statement
    FROM information_schema.triggers
    WHERE event_object_table = 'event_detections'
    ORDER BY trigger_name
""")

event_triggers = cursor.fetchall()

if event_triggers:
    print(f"\n✅ Found {len(event_triggers)} trigger(s) on event_detections:")
    
    for trig in event_triggers:
        if isinstance(trig, dict):
            name = trig['trigger_name']
            event = trig['event_manipulation']
            timing = trig['action_timing']
            statement = trig['action_statement']
        else:
            name, event, timing, statement = trig
        
        print(f"\n🔧 {name}")
        print(f"   Event: {event}")
        print(f"   Timing: {timing}")
        
        # Check if it does NOTIFY
        if 'pg_notify' in statement.lower() or 'notify' in statement.lower():
            print(f"   ✅ Contains NOTIFY!")
            
            # Try to extract channel name
            if 'emergency_alarm' in statement.lower():
                print(f"   ✅ Sends to 'emergency_alarm_channel'")
            else:
                print(f"   ⚠️  Sends to different channel")
        else:
            print(f"   ❌ Does NOT contain NOTIFY")
        
        print(f"   Statement preview:")
        # Show first 200 chars
        preview = statement[:200].replace('\n', '\n      ')
        print(f"      {preview}...")
        print("-" * 80)
else:
    print("❌ No triggers found on event_detections table!")

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

cursor.execute("""
    SELECT COUNT(*) FROM information_schema.triggers
    WHERE event_object_table = 'event_detections'
      AND action_statement LIKE '%pg_notify%'
      AND action_statement LIKE '%emergency_alarm%'
""")

notify_count = cursor.fetchone()
if notify_count:
    # Handle both dict (RealDictCursor) and tuple (regular cursor)
    if isinstance(notify_count, dict):
        count = notify_count['count']
    else:
        count = notify_count[0]
else:
    count = 0

if count > 0:
    print(f"✅ Found {count} trigger(s) that send NOTIFY to emergency_alarm_channel")
    print("   Emergency alarm system should work!")
else:
    print("❌ No triggers found that send NOTIFY to emergency_alarm_channel")
    print("   Emergency alarm system will NOT receive notifications!")
    print("\n💡 To fix:")
    print("   1. Open Supabase Dashboard → SQL Editor")
    print("   2. Run SQL from: database/create_emergency_alarm_trigger.sql")

db_service.return_connection(conn)

print("\n" + "=" * 80)
print("✅ Check completed!")
print("=" * 80)
