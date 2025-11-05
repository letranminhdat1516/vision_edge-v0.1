"""
Check trigger function details
"""

from dotenv import load_dotenv
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

load_dotenv()

print("=" * 80)
print("🔍 CHECKING TRIGGER FUNCTION: notify_alarm_trigger()")
print("=" * 80)

db_service = PostgreSQLHealthcareService()

conn = db_service.get_connection()
cursor = conn.cursor()

# Get function definition
print("\n📋 Getting function source code...")
cursor.execute("""
    SELECT 
        pg_get_functiondef(oid) as function_definition
    FROM pg_proc
    WHERE proname = 'notify_alarm_trigger'
""")

func = cursor.fetchone()

if func:
    if isinstance(func, dict):
        definition = func['function_definition']
    else:
        definition = func[0]
    
    print("\n✅ Function found!")
    print("=" * 80)
    print(definition)
    print("=" * 80)
    
    # Analyze the function
    print("\n📊 ANALYSIS:")
    print("-" * 80)
    
    # Check channel name
    if 'emergency_alarm' in definition.lower():
        print("✅ Sends to 'emergency_alarm_channel' (or similar)")
    elif 'system_alarm' in definition.lower():
        print("⚠️  Sends to 'system_alarm_channel'")
    else:
        print("⚠️  Unknown channel name")
    
    # Check conditions
    if 'manual_emergency' in definition.lower():
        print("✅ Triggers on manual_emergency events")
    else:
        print("⚠️  May not trigger on manual_emergency")
    
    if 'ALARM_ACTIVATED' in definition:
        print("✅ Triggers on ALARM_ACTIVATED state")
    else:
        print("⚠️  May not trigger on ALARM_ACTIVATED")
    
    print("-" * 80)
    
else:
    print("❌ Function not found!")

db_service.return_connection(conn)

print("\n✅ Done!")
