"""
Check event b6becba5 AUTOCALLED state
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

event_id = 'b6becba5-6517-4a3b-8216-98186ff43997'

cur.execute("""
    SELECT 
        event_id,
        event_type,
        lifecycle_state,
        created_at,
        last_action_at,
        notes
    FROM event_detections 
    WHERE event_id = %s
""", (event_id,))

row = cur.fetchone()

if row:
    print(f"""
Event Details:
--------------
Event ID: {row[0]}
Type: {row[1]}
State: {row[2]}
Created: {row[3]}
Last Action: {row[4]}
Notes: {row[5]}
""")
else:
    print(f"Event {event_id} not found")

# Check lifecycle_state enum values
cur.execute("""
    SELECT enumlabel 
    FROM pg_enum 
    WHERE enumtypid = (
        SELECT oid FROM pg_type WHERE typname = 'lifecycle_state'
    )
    ORDER BY enumsortorder
""")

print("\nValid lifecycle_state values:")
for (state,) in cur.fetchall():
    print(f"  - {state}")

conn.close()
