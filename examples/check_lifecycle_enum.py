"""
Check lifecycle_state enum values in database
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

from service.postgresql_healthcare_service import PostgreSQLHealthcareService

db = PostgreSQLHealthcareService()

print("=" * 60)
print("Checking lifecycle_state enum values")
print("=" * 60)

conn = db.get_connection()
cursor = conn.cursor()

# Check enum values
cursor.execute("""
    SELECT enumlabel 
    FROM pg_enum 
    WHERE enumtypid = (
        SELECT oid 
        FROM pg_type 
        WHERE typname = 'event_lifecycle_enum'
    )
    ORDER BY enumsortorder;
""")

enum_values = cursor.fetchall()

print("\nCurrent event_lifecycle_enum values:")
for row in enum_values:
    print(f"  - {row['enumlabel']}")

db.return_connection(conn)

print("\n" + "=" * 60)
print("To add AUTO_CALLED, run this SQL:")
print("=" * 60)
print("""
ALTER TYPE event_lifecycle_enum ADD VALUE IF NOT EXISTS 'AUTO_CALLED';
""")
print("=" * 60)
