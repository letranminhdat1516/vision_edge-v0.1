import psycopg2

conn = psycopg2.connect('postgresql://postgres.undznprwlqjpnxqsgyiv:Phanthihoaidiem7903@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres')
cursor = conn.cursor()

# Check event_detections schema
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'event_detections' ORDER BY ordinal_position")
cols = cursor.fetchall()
print('📋 event_detections columns:')
for c in cols:
    print(f'   - {c[0]} ({c[1]})')
print()

# Check snapshots schema
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'snapshots' ORDER BY ordinal_position")
cols = cursor.fetchall()
print('📋 snapshots columns:')
for c in cols:
    print(f'   - {c[0]} ({c[1]})')
print()

# Check snapshot_images schema
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'snapshot_images' ORDER BY ordinal_position")
cols = cursor.fetchall()
print('📋 snapshot_images columns:')
for c in cols:
    print(f'   - {c[0]} ({c[1]})')

cursor.close()
conn.close()
