import psycopg2

conn = psycopg2.connect('postgresql://postgres.undznprwlqjpnxqsgyiv:Phanthihoaidiem7903@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres')
cursor = conn.cursor()

# List tables
cursor.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name\")
tables = cursor.fetchall()
print('📋 Available tables:')
for t in tables:
    print(f'   - {t[0]}')
print()

# Check event in healthcare_events
event_id = '6dfd3e4e-e9cb-49a8-8535-14967d122e55'
print(f'🔍 Checking event {event_id}...\n')

cursor.execute('SELECT event_id, event_type, confidence, created_at, event_description FROM healthcare_events WHERE event_id = %s', (event_id,))
event = cursor.fetchone()

if event:
    print('📋 Event Details:')
    print(f'   Event ID: {event[0]}')
    print(f'   Type: {event[1]}')
    print(f'   Confidence: {event[2]}')
    print(f'   Created: {event[3]}')
    desc = event[4] if event[4] else 'None'
    print(f'   Description: {desc[:100]}...' if len(desc) > 100 else f'   Description: {desc}')
    print()
    
    # Check snapshots
    cursor.execute('''
        SELECT s.snapshot_id, s.event_type, s.confidence, s.created_at, 
               COUNT(si.image_id) as image_count
        FROM snapshots s
        LEFT JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
        WHERE s.event_id = %s
        GROUP BY s.snapshot_id, s.event_type, s.confidence, s.created_at
        ORDER BY s.created_at
    ''', (event_id,))
    
    snapshots = cursor.fetchall()
    
    if snapshots:
        print(f'📸 Found {len(snapshots)} snapshot(s):\n')
        for snap in snapshots:
            print(f'   Snapshot ID: {snap[0]}')
            print(f'   Type: {snap[1]}')
            print(f'   Confidence: {snap[2]}')
            print(f'   Created: {snap[3]}')
            print(f'   Images: {snap[4]}')
            print()
            
            # Get image details
            cursor.execute('''
                SELECT image_id, is_primary, image_path, file_size, created_at
                FROM snapshot_images
                WHERE snapshot_id = %s
                ORDER BY created_at
            ''', (snap[0],))
            
            images = cursor.fetchall()
            if images:
                print(f'   📷 Images ({len(images)}):')
                for idx, img in enumerate(images, 1):
                    print(f'      {idx}. ID: {img[0]}')
                    print(f'         Primary: {img[1]}')
                    print(f'         Path: {img[2]}')
                    print(f'         Size: {img[3]} bytes')
                    print(f'         Created: {img[4]}')
                    print()
    else:
        print('❌ No snapshots found for this event')
else:
    print(f'❌ Event {event_id} not found')

cursor.close()
conn.close()
