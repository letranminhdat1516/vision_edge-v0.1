import psycopg2

conn = psycopg2.connect('postgresql://postgres.undznprwlqjpnxqsgyiv:Phanthihoaidiem7903@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres')
cursor = conn.cursor()

# Check event in event_detections
event_id = '949477ef-b97f-4b00-8a53-6cb9c73d2919'
print(f'🔍 Checking event {event_id}...\n')

cursor.execute('SELECT event_id, event_type, confidence_score, created_at, event_description, snapshot_id FROM event_detections WHERE event_id = %s', (event_id,))
event = cursor.fetchone()

if event:
    print('📋 Event Details:')
    print(f'   Event ID: {event[0]}')
    print(f'   Type: {event[1]}')
    print(f'   Confidence: {event[2]}')
    print(f'   Created: {event[3]}')
    desc = event[4] if event[4] else 'None'
    print(f'   Description: {desc[:100]}...' if len(desc) > 100 else f'   Description: {desc}')
    print(f'   Snapshot ID: {event[5]}')
    print()
    
    # Check snapshots by snapshot_id from event
    if event[5]:
        cursor.execute('''
            SELECT s.snapshot_id, s.capture_type, s.captured_at, s.metadata,
                   COUNT(si.image_id) as image_count
            FROM snapshots s
            LEFT JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
            WHERE s.snapshot_id = %s
            GROUP BY s.snapshot_id, s.capture_type, s.captured_at, s.metadata
        ''', (event[5],))
        
        snapshot = cursor.fetchone()
        
        if snapshot:
            print(f'📸 Snapshot found:\n')
            print(f'   Snapshot ID: {snapshot[0]}')
            print(f'   Capture Type: {snapshot[1]}')
            print(f'   Captured: {snapshot[2]}')
            print(f'   Images: {snapshot[4]}')
            print()
            
            # Get image details
            cursor.execute('''
                SELECT image_id, is_primary, image_path, file_size, created_at, cloud_url
                FROM snapshot_images
                WHERE snapshot_id = %s
                ORDER BY created_at
            ''', (snapshot[0],))
            
            images = cursor.fetchall()
            if images:
                print(f'   📷 Images ({len(images)}):')
                for idx, img in enumerate(images, 1):
                    print(f'      {idx}. ID: {img[0]}')
                    print(f'         Primary: {img[1]}')
                    print(f'         Path: {img[2]}')
                    print(f'         Size: {img[3]} bytes')
                    print(f'         Created: {img[4]}')
                    url = img[5] if img[5] else 'None'
                    print(f'         URL: {url[:80]}...' if len(url) > 80 else f'         URL: {url}')
                    print()
        else:
            print('❌ Snapshot not found')
    else:
        print('❌ No snapshot_id in event')
else:
    print(f'❌ Event {event_id} not found')

cursor.close()
conn.close()
