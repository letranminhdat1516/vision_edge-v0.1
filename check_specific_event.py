"""
Check specific event and its snapshot images
"""
import sys
sys.path.insert(0, 'src')

from service.postgresql_healthcare_service import PostgreSQLHealthcareService
from psycopg2.extras import RealDictCursor

def check_event_snapshot(event_id, snapshot_id):
    service = PostgreSQLHealthcareService()
    conn = service.get_connection()
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check event
        print('=' * 80)
        print(f'CHECKING EVENT: {event_id}')
        print('=' * 80)
        
        cursor.execute('''
            SELECT event_id, event_type, status, snapshot_id, 
                   event_description, detected_at, confidence_score
            FROM event_detections 
            WHERE event_id = %s
        ''', (event_id,))
        
        event = cursor.fetchone()
        if event:
            print(f"Event Type: {event['event_type']}")
            print(f"Status: {event['status']}")
            print(f"Snapshot ID: {event['snapshot_id']}")
            print(f"Confidence: {event['confidence_score']}")
            print(f"Time: {event['detected_at']}")
            print(f"Description: {event['event_description'][:100] if event['event_description'] else 'N/A'}")
        else:
            print('❌ EVENT NOT FOUND!')
            return
        
        # Check snapshot
        print('\n' + '=' * 80)
        print(f'CHECKING SNAPSHOT: {snapshot_id}')
        print('=' * 80)
        
        cursor.execute('''
            SELECT snapshot_id, camera_id, user_id, capture_type,
                   captured_at, is_processed, metadata
            FROM snapshots 
            WHERE snapshot_id = %s
        ''', (snapshot_id,))
        
        snapshot = cursor.fetchone()
        if snapshot:
            print(f"Camera ID: {snapshot['camera_id']}")
            print(f"User ID: {snapshot['user_id']}")
            print(f"Capture Type: {snapshot['capture_type']}")
            print(f"Captured At: {snapshot['captured_at']}")
            print(f"Is Processed: {snapshot['is_processed']}")
            meta_str = str(snapshot.get('metadata', 'N/A'))
            print(f"Metadata: {meta_str[:100] if meta_str else 'N/A'}...")
        else:
            print('❌ SNAPSHOT NOT FOUND!')
            return
        
        # Check images
        print('\n' + '=' * 80)
        print('CHECKING SNAPSHOT IMAGES')
        print('=' * 80)
        
        cursor.execute('''
            SELECT image_id, image_path, cloud_url, is_primary, 
                   file_size, created_at
            FROM snapshot_images 
            WHERE snapshot_id = %s
            ORDER BY created_at DESC
        ''', (snapshot_id,))
        
        images = cursor.fetchall()
        
        if images:
            print(f"✅ Found {len(images)} image(s):\n")
            for i, img in enumerate(images, 1):
                print(f"  📸 Image {i}:")
                print(f"     Image ID: {img['image_id']}")
                print(f"     Primary: {img['is_primary']}")
                print(f"     File Size: {img['file_size']}")
                print(f"     Created: {img['created_at']}")
                if img['cloud_url']:
                    print(f"     ✅ Cloud URL: {img['cloud_url']}")
                if img['image_path']:
                    print(f"     📁 Local Path: {img['image_path']}")
                print()
        else:
            print('❌ NO IMAGES FOUND!')
            print('\n⚠️ This snapshot has NO images associated with it.')
            print('   This means snapshot record was created but image upload failed or was skipped.')
        
        cursor.close()
        service.return_connection(conn)
        
    except Exception as e:
        print(f'❌ ERROR: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    event_id = 'f0f5da5d-67ad-4e7a-bb8c-e53a23400f90'
    snapshot_id = 'c855c8bf-f54b-42f0-90b7-af4741a60a83'
    
    check_event_snapshot(event_id, snapshot_id)
