"""
Check if NORMAL events have snapshots/images in database
"""
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from dotenv import load_dotenv
from service.postgresql_healthcare_service import PostgreSQLHealthcareService
from psycopg2.extras import RealDictCursor

# Load environment
load_dotenv()

def check_normal_events():
    """Check NORMAL events and their snapshots"""
    
    # Use PostgreSQL service
    service = PostgreSQLHealthcareService()
    conn = service.get_connection()
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get NORMAL events with snapshot info
        print("=" * 80)
        print("CHECKING NORMAL EVENTS WITH SNAPSHOTS")
        print("=" * 80)
        
        query = """
        SELECT 
            e.event_id,
            e.event_type,
            e.status,
            e.confidence_score,
            e.event_description,
            e.detected_at,
            e.snapshot_id,
            s.snapshot_id as snapshot_exists,
            COUNT(si.image_id) as image_count
        FROM event_detections e
        LEFT JOIN snapshots s ON e.snapshot_id = s.snapshot_id
        LEFT JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
        WHERE e.status = 'normal'
        GROUP BY e.event_id, e.event_type, e.status, e.confidence_score, 
                 e.event_description, e.detected_at, e.snapshot_id, 
                 s.snapshot_id
        ORDER BY e.detected_at DESC
        LIMIT 20
        """
        
        cursor.execute(query)
        events = cursor.fetchall()
        
        print(f"\n📊 Found {len(events)} NORMAL events (showing latest 20)")
        print("=" * 80)
        
        events_with_snapshots = 0
        events_without_snapshots = 0
        
        for event in events:
            print(f"\n🆔 Event ID: {event['event_id']}")
            print(f"   Type: {event['event_type']}")
            print(f"   Status: {event['status']}")
            print(f"   Confidence: {event['confidence_score']}")
            print(f"   Time: {event['detected_at']}")
            print(f"   Description: {event['event_description'][:80]}...")
            
            if event['snapshot_id']:
                print(f"   ✅ HAS SNAPSHOT: {event['snapshot_id']}")
                print(f"      📸 Images: {event['image_count']}")
                events_with_snapshots += 1
                
                # Get image URLs
                cursor.execute("""
                    SELECT image_path, cloud_url, created_at
                    FROM snapshot_images
                    WHERE snapshot_id = %s
                    ORDER BY created_at DESC
                """, (event['snapshot_id'],))
                
                images = cursor.fetchall()
                for img in images:
                    if img['cloud_url']:
                        print(f"         🔗 {img['cloud_url']}")
                    elif img['image_path']:
                        print(f"         📁 {img['image_path']}")
                    else:
                        print(f"         ⚠️ NO IMAGE DATA")
            else:
                print(f"   ❌ NO SNAPSHOT")
                events_without_snapshots += 1
        
        print("\n" + "=" * 80)
        print(f"📊 SUMMARY:")
        print(f"   Total NORMAL events: {len(events)}")
        print(f"   ✅ With snapshots: {events_with_snapshots}")
        print(f"   ❌ Without snapshots: {events_without_snapshots}")
        print(f"   📈 Success rate: {events_with_snapshots / len(events) * 100:.1f}%" if events else "")
        print("=" * 80)
        
        # Check most recent NORMAL event
        print("\n📍 MOST RECENT NORMAL EVENT:")
        print("=" * 80)
        
        cursor.execute("""
            SELECT 
                e.*,
                s.snapshot_id as has_snapshot,
                COUNT(si.image_id) as image_count
            FROM event_detections e
            LEFT JOIN snapshots s ON e.snapshot_id = s.snapshot_id
            LEFT JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
            WHERE e.event_type = 'normal_activity'
            GROUP BY e.event_id, s.snapshot_id
            ORDER BY e.detected_at DESC
            LIMIT 1
        """)
        
        latest = cursor.fetchone()
        if latest:
            print(f"Event ID: {latest['event_id']}")
            print(f"Time: {latest['detected_at']}")
            print(f"Description: {latest['event_description']}")
            print(f"Snapshot ID: {latest['snapshot_id']}")
            print(f"Has Snapshot: {'✅ YES' if latest['has_snapshot'] else '❌ NO'}")
            print(f"Image Count: {latest['image_count']}")
        
    finally:
        cursor.close()
        service.return_connection(conn)

if __name__ == '__main__':
    check_normal_events()
