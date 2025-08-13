"""
Check Database Records - Healthcare Events
"""
import psycopg2
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

def check_database_records():
    """Check all healthcare events in database"""
    try:
        conn = psycopg2.connect(
            host="aws-1-ap-southeast-1.pooler.supabase.com",
            port=5432,
            database="postgres",
            user="postgres.undznprwlqjpnxqsgyiv",
            password=os.getenv("DB_PASSWORD", "12345678")
        )
        
        cur = conn.cursor()
        
        print("🗃️ HEALTHCARE DATABASE RECORDS")
        print("=" * 80)
        
        # 1. Check event_detections table
        print("\n📋 1. EVENT_DETECTIONS TABLE:")
        print("-" * 50)
        cur.execute("""
            SELECT event_id, event_type, event_description, confidence_score, 
                   detected_at, created_at, camera_id, user_id, status
            FROM event_detections 
            ORDER BY created_at DESC 
            LIMIT 10;
        """)
        
        events = cur.fetchall()
        if events:
            for i, event in enumerate(events, 1):
                event_id, event_type, description, confidence, detected_at, created_at, camera_id, user_id, status = event
                print(f"   {i}. Event ID: {event_id}")
                print(f"      Type: {event_type}")
                print(f"      Description: {description}")
                print(f"      Confidence: {confidence:.2%}")
                print(f"      Status: {status}")
                print(f"      Camera: {camera_id}")
                print(f"      User: {user_id}")
                print(f"      Detected: {detected_at}")
                print(f"      Created: {created_at}")
                print()
        else:
            print("   No events found")
        
        # 2. Check alerts table
        print("\n🚨 2. ALERTS TABLE:")
        print("-" * 50)
        cur.execute("""
            SELECT alert_id, event_id, alert_type, severity, alert_message, 
                   status, created_at
            FROM alerts 
            ORDER BY created_at DESC 
            LIMIT 5;
        """)
        
        alerts = cur.fetchall()
        if alerts:
            for i, alert in enumerate(alerts, 1):
                alert_id, event_id, alert_type, severity, message, status, created_at = alert
                print(f"   {i}. Alert ID: {alert_id}")
                print(f"      Event ID: {event_id}")
                print(f"      Type: {alert_type}")
                print(f"      Severity: {severity}")
                print(f"      Message: {message}")
                print(f"      Status: {status}")
                print(f"      Created: {created_at}")
                print()
        else:
            print("   No alerts found")
        
        # 3. Check snapshots table
        print("\n📸 3. SNAPSHOTS TABLE:")
        print("-" * 50)
        cur.execute("""
            SELECT snapshot_id, camera_id, image_path, capture_type, 
                   captured_at, is_processed
            FROM snapshots 
            ORDER BY captured_at DESC 
            LIMIT 5;
        """)
        
        snapshots = cur.fetchall()
        if snapshots:
            for i, snapshot in enumerate(snapshots, 1):
                snapshot_id, camera_id, image_path, capture_type, captured_at, is_processed = snapshot
                print(f"   {i}. Snapshot ID: {snapshot_id}")
                print(f"      Camera: {camera_id}")
                print(f"      Image: {image_path}")
                print(f"      Type: {capture_type}")
                print(f"      Processed: {is_processed}")
                print(f"      Captured: {captured_at}")
                print()
        else:
            print("   No snapshots found")
        
        # 4. Count records by type
        print("\n📊 4. SUMMARY STATISTICS:")
        print("-" * 50)
        
        # Event counts
        cur.execute("SELECT event_type, COUNT(*) FROM event_detections GROUP BY event_type ORDER BY count DESC;")
        event_counts = cur.fetchall()
        
        print("   Event Types:")
        for event_type, count in event_counts:
            print(f"     {event_type}: {count} events")
        
        # Total counts
        cur.execute("SELECT COUNT(*) FROM event_detections;")
        total_events = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM alerts;")
        total_alerts = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM snapshots;")
        total_snapshots = cur.fetchone()[0]
        
        print(f"\n   Total Records:")
        print(f"     Events: {total_events}")
        print(f"     Alerts: {total_alerts}")
        print(f"     Snapshots: {total_snapshots}")
        
        # 5. Recent activity (last hour)
        print("\n⏰ 5. RECENT ACTIVITY (Last Hour):")
        print("-" * 50)
        cur.execute("""
            SELECT event_type, event_description, confidence_score, detected_at
            FROM event_detections 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
            ORDER BY detected_at DESC;
        """)
        
        recent_events = cur.fetchall()
        if recent_events:
            for event in recent_events:
                event_type, description, confidence, detected_at = event
                print(f"   {event_type}: {description} ({confidence:.1%}) at {detected_at}")
        else:
            print("   No recent activity")
        
        cur.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ Database check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_records()
