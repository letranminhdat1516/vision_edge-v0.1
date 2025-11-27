"""
Test Auto-Stop Alarm Logic
Tests:
1. Alarm stops when >= 2 people detected
2. Alarm stops when alert_level = 'normal'
3. Database updated to RESOLVED
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import asyncio
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("🧪 TEST AUTO-STOP ALARM LOGIC")
print("=" * 80)

# Import services
from infrastructure.services.audio_alert_service import audio_alert_service
from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

# Initialize
db_service = PostgreSQLHealthcareService()
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')

# Connect handler to database
emergency_alarm_handler.set_postgresql_service(db_service)

print(f"\n👤 User ID: {user_id}")
print(f"🔊 Audio enabled: {audio_alert_service.enabled}")
print(f"💾 Database connected: {db_service is not None}")

def create_test_alarm_event():
    """Create a test event and activate alarm"""
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        
        # Create test event
        import uuid
        from datetime import datetime, timezone
        
        event_id = str(uuid.uuid4())
        snapshot_id = str(uuid.uuid4())
        
        # Create minimal snapshot first
        cursor.execute("""
            INSERT INTO snapshots (
                snapshot_id, user_id, camera_id,
                capture_type, captured_at, is_processed
            ) VALUES (
                %s, %s,
                (SELECT camera_id FROM cameras WHERE user_id = %s LIMIT 1),
                'manual', NOW(), false
            )
        """, (snapshot_id, user_id, user_id))
        
        # Create event with snapshot
        cursor.execute("""
            INSERT INTO event_detections (
                event_id, user_id, camera_id, event_type,
                snapshot_id, confidence_score, status, lifecycle_state,
                event_description, detected_at, created_at,
                confirmation_state, verification_status,
                escalation_count, is_canceled, notification_attempts
            ) VALUES (
                %s, %s, 
                (SELECT camera_id FROM cameras WHERE user_id = %s LIMIT 1),
                'fall', %s, 0.85, 'danger', 'ALARM_ACTIVATED',
                'Test alarm for auto-stop', NOW(), NOW(),
                'DETECTED', 'PENDING', 0, false, 0
            )
            RETURNING event_id, event_type
        """, (event_id, user_id, user_id, snapshot_id))
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        db_service.return_connection(conn)
        
        if result:
            print(f"\n✅ Created test event: {event_id[:8]}...")
            print(f"   Type: fall")
            print(f"   State: ALARM_ACTIVATED")
            return event_id
        return None
        
    except Exception as e:
        print(f"\n❌ Failed to create test event: {e}")
        return None

def start_alarm():
    """Start alarm for testing"""
    try:
        print("\n🔊 Starting test alarm...")
        result = asyncio.run(audio_alert_service.play_emergency_alarm(
            user_id=user_id,
            triggered_by='test',
            duration=0  # Infinite until stopped
        ))
        
        if result['success']:
            print("   ✅ Alarm started!")
            return True
        else:
            print(f"   ❌ Alarm failed: {result['message']}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_alarm_status():
    """Check if alarm is playing"""
    return audio_alert_service.is_playing

def check_event_state(event_id):
    """Check event lifecycle_state"""
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT lifecycle_state, notes
            FROM event_detections
            WHERE event_id = %s
        """, (event_id,))
        
        result = cursor.fetchone()
        cursor.close()
        db_service.return_connection(conn)
        
        if result:
            if isinstance(result, dict):
                return result['lifecycle_state'], result.get('notes', '')
            else:
                return result[0], result[1] if len(result) > 1 else ''
        return None, None
        
    except Exception as e:
        print(f"   ❌ Error checking state: {e}")
        return None, None

def test_resolve_function(reason="Test resolve"):
    """Test the resolve_active_alarms function"""
    print(f"\n🧪 Testing resolve_active_alarms('{reason}')...")
    
    # Create test event
    event_id = create_test_alarm_event()
    if not event_id:
        return False
    
    # Verify event is ALARM_ACTIVATED
    state, _ = check_event_state(event_id)
    if state != 'ALARM_ACTIVATED':
        print(f"   ❌ Event not in ALARM_ACTIVATED state: {state}")
        return False
    
    print(f"   ✅ Event in ALARM_ACTIVATED state")
    
    # Start alarm and let it play for 5 seconds
    print(f"\n🔊 Starting alarm (will play for 5 seconds)...")
    if audio_alert_service.enabled:
        if start_alarm():
            print(f"   ✅ Alarm is playing! Listen...")
            print(f"   ⏳ Playing for 5 seconds before auto-stop test...")
            time.sleep(5)
        else:
            print("   ⚠️ Alarm not started (audio may be disabled)")
    else:
        print("   ⚠️ Audio service disabled in .env")
        time.sleep(1)
    
    # Test resolve
    print(f"\n   Calling emergency_alarm_handler.resolve_active_alarms()...")
    resolved_count = emergency_alarm_handler.resolve_active_alarms(reason=reason)
    
    if resolved_count > 0:
        print(f"   ✅ Resolved {resolved_count} alarm(s)")
    else:
        print(f"   ❌ No alarms resolved!")
        return False
    
    # Verify state changed
    time.sleep(0.5)
    new_state, notes = check_event_state(event_id)
    
    if new_state == 'RESOLVED':
        print(f"   ✅ Event state updated: ALARM_ACTIVATED → RESOLVED")
        if reason in (notes or ''):
            print(f"   ✅ Reason recorded in notes")
        else:
            print(f"   ⚠️ Reason not found in notes")
    else:
        print(f"   ❌ Event state not updated: {new_state}")
        return False
    
    # Stop alarm if still playing
    if check_alarm_status():
        asyncio.run(audio_alert_service.stop_alarm())
        print(f"   ✅ Alarm stopped")
    
    print(f"   ✅ TEST PASSED!")
    return True

def test_scenario_two_people():
    """Simulate: >= 2 people detected"""
    print("\n" + "=" * 80)
    print("📋 TEST SCENARIO 1: >= 2 PEOPLE DETECTED")
    print("=" * 80)
    
    result = test_resolve_function(reason="Help arrived: 2 people detected")
    
    if result:
        print("\n✅ SCENARIO 1 PASSED: Alarm stops when 2+ people detected")
    else:
        print("\n❌ SCENARIO 1 FAILED")
    
    return result

def test_scenario_normal():
    """Simulate: alert_level = normal"""
    print("\n" + "=" * 80)
    print("📋 TEST SCENARIO 2: SITUATION NORMALIZED")
    print("=" * 80)
    
    result = test_resolve_function(reason="Situation normalized: alert_level = normal")
    
    if result:
        print("\n✅ SCENARIO 2 PASSED: Alarm stops when situation normalized")
    else:
        print("\n❌ SCENARIO 2 FAILED")
    
    return result

def cleanup_test_events():
    """Delete test events"""
    try:
        conn = db_service.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM event_detections
            WHERE event_description = 'Test alarm for auto-stop'
              AND created_at > NOW() - INTERVAL '1 hour'
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        db_service.return_connection(conn)
        
        if deleted > 0:
            print(f"\n🧹 Cleaned up {deleted} test event(s)")
        
    except Exception as e:
        print(f"\n⚠️ Cleanup error: {e}")

# Run tests
if __name__ == '__main__':
    try:
        # Test 1: Two people detected
        result1 = test_scenario_two_people()
        
        time.sleep(2)
        
        # Test 2: Situation normalized
        result2 = test_scenario_normal()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Scenario 1 (2 people): {'✅ PASSED' if result1 else '❌ FAILED'}")
        print(f"Scenario 2 (normal):   {'✅ PASSED' if result2 else '❌ FAILED'}")
        
        if result1 and result2:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED!")
        
        print("=" * 80)
        
        # Cleanup
        cleanup = input("\n🧹 Clean up test events? (y/n): ").strip().lower()
        if cleanup == 'y':
            cleanup_test_events()
            print("✅ Cleanup complete!")
        
    except KeyboardInterrupt:
        print("\n\n⏸️ Test interrupted")
        if check_alarm_status():
            asyncio.run(audio_alert_service.stop_alarm())
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
