"""
Test Healthcare Event Publisher with PostgreSQL
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.healthcare_event_publisher import healthcare_publisher

def test_fall_detection():
    """Test fall detection event publishing"""
    print("🚨 Testing fall detection event...")
    
    event_id = healthcare_publisher.publish_fall_detection(
        confidence=0.89,
        bounding_boxes=[
            {
                'x': 100, 'y': 150, 
                'width': 200, 'height': 300,
                'class': 'person', 'confidence': 0.95
            }
        ],
        context={
            'test': True, 
            'motion_level': 0.85,
            'frame_number': 1500,
            'camera_location': 'Test Room',
            'camera_id': '3c0b0000-0000-4000-8000-000000000001',  # Use real camera ID
            'room_id': '2d0a0000-0000-4000-8000-000000000001',    # Use real room ID
            'user_id': '34e92ef3-1300-40d0-a0e0-72989cf30121'     # Use real user ID
        }
    )
    
    if event_id:
        print(f"✅ Fall event published with ID: {event_id}")
        return event_id
    else:
        print("❌ Failed to publish fall event")
        return None

def test_seizure_detection():
    """Test seizure detection event publishing"""
    print("🚨 Testing seizure detection event...")
    
    event_id = healthcare_publisher.publish_seizure_detection(
        confidence=0.76,
        bounding_boxes=[
            {
                'x': 120, 'y': 180,
                'width': 180, 'height': 280,
                'class': 'person', 'confidence': 0.92
            }
        ],
        context={
            'test': True,
            'motion_level': 0.92,
            'temporal_ready': True,
            'confirmation_frames': 3,
            'frame_number': 2100,
            'camera_id': '3c0b0000-0000-4000-8000-000000000002',  # Use real camera ID
            'room_id': '2d0a0000-0000-4000-8000-000000000002',    # Use real room ID
            'user_id': '361a335c-4f4d-4ed4-9e5c-ab7715d081b4'     # Use real user ID
        }
    )
    
    if event_id:
        print(f"✅ Seizure event published with ID: {event_id}")
        return event_id
    else:
        print("❌ Failed to publish seizure event")
        return None

def test_recent_events():
    """Test getting recent events"""
    print("📋 Getting recent events...")
    
    events = healthcare_publisher.get_recent_events(limit=5)
    print(f"Found {len(events)} recent events:")
    
    for i, event in enumerate(events, 1):
        event_type = event.get('event_type', 'unknown')
        confidence = event.get('confidence_score', 0)
        created_at = str(event.get('created_at', 'unknown'))[:19]  # Truncate timestamp
        
        print(f"  {i}. {event_type} (confidence: {confidence:.1%}) at {created_at}")
    
    return events

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 HEALTHCARE EVENT PUBLISHER TEST")
    print("=" * 60)
    print()
    
    # Test fall detection
    fall_event_id = test_fall_detection()
    print()
    
    # Test seizure detection
    seizure_event_id = test_seizure_detection()
    print()
    
    # Test getting recent events
    recent_events = test_recent_events()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if fall_event_id and seizure_event_id:
        print("✅ All tests passed!")
        print(f"✅ Fall event ID: {fall_event_id}")
        print(f"✅ Seizure event ID: {seizure_event_id}")
        print(f"✅ Found {len(recent_events)} recent events")
    else:
        print("❌ Some tests failed")
        if not fall_event_id:
            print("❌ Fall detection test failed")
        if not seizure_event_id:
            print("❌ Seizure detection test failed")
    
    print()
    print("💡 Next Steps:")
    print("   1. Run: python healthcare_realtime_client.py")
    print("   2. In another terminal run: python src/main.py")
    print("   3. Watch real-time events flow!")

if __name__ == "__main__":
    main()
