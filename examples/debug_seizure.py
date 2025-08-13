"""
Test seizure detection separately with detailed logging
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.healthcare_event_publisher import healthcare_publisher

def test_seizure_detection_debug():
    """Debug seizure detection"""
    print("🚨 Testing seizure detection with debug...")
    
    context = {
        'test': True,
        'motion_level': 0.92,
        'temporal_ready': True,
        'confirmation_frames': 3,
        'frame_number': 2100,
        'camera_id': '3c0b0000-0000-4000-8000-000000000002',  # Use real camera ID
        'room_id': '2d0a0000-0000-4000-8000-000000000002',    # Use real room ID
        'user_id': '361a335c-4f4d-4ed4-9e5c-ab7715d081b4'     # Use real user ID
    }
    
    print(f"Context: {context}")
    
    # Call with explicit parameters
    event_id = healthcare_publisher.publish_seizure_detection(
        confidence=0.76,
        bounding_boxes=[
            {
                'x': 120, 'y': 180,
                'width': 180, 'height': 280,
                'class': 'person', 'confidence': 0.92
            }
        ],
        context=context,
        camera_id='3c0b0000-0000-4000-8000-000000000002',
        room_id='2d0a0000-0000-4000-8000-000000000002',
        user_id='361a335c-4f4d-4ed4-9e5c-ab7715d081b4'
    )
    
    print(f"Event ID: {event_id}")

if __name__ == "__main__":
    test_seizure_detection_debug()
