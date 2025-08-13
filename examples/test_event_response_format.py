"""
Test new event response format with imageUrl, status, action, time
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.healthcare_event_publisher import healthcare_publisher
import json

def test_fall_detection_response():
    """Test fall detection response format"""
    print("🚨 Testing Fall Detection Response Format...")
    
    # Test fall detection with high confidence (danger)
    response = healthcare_publisher.publish_fall_detection(
        confidence=0.85,  # High confidence -> danger
        bounding_boxes=[
            {
                'x': 120, 'y': 180,
                'width': 180, 'height': 280,
                'class': 'person', 'confidence': 0.85
            }
        ],
        context={'test': True, 'motion_level': 0.92},
        camera_id='3c0b0000-0000-4000-8000-000000000001',
        room_id='2d0a0000-0000-4000-8000-000000000001',
        user_id='34e92ef3-1300-40d0-a0e0-72989cf30121'
    )
    
    print("📋 FALL DETECTION RESPONSE (Danger Level):")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print()
    
    # Test fall detection with medium confidence (warning)
    response2 = healthcare_publisher.publish_fall_detection(
        confidence=0.65,  # Medium confidence -> warning
        bounding_boxes=[
            {
                'x': 100, 'y': 150,
                'width': 160, 'height': 250,
                'class': 'person', 'confidence': 0.65
            }
        ]
    )
    
    print("📋 FALL DETECTION RESPONSE (Warning Level):")
    print(json.dumps(response2, indent=2, ensure_ascii=False))
    print()

def test_seizure_detection_response():
    """Test seizure detection response format"""
    print("🧠 Testing Seizure Detection Response Format...")
    
    # Test seizure detection with high confidence (danger)
    response = healthcare_publisher.publish_seizure_detection(
        confidence=0.76,  # High confidence -> danger
        bounding_boxes=[
            {
                'x': 120, 'y': 180,
                'width': 180, 'height': 280,
                'class': 'person', 'confidence': 0.92
            }
        ],
        context={'test': True, 'temporal_ready': True},
        camera_id='3c0b0000-0000-4000-8000-000000000002',
        room_id='2d0a0000-0000-4000-8000-000000000002',
        user_id='361a335c-4f4d-4ed4-9e5c-ab7715d081b4'
    )
    
    print("📋 SEIZURE DETECTION RESPONSE (Danger Level):")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print()
    
    # Test seizure detection with medium confidence (warning)
    response2 = healthcare_publisher.publish_seizure_detection(
        confidence=0.55,  # Medium confidence -> warning
        bounding_boxes=[
            {
                'x': 150, 'y': 200,
                'width': 140, 'height': 220,
                'class': 'person', 'confidence': 0.78
            }
        ]
    )
    
    print("📋 SEIZURE DETECTION RESPONSE (Warning Level):")
    print(json.dumps(response2, indent=2, ensure_ascii=False))
    print()

def test_normal_status():
    """Test normal status response"""
    print("✅ Testing Normal Status Response...")
    
    # Test with low confidence (normal)
    response = healthcare_publisher.publish_fall_detection(
        confidence=0.35,  # Low confidence -> normal
        bounding_boxes=[]
    )
    
    print("📋 NORMAL STATUS RESPONSE:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print()

def main():
    """Main test function"""
    print("🔬 TESTING NEW EVENT RESPONSE FORMAT")
    print("=" * 60)
    print()
    
    # Test all response formats
    test_fall_detection_response()
    test_seizure_detection_response()
    test_normal_status()
    
    print("=" * 60)
    print("✅ Response Format Testing Completed!")
    print()
    print("📋 RESPONSE FORMAT SPECIFICATION:")
    print("""
    {
        "imageUrl": "https://healthcare-system.com/snapshots/{event_id}.jpg",
        "status": "normal|warning|danger",
        "action": "Mô tả hành động được phát hiện",
        "time": "2025-08-14T10:30:00.123456"
    }
    
    STATUS LEVELS:
    - normal: confidence < 50% (fall) / < 50% (seizure)
    - warning: confidence 60-80% (fall) / 50-70% (seizure)  
    - danger: confidence >= 80% (fall) / >= 70% (seizure)
    """)

if __name__ == "__main__":
    main()
