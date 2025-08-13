"""
Test Complete Mobile Realtime Healthcare Notification System
Tests the new JSON format with mobile notifications
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.healthcare_event_publisher_v2 import healthcare_publisher
from service.mobile_realtime_notification_service import start_mobile_notifications, stop_mobile_notifications
import json
import time

def test_mobile_realtime_system():
    """Test complete mobile realtime notification system"""
    print("📱 TESTING MOBILE REALTIME HEALTHCARE NOTIFICATION SYSTEM")
    print("=" * 70)
    
    # Start mobile notification service
    start_mobile_notifications()
    print()
    
    # Test cases with different confidence levels and statuses
    test_cases = [
        {
            "name": "Normal Fall Detection (Low Confidence)",
            "type": "fall",
            "confidence": 0.35,
            "expected_status": "normal"
        },
        {
            "name": "Warning Fall Detection (Medium Confidence)",
            "type": "fall", 
            "confidence": 0.65,
            "expected_status": "warning"
        },
        {
            "name": "Danger Fall Detection (High Confidence)",
            "type": "fall",
            "confidence": 0.85,
            "expected_status": "danger"
        },
        {
            "name": "Normal Seizure Detection (Low Confidence)",
            "type": "seizure",
            "confidence": 0.45,
            "expected_status": "normal"
        },
        {
            "name": "Warning Seizure Detection (Medium Confidence)",
            "type": "seizure",
            "confidence": 0.60,
            "expected_status": "warning"
        },
        {
            "name": "Danger Seizure Detection (High Confidence)",
            "type": "seizure",
            "confidence": 0.75,
            "expected_status": "danger"
        }
    ]
    
    print("🔬 RUNNING TEST CASES:")
    print("-" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"   Type: {test_case['type']}")
        print(f"   Confidence: {test_case['confidence']:.0%}")
        print(f"   Expected Status: {test_case['expected_status']}")
        
        # Create sample bounding boxes
        bounding_boxes = [
            {
                'x': 120, 'y': 180,
                'width': 180, 'height': 280,
                'class': 'person', 
                'confidence': test_case['confidence']
            }
        ]
        
        # Context data
        context = {
            'test': True,
            'test_case': test_case['name'],
            'camera_location': 'Room 101',
            'motion_level': 0.8
        }
        
        # Call appropriate detection method
        if test_case['type'] == 'fall':
            response = healthcare_publisher.publish_fall_detection(
                confidence=test_case['confidence'],
                bounding_boxes=bounding_boxes,
                context=context
            )
        else:  # seizure
            response = healthcare_publisher.publish_seizure_detection(
                confidence=test_case['confidence'],
                bounding_boxes=bounding_boxes,
                context=context
            )
        
        # Validate response format
        print(f"\n   📄 Response JSON:")
        print(f"   {json.dumps(response, indent=6, ensure_ascii=False)}")
        
        # Validate status
        actual_status = response.get('status')
        if actual_status == test_case['expected_status']:
            print(f"   ✅ Status validation: PASS ({actual_status})")
        else:
            print(f"   ❌ Status validation: FAIL (expected {test_case['expected_status']}, got {actual_status})")
        
        # Validate required fields
        required_fields = ['imageUrl', 'status', 'action', 'time']
        missing_fields = [field for field in required_fields if field not in response]
        
        if not missing_fields:
            print(f"   ✅ Format validation: PASS (all required fields present)")
        else:
            print(f"   ❌ Format validation: FAIL (missing fields: {missing_fields})")
        
        # Wait before next test
        time.sleep(2)
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print("=" * 70)
    
    # Summary of status logic
    print("""
    🎯 STATUS LOGIC IMPLEMENTED:
    
    FALL DETECTION:
    • normal:  confidence < 60%   → "Không có gì bất thường"
    • warning: confidence 60-80%  → "Phát hiện té (XX% confidence) - Cần theo dõi"  
    • danger:  confidence >= 80%  → "⚠️ BÁO ĐỘNG NGUY HIỂM: Phát hiện té - Yêu cầu hỗ trợ gấp!"
    
    SEIZURE DETECTION:
    • normal:  confidence < 50%   → "Không có gì bất thường"
    • warning: confidence 50-70%  → "Phát hiện co giật (XX% confidence) - Cần theo dõi"
    • danger:  confidence >= 70%  → "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!"
    
    📱 MOBILE NOTIFICATION FORMAT:
    {
        "imageUrl": "https://healthcare-system.com/snapshots/{event_id}.jpg",
        "status": "normal|warning|danger",
        "action": "Action description based on status and type",
        "time": "Timestamp from snapshot creation (ISO format)"
    }
    """)
    
    # Stop mobile notification service
    stop_mobile_notifications()
    print("✅ Mobile Realtime Healthcare Notification System Test Completed!")

if __name__ == "__main__":
    test_mobile_realtime_system()
