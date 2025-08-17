#!/usr/bin/env python3
"""
Test FCM Notification Service
Demo script để test Firebase Cloud Messaging integration
"""

import sys
import os
from pathlib import Path
import asyncio
import time

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from service.fcm_notification_service import fcm_service

async def test_fcm_notifications():
    """Test FCM notification service"""
    
    print("🔥 Testing FCM Notification Service")
    print("="*50)
    
    # Test tokens (thay bằng real FCM tokens từ mobile app)
    test_tokens = [
        "EXAMPLE_FCM_TOKEN_1_REPLACE_WITH_REAL_TOKEN",
        "EXAMPLE_FCM_TOKEN_2_REPLACE_WITH_REAL_TOKEN"
    ]
    
    print(f"📱 Testing with {len(test_tokens)} FCM tokens")
    print()
    
    # Test 1: Fall Detection Alert
    print("🚨 Test 1: Critical Fall Detection")
    fall_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.85,
        user_tokens=test_tokens,
        additional_data={
            "location": "Living Room",
            "camera_id": "cam_001",
            "urgency": "high"
        }
    )
    print(f"Response: {fall_response}")
    print()
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Test 2: Seizure Detection Alert  
    print("🧠 Test 2: Critical Seizure Detection")
    seizure_response = await fcm_service.send_emergency_alert(
        event_type="seizure", 
        confidence=0.75,
        user_tokens=test_tokens,
        additional_data={
            "location": "Bedroom",
            "camera_id": "cam_002", 
            "duration": "30s"
        }
    )
    print(f"Response: {seizure_response}")
    print()
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Test 3: Topic Notification
    print("📢 Test 3: Topic Notification")
    topic_response = await fcm_service.send_topic_notification(
        topic="emergency_alerts",
        title="🏥 System Health Check", 
        body="Healthcare monitoring system is online and functioning properly.",
        data={
            "type": "system_status",
            "status": "online",
            "timestamp": str(int(time.time()))
        }
    )
    print(f"Response: {topic_response}")
    print()
    
    # Test 4: Low confidence warning (should still send)
    print("⚠️ Test 4: Low Confidence Warning")
    warning_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.45,
        user_tokens=test_tokens,
        additional_data={
            "location": "Kitchen",
            "camera_id": "cam_003",
            "type": "warning"
        }
    )
    print(f"Response: {warning_response}")
    print()
    
    print("="*50)
    print("✅ FCM Testing completed!")
    print()
    print("📝 Notes:")
    print("- If FCM is not initialized, you'll see mock notifications")
    print("- Replace test tokens with real FCM tokens from mobile apps")
    print("- Check mobile apps for actual notifications")
    print("- Configure firebase-adminsdk.json for production use")

def test_integration_with_pipeline():
    """Test FCM integration với healthcare pipeline"""
    
    print("\n🏥 Testing Healthcare Pipeline Integration")
    print("="*50)
    
    # Mock tokens
    test_tokens = [
        "EXAMPLE_MOBILE_TOKEN_1",
        "EXAMPLE_MOBILE_TOKEN_2"
    ]
    
    # Simulate detection results
    fall_result = {
        'fall_detected': True,
        'fall_confidence': 0.82,
        'alert_level': 'high',
        'emergency_type': 'fall'
    }
    
    seizure_result = {
        'seizure_detected': True, 
        'seizure_confidence': 0.78,
        'alert_level': 'critical',
        'emergency_type': 'seizure'
    }
    
    # Test pipeline integration (Mock)
    from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
    
    # Create mock pipeline với FCM tokens
    pipeline = type('MockPipeline', (), {
        'user_fcm_tokens': test_tokens,
        'send_fcm_emergency_notification': AdvancedHealthcarePipeline.send_fcm_emergency_notification
    })()
    
    print("📱 Testing pipeline FCM integration...")
    
    # Test fall notification
    print("\n🚨 Simulating fall detection...")
    try:
        pipeline.send_fcm_emergency_notification(pipeline, fall_result)
        print("✅ Fall notification triggered")
    except Exception as e:
        print(f"❌ Fall notification error: {e}")
    
    # Wait
    time.sleep(3)
    
    # Test seizure notification  
    print("\n🧠 Simulating seizure detection...")
    try:
        pipeline.send_fcm_emergency_notification(pipeline, seizure_result)
        print("✅ Seizure notification triggered")
    except Exception as e:
        print(f"❌ Seizure notification error: {e}")
    
    print("\n✅ Pipeline integration test completed!")

if __name__ == "__main__":
    print("🔥 FCM Healthcare Emergency Notification System")
    print("=" * 60)
    print()
    
    # Test basic FCM service
    asyncio.run(test_fcm_notifications())
    
    # Test pipeline integration
    test_integration_with_pipeline()
    
    print("\n🎯 Next Steps:")
    print("1. Setup Firebase project and download credentials")
    print("2. Replace test tokens with real FCM tokens from mobile apps")  
    print("3. Test with actual mobile devices")
    print("4. Configure notification channels in mobile apps")
    print("5. Test emergency scenarios end-to-end")
