#!/usr/bin/env python3
"""
Real-time FCM Healthcare Test
Test FCM notifications với real configuration từ .env
"""

import sys
import os
from pathlib import Path
import asyncio
import time
from datetime import datetime

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from service.fcm_notification_service import fcm_service

async def test_realtime_fcm():
    """Test FCM với real configuration"""
    
    print("🔥 REAL-TIME FCM HEALTHCARE TEST")
    print("="*60)
    print(f"🕐 Test started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Kiểm tra configuration
    print("📋 FCM Configuration Status:")
    print(f"   Project ID: {fcm_service.project_id}")
    print(f"   Credentials Path: {fcm_service.default_credentials_path}")
    print(f"   Notifications Enabled: {fcm_service.enable_notifications}")
    print(f"   Device Tokens: {len(fcm_service.device_tokens)}")
    print(f"   Caregiver Tokens: {len(fcm_service.caregiver_tokens)}")
    print(f"   Emergency Tokens: {len(fcm_service.emergency_tokens)}")
    print(f"   Total Tokens: {len(fcm_service.all_tokens)}")
    print(f"   FCM Initialized: {fcm_service.initialized}")
    print()
    
    if not fcm_service.all_tokens:
        print("⚠️ WARNING: No real FCM tokens found!")
        print("   Update .env with real FCM tokens from mobile apps")
        print("   Example:")
        print("   FCM_DEVICE_TOKENS=dxxxxx:APA91bGxxx,exxxxx:APA91bHxxx")
        print()
    
    # Test 1: Critical Fall Alert
    print("🚨 TEST 1: Critical Fall Detection Alert")
    print("-" * 50)
    start_time = time.time()
    
    fall_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.89,
        additional_data={
            "location": "Living Room - Camera 01",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "urgency": "critical",
            "patient_id": "patient_001",
            "room_id": "room_living"
        }
    )
    
    end_time = time.time()
    print(f"📤 Response: {fall_response}")
    print(f"⏱️ Processing time: {(end_time - start_time)*1000:.1f}ms")
    print()
    
    await asyncio.sleep(2)
    
    # Test 2: Critical Seizure Alert
    print("🧠 TEST 2: Critical Seizure Detection Alert") 
    print("-" * 50)
    start_time = time.time()
    
    seizure_response = await fcm_service.send_emergency_alert(
        event_type="seizure",
        confidence=0.82,
        additional_data={
            "location": "Bedroom - Camera 02",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": "45 seconds",
            "patient_id": "patient_001",
            "room_id": "room_bedroom"
        }
    )
    
    end_time = time.time()
    print(f"📤 Response: {seizure_response}")
    print(f"⏱️ Processing time: {(end_time - start_time)*1000:.1f}ms")
    print()
    
    await asyncio.sleep(2)
    
    # Test 3: Warning Level Alert
    print("⚠️ TEST 3: Warning Level Alert")
    print("-" * 50)
    start_time = time.time()
    
    warning_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.55,
        additional_data={
            "location": "Kitchen - Camera 03",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "verification_needed": "true",
            "alert_level": "warning"
        }
    )
    
    end_time = time.time()
    print(f"📤 Response: {warning_response}")
    print(f"⏱️ Processing time: {(end_time - start_time)*1000:.1f}ms")
    print()
    
    await asyncio.sleep(2)
    
    # Test 4: Topic Notification to All Devices
    print("📢 TEST 4: Topic Notification (All Devices)")
    print("-" * 50)
    start_time = time.time()
    
    topic_response = await fcm_service.send_topic_notification(
        topic=fcm_service.default_topic,
        title="🏥 Healthcare System Status",
        body=f"All monitoring systems operational. Last check: {datetime.now().strftime('%H:%M:%S')}",
        data={
            "type": "system_status",
            "status": "online",
            "cameras_active": "3",
            "last_check": datetime.now().isoformat(),
            "system_version": "v1.0"
        }
    )
    
    end_time = time.time()
    print(f"📤 Response: {topic_response}")
    print(f"⏱️ Processing time: {(end_time - start_time)*1000:.1f}ms")
    print()
    
    # Test 5: Multiple rapid alerts (stress test)
    print("⚡ TEST 5: Rapid Multiple Alerts (Stress Test)")
    print("-" * 50)
    
    for i in range(3):
        start_time = time.time()
        
        rapid_response = await fcm_service.send_emergency_alert(
            event_type="seizure" if i % 2 else "fall",
            confidence=0.7 + (i * 0.1),
            additional_data={
                "test_sequence": str(i + 1),
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "rapid_test": "true"
            }
        )
        
        end_time = time.time()
        success = rapid_response.get('success', False)
        status = "✅" if success else "❌"
        print(f"   {status} Alert {i+1}: {(end_time - start_time)*1000:.1f}ms")
        
        await asyncio.sleep(0.5)
    
    print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"🕐 Test completed at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📱 Total FCM tokens configured: {len(fcm_service.all_tokens)}")
    print(f"🔥 Firebase project: {fcm_service.project_id}")
    print(f"✅ FCM service initialized: {fcm_service.initialized}")
    print(f"🔔 Notifications enabled: {fcm_service.enable_notifications}")
    
    if fcm_service.all_tokens:
        print("\n📱 MOBILE INTEGRATION STATUS:")
        print("   ✅ Real FCM tokens detected")
        print("   ✅ Ready for mobile notifications")
        print("   🎯 Check mobile devices for received notifications")
    else:
        print("\n📱 MOBILE INTEGRATION NEEDED:")
        print("   ⚠️ No real FCM tokens found in .env")
        print("   📋 Steps to complete setup:")
        print("     1. Integrate Firebase SDK in mobile app")
        print("     2. Get FCM registration tokens")
        print("     3. Add tokens to .env file")
        print("     4. Re-run this test")

def test_healthcare_pipeline_integration():
    """Test integration với healthcare pipeline"""
    
    print("\n🏥 HEALTHCARE PIPELINE INTEGRATION TEST")
    print("="*60)
    
    try:
        from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
        
        # Mock services for testing
        mock_camera = type('MockCamera', (), {'camera_id': 'test_cam_001'})()
        mock_video_processor = type('MockVideoProcessor', (), {})()
        mock_fall_detector = type('MockFallDetector', (), {})()
        mock_seizure_detector = type('MockSeizureDetector', (), {})()
        mock_seizure_predictor = type('MockSeizurePredictor', (), {})()
        
        # Create pipeline without FCM tokens (will use .env)
        pipeline = AdvancedHealthcarePipeline(
            mock_camera, mock_video_processor, mock_fall_detector,
            mock_seizure_detector, mock_seizure_predictor, 
            "test_alerts_folder"
        )
        
        print("✅ Healthcare pipeline created successfully")
        print("📱 FCM integration ready")
        
        # Test detection scenarios
        test_scenarios = [
            {
                'name': 'High Confidence Fall',
                'result': {
                    'fall_detected': True,
                    'fall_confidence': 0.87,
                    'alert_level': 'high',
                    'emergency_type': 'fall'
                }
            },
            {
                'name': 'Critical Seizure',
                'result': {
                    'seizure_detected': True,
                    'seizure_confidence': 0.79,
                    'alert_level': 'critical',
                    'emergency_type': 'seizure'
                }
            }
        ]
        
        print("\n🧪 Testing detection scenarios:")
        for scenario in test_scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            print("   Triggering FCM notification...")
            
            try:
                pipeline.send_fcm_emergency_notification(scenario['result'])
                print("   ✅ FCM notification triggered successfully")
            except Exception as e:
                print(f"   ❌ FCM notification failed: {e}")
        
        print("\n✅ Healthcare pipeline integration test completed")
        
    except Exception as e:
        print(f"❌ Healthcare pipeline integration test failed: {e}")

if __name__ == "__main__":
    print("🔥 VISION EDGE - REAL-TIME FCM TEST SUITE")
    print("Healthcare Emergency Notification System")
    print("="*70)
    print()
    
    # Run real-time FCM tests
    asyncio.run(test_realtime_fcm())
    
    # Test healthcare pipeline integration
    test_healthcare_pipeline_integration()
    
    print("\n🎯 NEXT STEPS:")
    print("="*70)
    print("1. 📱 Verify notifications received on mobile devices")
    print("2. 🔊 Test notification sounds and vibrations")
    print("3. 📊 Monitor FCM delivery statistics")
    print("4. 🧪 Test with actual fall/seizure detection")
    print("5. 📈 Set up notification analytics and monitoring")
    print()
    print("💡 For production deployment:")
    print("   - Monitor FCM token validity and refresh expired tokens")
    print("   - Implement notification acknowledgment system")
    print("   - Set up delivery receipt tracking")
    print("   - Configure different notification channels for different alert types")
