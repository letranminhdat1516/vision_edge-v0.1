#!/usr/bin/env python3
"""
Quick FCM Demo - Test Firebase notifications without credentials
Demonstrates FCM integration with mock notifications
"""

import sys
import os
from pathlib import Path
import asyncio

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import FCM service
from service.fcm_notification_service import fcm_service

async def demo_fcm_notifications():
    """Demo FCM notifications (will run in mock mode without credentials)"""
    
    print("🔥 FCM Healthcare Emergency Demo")
    print("="*60)
    print("📱 This demo runs in MOCK mode (no real notifications)")
    print("   Set up Firebase credentials for real notifications")
    print("="*60)
    print()
    
    # Demo FCM tokens (examples)
    demo_tokens = [
        "demo_token_user1_phone",
        "demo_token_user2_tablet", 
        "demo_token_caregiver_device"
    ]
    
    print(f"📱 Demo with {len(demo_tokens)} device tokens")
    print()
    
    # Demo 1: Critical Fall Detection
    print("🚨 DEMO 1: Critical Fall Alert")
    print("-" * 40)
    fall_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.89,
        user_tokens=demo_tokens,
        additional_data={
            "location": "Living Room Camera",
            "urgency": "critical",
            "response_time": "immediate"
        }
    )
    print(f"📤 Result: {fall_response}")
    print()
    
    await asyncio.sleep(1)
    
    # Demo 2: Critical Seizure Detection  
    print("🧠 DEMO 2: Critical Seizure Alert")
    print("-" * 40)
    seizure_response = await fcm_service.send_emergency_alert(
        event_type="seizure",
        confidence=0.76, 
        user_tokens=demo_tokens,
        additional_data={
            "location": "Bedroom Camera",
            "duration_detected": "45 seconds",
            "urgency": "critical"
        }
    )
    print(f"📤 Result: {seizure_response}")
    print()
    
    await asyncio.sleep(1)
    
    # Demo 3: Warning Level
    print("⚠️ DEMO 3: Warning Level Alert")
    print("-" * 40)
    warning_response = await fcm_service.send_emergency_alert(
        event_type="fall",
        confidence=0.55,
        user_tokens=demo_tokens,
        additional_data={
            "location": "Kitchen Camera",
            "urgency": "medium", 
            "verification_needed": True
        }
    )
    print(f"📤 Result: {warning_response}")
    print()
    
    # Demo 4: Topic Notification
    print("📢 DEMO 4: Topic Broadcast")
    print("-" * 40)
    topic_response = await fcm_service.send_topic_notification(
        topic="emergency_alerts",
        title="🏥 Healthcare System Online",
        body="All monitoring systems are operational. Emergency detection active.",
        data={
            "type": "system_status",
            "status": "online",
            "cameras_active": "3",
            "last_check": "2025-08-18 10:30:00"
        }
    )
    print(f"📤 Result: {topic_response}")
    print()
    
    print("✅ FCM Demo completed!")
    print()
    
def demo_healthcare_pipeline_integration():
    """Demo tích hợp với healthcare pipeline"""
    
    print("🏥 HEALTHCARE PIPELINE INTEGRATION DEMO")
    print("="*60)
    
    # Mock FCM tokens
    fcm_tokens = ["mock_token_1", "mock_token_2"]
    
    # Simulate detection results
    detection_scenarios = [
        {
            'name': 'High Confidence Fall',
            'result': {
                'fall_detected': True,
                'fall_confidence': 0.85,
                'alert_level': 'high', 
                'emergency_type': 'fall'
            }
        },
        {
            'name': 'Critical Seizure',
            'result': {
                'seizure_detected': True,
                'seizure_confidence': 0.78,
                'alert_level': 'critical',
                'emergency_type': 'seizure'  
            }
        },
        {
            'name': 'Fall Warning',
            'result': {
                'fall_detected': False,
                'fall_confidence': 0.45,
                'alert_level': 'warning',
                'emergency_type': 'fall_warning'
            }
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(detection_scenarios, 1):
        print(f"📊 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        # Mock pipeline behavior
        result = scenario['result']
        alert_level = result.get('alert_level', 'normal')
        
        if alert_level in ['critical', 'high']:
            print(f"🚨 EMERGENCY DETECTED: {result['emergency_type']}")
            print(f"📱 FCM notification would be sent to {len(fcm_tokens)} devices")
            
            if 'fall' in result['emergency_type']:
                confidence = result.get('fall_confidence', 0.0)
                print(f"   Event: Fall Detection (confidence: {confidence:.2f})")
            elif 'seizure' in result['emergency_type']:
                confidence = result.get('seizure_confidence', 0.0) 
                print(f"   Event: Seizure Detection (confidence: {confidence:.2f})")
                
            print(f"   Alert Level: {alert_level.upper()}")
            print(f"   Notification: Emergency alert sent")
        else:
            print(f"ℹ️ Normal detection - no emergency notification")
            
        print()

if __name__ == "__main__":
    print("🔥 VISION EDGE FCM INTEGRATION DEMO")
    print("Healthcare Emergency Notification System")
    print("="*70)
    print()
    
    # Run FCM demo
    asyncio.run(demo_fcm_notifications())
    
    # Run pipeline integration demo
    demo_healthcare_pipeline_integration()
    
    print("🎯 SETUP INSTRUCTIONS:")
    print("="*70)
    print("1. 📁 Create Firebase project at https://console.firebase.google.com/")
    print("2. 🔑 Generate service account key (firebase-adminsdk.json)")
    print("3. 📂 Place credentials in src/config/firebase-adminsdk.json")
    print("4. 📱 Get FCM tokens from mobile apps")
    print("5. 🔄 Replace demo tokens with real FCM tokens")
    print("6. 🚀 Run with real Firebase credentials for actual notifications")
    print()
    print("💡 Next steps:")
    print("   - Test with real mobile devices")
    print("   - Configure notification channels")
    print("   - Set up notification sounds and vibrations")
    print("   - Implement notification acknowledgment system")
