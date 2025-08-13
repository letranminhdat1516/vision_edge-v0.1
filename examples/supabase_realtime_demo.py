"""
Demo script to test Supabase Realtime Integration
Simulates healthcare events and publishes to Supabase
"""

import time
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.healthcare_event_publisher import healthcare_publisher

def demo_fall_detection():
    """Demo fall detection event"""
    print("🚨 Simulating FALL DETECTION...")
    
    # Simulate bounding box data
    bounding_boxes = [
        {
            "x": 150, "y": 200,
            "width": 180, "height": 250,
            "class": "person", "confidence": 0.92
        }
    ]
    
    # Simulate context data
    context = {
        "motion_level": 0.85,
        "detection_type": "direct",
        "processing_time": 0.045,
        "frame_number": 1500,
        "camera_location": "Patient Room 101",
        "demo_mode": True
    }
    
    # Publish fall detection
    event_id = healthcare_publisher.publish_fall_detection(
        confidence=0.87,
        bounding_boxes=bounding_boxes,
        context=context
    )
    
    if event_id:
        print(f"✅ Fall detection event published: {event_id}")
    else:
        print("❌ Failed to publish fall detection")
    
    return event_id

def demo_seizure_detection():
    """Demo seizure detection event"""
    print("🚨 Simulating SEIZURE DETECTION...")
    
    # Simulate bounding box data
    bounding_boxes = [
        {
            "x": 120, "y": 180,
            "width": 200, "height": 280,
            "class": "person", "confidence": 0.89
        }
    ]
    
    # Simulate context data
    context = {
        "motion_level": 0.92,
        "detection_type": "confirmation",
        "confirmation_frames": 3,
        "temporal_ready": True,
        "processing_time": 0.067,
        "frame_number": 2100,
        "camera_location": "ICU Room 205",
        "demo_mode": True
    }
    
    # Publish seizure detection
    event_id = healthcare_publisher.publish_seizure_detection(
        confidence=0.74,
        bounding_boxes=bounding_boxes,
        context=context
    )
    
    if event_id:
        print(f"✅ Seizure detection event published: {event_id}")
    else:
        print("❌ Failed to publish seizure detection")
    
    return event_id

def demo_system_status():
    """Demo system status broadcast"""
    print("📊 Broadcasting system status...")
    
    metrics = {
        "fps": 15.2,
        "cpu_usage": 45.6,
        "memory_usage": 67.8,
        "active_cameras": 3,
        "events_today": 12,
        "alerts_today": 4,
        "uptime_hours": 24.5,
        "demo_mode": True
    }
    
    healthcare_publisher.publish_system_status(
        status="online",
        metrics=metrics
    )
    
    print("✅ System status broadcasted")

def demo_recent_events():
    """Demo getting recent events"""
    print("📋 Fetching recent events...")
    
    recent_events = healthcare_publisher.get_recent_events(limit=5)
    
    if recent_events:
        print(f"✅ Found {len(recent_events)} recent events:")
        for i, event in enumerate(recent_events, 1):
            event_type = event.get('event_type', 'unknown')
            confidence = event.get('confidence_score', 0)
            created_at = event.get('created_at', 'unknown')
            print(f"  {i}. {event_type} (confidence: {confidence:.2%}) at {created_at}")
    else:
        print("ℹ️ No recent events found")
    
    return recent_events

def main():
    """Main demo function"""
    print("=" * 60)
    print("🚀 SUPABASE REALTIME HEALTHCARE DEMO")
    print("=" * 60)
    print()
    
    # Check if we have valid configuration
    try:
        from service.supabase_realtime_service import realtime_service
        
        if not realtime_service.is_connected:
            print("❌ Supabase not connected. Please check your configuration:")
            print("   1. Copy .env.example to .env")
            print("   2. Update SUPABASE_URL and keys in .env")
            print("   3. Run the database schema in Supabase")
            return
        else:
            print("✅ Supabase connected successfully")
            
    except Exception as e:
        print(f"❌ Supabase connection error: {e}")
        print("ℹ️ Running in demo mode without actual database connection")
    
    print()
    
    # Demo sequence
    demos = [
        ("Fall Detection Demo", demo_fall_detection),
        ("Seizure Detection Demo", demo_seizure_detection), 
        ("System Status Demo", demo_system_status),
        ("Recent Events Demo", demo_recent_events)
    ]
    
    for demo_name, demo_func in demos:
        print(f"📍 {demo_name}")
        print("-" * 40)
        
        try:
            result = demo_func()
            print("✅ Demo completed successfully")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        
        print()
        time.sleep(2)  # Pause between demos
    
    print("=" * 60)
    print("🎉 DEMO COMPLETED")
    print("=" * 60)
    print()
    print("💡 Next Steps:")
    print("   1. Run: python examples/healthcare_realtime_client.py")
    print("   2. In another terminal run: python src/main.py")
    print("   3. Watch real-time events flow between systems!")
    print()
    print("📚 Documentation:")
    print("   • docs/SUPABASE_REALTIME_GUIDE.md")
    print("   • docs/supabase_realtime_format.json")
    print()

if __name__ == "__main__":
    main()
