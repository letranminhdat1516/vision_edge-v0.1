"""
Simple Supabase Insert Test - Test realtime notifications
Chạy script này để test insert vào Supabase và xem realtime notification trên HTML demo
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_supabase_insert():
    """Test insert event vào Supabase để trigger realtime notification"""
    
    print("🧪 Testing Supabase Realtime Insert...")
    print("=" * 50)
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials!")
        print("💡 Tạo file .env với:")
        print("   SUPABASE_URL=https://your-project.supabase.co")
        print("   SUPABASE_SERVICE_KEY=your-service-key")
        return
    
    try:
        from supabase import create_client
        
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_key)
        print(f"✅ Connected to Supabase: {supabase_url}")
        
        # Test data - fall detection
        fall_event = {
            'event_id': f'test_fall_{int(datetime.now().timestamp())}',
            'event_type': 'fall',
            'confidence_score': 0.87,
            'detected_at': datetime.now().isoformat(),
            'camera_id': 'test_camera',
            'location': 'Test Room',
            'description': 'Test fall detection for realtime notification',
            'bounding_box': {'x': 100, 'y': 150, 'width': 200, 'height': 300},
            'snapshot_path': '/snapshots/test_fall.jpg',
            'metadata': {'test': True, 'method': 'manual_insert'}
        }
        
        print(f"📤 Inserting fall event: {fall_event['event_id']}")
        
        # Insert to Supabase
        result = supabase.table('event_detections').insert(fall_event).execute()
        
        if result.data:
            print("✅ Fall event inserted successfully!")
            print(f"🔔 Event ID: {result.data[0]['event_id']}")
            print("📱 Check your HTML demo for realtime notification!")
            
            # Test data - seizure detection  
            print("\n" + "=" * 50)
            
            seizure_event = {
                'event_id': f'test_seizure_{int(datetime.now().timestamp())}',
                'event_type': 'abnormal_behavior',
                'confidence_score': 0.72,
                'detected_at': datetime.now().isoformat(),
                'camera_id': 'test_camera',
                'location': 'Test Room', 
                'description': 'Test seizure detection for realtime notification',
                'bounding_box': {'x': 150, 'y': 200, 'width': 180, 'height': 280},
                'snapshot_path': '/snapshots/test_seizure.jpg',
                'metadata': {'test': True, 'method': 'manual_insert'}
            }
            
            print(f"📤 Inserting seizure event: {seizure_event['event_id']}")
            
            result2 = supabase.table('event_detections').insert(seizure_event).execute()
            
            if result2.data:
                print("✅ Seizure event inserted successfully!")
                print(f"🔔 Event ID: {result2.data[0]['event_id']}")
                print("\n🎉 Both test events inserted!")
                print("📱 Open mobile_realtime_demo.html to see realtime notifications")
                print("🌐 HTML Demo: examples/mobile_realtime_demo.html")
                
            else:
                print("❌ Failed to insert seizure event")
        else:
            print("❌ Failed to insert fall event")
            
    except ImportError:
        print("❌ Supabase library not installed!")
        print("💡 Run: pip install supabase")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check .env file có đúng SUPABASE_URL và keys")
        print("   2. Check Supabase project có table 'event_detections'")
        print("   3. Check RLS policy allows INSERT")

def show_mobile_payload_examples():
    """Show examples of mobile notification payloads"""
    print("\n📱 Mobile Notification Payload Examples:")
    print("=" * 50)
    
    # Fall detection example
    fall_payload = {
        "imageUrl": "/snapshots/test_fall.jpg",
        "status": "danger",
        "action": "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện té ngã nghiêm trọng (87%) - Yêu cầu hỗ trợ gấp!",
        "time": datetime.now().isoformat(),
        "eventId": "test_fall_123",
        "confidence": 0.87,
        "eventType": "fall"
    }
    
    print("🔴 Danger Level - Fall Detection:")
    print(json.dumps(fall_payload, indent=2, ensure_ascii=False))
    
    # Seizure detection example
    seizure_payload = {
        "imageUrl": "/snapshots/test_seizure.jpg", 
        "status": "danger",
        "action": "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật nghiêm trọng (72%) - Yêu cầu hỗ trợ gấp!",
        "time": datetime.now().isoformat(),
        "eventId": "test_seizure_456",
        "confidence": 0.72,
        "eventType": "abnormal_behavior"
    }
    
    print("\n🟠 Warning Level - Seizure Detection:")
    print(json.dumps(seizure_payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("🏥 Supabase Realtime Test for Healthcare System")
    print("Mục đích: Test insert database để trigger realtime notification")
    print("")
    
    # Show examples first
    show_mobile_payload_examples()
    
    print("\n" + "=" * 70)
    input("📱 Mở file examples/mobile_realtime_demo.html trong browser trước, rồi nhấn Enter để tiếp tục...")
    
    # Run the test
    test_supabase_insert()
