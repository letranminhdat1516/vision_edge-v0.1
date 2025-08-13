#!/usr/bin/env python3
"""
Test Supabase Realtime Integration
Chạy file này để test insert event thật vào Supabase và xem realtime trên HTML demo
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class SupabaseRealtimeIntegrationTest:
    def __init__(self):
        # Supabase configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')  # Changed from SUPABASE_ANON_KEY
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong .env file")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        print(f"✅ Supabase client initialized")
        print(f"🔗 URL: {self.supabase_url}")
        
    def create_test_event(self, event_type="fall", confidence=0.85):
        """Tạo một event test để insert vào Supabase"""
        
        # Determine status based on confidence
        if event_type == "fall":
            if confidence < 0.6:
                status = "normal"
            elif confidence < 0.8:
                status = "warning"
            else:
                status = "danger"
        else:  # abnormal_behavior (seizure)
            if confidence < 0.5:
                status = "normal"
            elif confidence < 0.7:
                status = "warning"
            else:
                status = "danger"
        
        # Generate action message
        confidence_percent = int(confidence * 100)
        if status == "normal":
            action = f"Tình huống bình thường - {event_type} được phát hiện với độ tin cậy {confidence_percent}%"
        elif status == "warning":
            action = f"⚠️ Cảnh báo - Phát hiện {event_type} với độ tin cậy {confidence_percent}%. Cần theo dõi."
        else:  # danger
            action = f"🚨 KHẨN CẤP - Phát hiện {event_type} với độ tin cậy {confidence_percent}%. Cần hỗ trợ ngay lập tức!"
        
        event_data = {
            'event_type': event_type,
            'confidence_score': confidence,
            'detected_at': datetime.now().isoformat(),
            'camera_id': 'test_camera_01',
            'location': 'Test Room A',
            'status': status,
            'action_message': action,
            'metadata': {
                'test_mode': True,
                'confidence_percent': confidence_percent,
                'created_by': 'supabase_realtime_test'
            }
        }
        
        return event_data
    
    async def insert_test_event(self, event_data):
        """Insert event vào Supabase database"""
        try:
            result = self.supabase.table('event_detections').insert(event_data).execute()
            
            if result.data:
                event_id = result.data[0].get('id')
                print(f"✅ Event inserted successfully:")
                print(f"   📋 ID: {event_id}")
                print(f"   🎯 Type: {event_data['event_type']}")
                print(f"   📊 Confidence: {event_data['confidence_score']:.2f}")
                print(f"   🚨 Status: {event_data['status']}")
                print(f"   💬 Action: {event_data['action_message']}")
                return result.data[0]
            else:
                print("❌ No data returned from insert")
                return None
                
        except Exception as e:
            print(f"❌ Error inserting event: {e}")
            return None
    
    async def test_multiple_events(self):
        """Test với nhiều loại events khác nhau"""
        
        test_scenarios = [
            # Normal events
            {'event_type': 'fall', 'confidence': 0.45, 'description': 'Fall - Normal'},
            {'event_type': 'abnormal_behavior', 'confidence': 0.35, 'description': 'Seizure - Normal'},
            
            # Warning events  
            {'event_type': 'fall', 'confidence': 0.70, 'description': 'Fall - Warning'},
            {'event_type': 'abnormal_behavior', 'confidence': 0.60, 'description': 'Seizure - Warning'},
            
            # Danger events
            {'event_type': 'fall', 'confidence': 0.90, 'description': 'Fall - Danger'},
            {'event_type': 'abnormal_behavior', 'confidence': 0.85, 'description': 'Seizure - Danger'},
        ]
        
        print("\n🧪 Testing multiple event scenarios...")
        print("=" * 60)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📝 Test {i}/6: {scenario['description']}")
            
            # Create test event
            event_data = self.create_test_event(
                event_type=scenario['event_type'],
                confidence=scenario['confidence']
            )
            
            # Insert to Supabase
            result = await self.insert_test_event(event_data)
            
            if result:
                print(f"   ✨ Event sent to realtime channel!")
                print(f"   🔄 Check HTML demo for realtime notification")
            
            # Wait between events
            print("   ⏱️  Waiting 3 seconds...")
            await asyncio.sleep(3)
        
        print(f"\n🎉 All {len(test_scenarios)} test events completed!")
        print("📱 Check your HTML demo for realtime notifications")

    async def test_single_event(self, event_type="fall", confidence=0.85):
        """Test với một event duy nhất"""
        
        print(f"\n🎯 Testing single event:")
        print(f"   Type: {event_type}")  
        print(f"   Confidence: {confidence}")
        
        # Create and insert event
        event_data = self.create_test_event(event_type, confidence)
        result = await self.insert_test_event(event_data)
        
        if result:
            print(f"\n✨ Event sent successfully!")
            print(f"📱 Check HTML demo at: http://localhost:8000/mobile_realtime_demo.html")
            print(f"🔄 The event should appear in realtime!")
        
        return result

async def main():
    """Main test function"""
    
    print("🚀 Supabase Realtime Integration Test")
    print("=" * 50)
    
    try:
        # Initialize test
        test = SupabaseRealtimeIntegrationTest()
        
        # Ask user what to test
        print("\nChọn loại test:")
        print("1. Test một event đơn lẻ")
        print("2. Test nhiều events (6 scenarios)")
        print("3. Test event nguy hiểm (danger)")
        
        choice = input("\nNhập lựa chọn (1/2/3): ").strip()
        
        if choice == "1":
            await test.test_single_event("fall", 0.75)
        elif choice == "2":
            await test.test_multiple_events()
        elif choice == "3":
            await test.test_single_event("abnormal_behavior", 0.92)
        else:
            print("Chọn mặc định: Test event đơn lẻ")
            await test.test_single_event()
        
        print("\n" + "="*50)
        print("🎯 Test completed!")
        print("📱 Mở HTML demo để xem realtime notifications:")
        print("   http://localhost:8000/mobile_realtime_demo.html")
        print("\n💡 Lưu ý: Đảm bảo HTML demo đã được cấu hình đúng Supabase URL/Key")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
