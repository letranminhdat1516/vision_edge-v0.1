"""
Test script cho Alarm Control API
Chỉ cần 1 API duy nhất để BẬT/TẮT alarm
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8000"
CONTROL_ENDPOINT = f"{BASE_URL}/api/alarm/control"

def test_health_check():
    """Test health check endpoint"""
    print("=" * 60)
    print("1️⃣ Testing Health Check...")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_trigger_alarm(event_id, user_id, camera_id):
    """BẬT ALARM - enabled=true"""
    print("=" * 60)
    print("2️⃣ Testing BẬT ALARM (enabled=true)...")
    print("=" * 60)
    
    payload = {
        "event_id": event_id,
        "user_id": user_id,
        "camera_id": camera_id,
        "enabled": True  # BẬT ALARM
    }
    
    print(f"Request: POST {CONTROL_ENDPOINT}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(CONTROL_ENDPOINT, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_stop_alarm(event_id, user_id, camera_id):
    """TẮT ALARM - enabled=false"""
    print("=" * 60)
    print("3️⃣ Testing TẮT ALARM (enabled=false)...")
    print("=" * 60)
    
    payload = {
        "event_id": event_id,
        "user_id": user_id,
        "camera_id": camera_id,
        "enabled": False  # TẮT ALARM
    }
    
    print(f"Request: POST {CONTROL_ENDPOINT}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(CONTROL_ENDPOINT, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_status():
    """Get alarm status"""
    print("=" * 60)
    print("4️⃣ Testing Get Alarm Status...")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/alarm/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("\n🧪 TESTING ALARM CONTROL API")
    print("API chỉ có 1 endpoint duy nhất: POST /api/alarm/control")
    print("- enabled=true  → BẬT alarm")
    print("- enabled=false → TẮT alarm")
    print()
    
    # Thay đổi các giá trị này theo database của bạn
    EVENT_ID = "test-event-001"
    USER_ID = "b7757b17-4b5e-4f21-86db-5d6e5afe81c7"  # Thay bằng user_id thật
    CAMERA_ID = "test-camera-001"
    
    try:
        # 1. Health check
        test_health_check()
        time.sleep(1)
        
        # 2. BẬT alarm
        test_trigger_alarm(EVENT_ID, USER_ID, CAMERA_ID)
        time.sleep(2)
        
        # 3. Check status
        test_get_status()
        time.sleep(1)
        
        # 4. TẮT alarm
        test_stop_alarm(EVENT_ID, USER_ID, CAMERA_ID)
        time.sleep(1)
        
        # 5. Check status again
        test_get_status()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server!")
        print("💡 Make sure to run: python src/main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")
