"""
Test Manual Alarm Stop → RESOLVED
Kiểm tra khi tắt alarm thủ công, event tự động chuyển sang RESOLVED
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import time
import requests

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🧪 TEST MANUAL ALARM STOP → RESOLVED")
print("=" * 80)
print("\n📋 Quy trình:")
print("   1. Tạo event với ALARM_ACTIVATED (giả lập alarm đang phát)")
print("   2. Gọi API stop alarm (POST /api/alarm/control enabled=false)")
print("   3. Kiểm tra event tự động chuyển → RESOLVED")
print("=" * 80)

# Import service
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

db = PostgreSQLHealthcareService()

# Get user/camera IDs
user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')

# Get camera_id from database
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT camera_id FROM cameras WHERE user_id = %s AND status = 'active' LIMIT 1", (user_id,))
result = cursor.fetchone()
camera_id = str(result['camera_id']) if result else str(uuid.uuid4())

print(f"\n👤 User ID: {user_id}")
print(f"📷 Camera ID: {camera_id}")

# Create event ID
event_id = str(uuid.uuid4())

print("\n" + "=" * 80)
print("📝 TẠO EVENT VỚI ALARM_ACTIVATED STATE")
print("=" * 80)

# Get snapshot_id
snapshot_id = db._create_minimal_snapshot(camera_id, user_id)
if not snapshot_id:
    snapshot_id = str(uuid.uuid4())

# Insert event with ALARM_ACTIVATED state
insert_query = """
    INSERT INTO event_detections (
        event_id, user_id, camera_id, snapshot_id, event_type,
        event_description, confidence_score, status, 
        lifecycle_state, escalated_at, auto_escalation_reason,
        acknowledged_at, is_canceled,
        detected_at, created_at, last_action_at,
        detection_data, notes
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s
    )
"""

import json

now = datetime.now()

event_data = (
    event_id,
    user_id,
    camera_id,
    snapshot_id,
    'fall',
    '🚨 Test event for manual alarm stop',
    0.85,
    'danger',
    'ALARM_ACTIVATED',  # Giả lập alarm đang phát
    now,
    'test_alarm',
    None,  # acknowledged_at
    False,
    now,
    now,
    now,
    json.dumps({'test_case': True, 'manual_stop_test': True}),
    f'[{now.isoformat()}] Test event - Alarm activated for manual stop test'
)

try:
    cursor.execute(insert_query, event_data)
    conn.commit()
    
    print(f"✅ Event created!")
    print(f"   🆔 Event ID: {event_id}")
    print(f"   📊 State: ALARM_ACTIVATED")
    print(f"   🎯 Status: danger")
    
except Exception as e:
    print(f"❌ Failed to create event: {e}")
    import traceback
    traceback.print_exc()
    db.return_connection(conn)
    exit(1)

db.return_connection(conn)

# Check initial state
print("\n" + "=" * 80)
print("📊 TRẠNG THÁI BAN ĐẦU")
print("=" * 80)

conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT lifecycle_state, status, escalated_at, notes
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

initial = cursor.fetchone()
print(f"\n   Lifecycle State: {initial['lifecycle_state']}")
print(f"   Status: {initial['status']}")
print(f"   Escalated At: {initial['escalated_at']}")

db.return_connection(conn)

# Call API to stop alarm
print("\n" + "=" * 80)
print("🔌 GỌI API ĐỂ TẮT ALARM")
print("=" * 80)

api_url = "http://localhost:8000/api/alarm/control"

# Build payload
payload = {
    "enabled": False,  # TẮT alarm
    "event_id": event_id,
    "user_id": user_id,
    "reason": "Manual stop test - Testing RESOLVED transition"
}

print(f"\n📡 Endpoint: POST {api_url}")
print(f"📦 Payload:")
print(f"   enabled: {payload['enabled']}")
print(f"   event_id: {event_id[:8]}...")
print(f"   user_id: {user_id[:8]}...")
print(f"   reason: {payload['reason']}")

try:
    response = requests.post(api_url, json=payload, timeout=10)
    
    print(f"\n📥 Response:")
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Success: {result.get('success')}")
        print(f"   Message: {result.get('message')}")
        print(f"   Alarm Stopped: {result.get('alarm_stopped')}")
        
        if result.get('success'):
            print(f"\n✅ API call successful!")
        else:
            print(f"\n⚠️ API returned success=false: {result.get('message')}")
    else:
        print(f"\n❌ API call failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print(f"\n❌ Cannot connect to API server!")
    print(f"   Make sure main.py is running (port 8000)")
    print(f"\n💡 Start API server:")
    print(f"   cd src && python main.py")
    exit(1)
    
except Exception as e:
    print(f"\n❌ API call error: {e}")
    exit(1)

# Wait for update to propagate
print("\n⏳ Waiting 2 seconds for database update...")
time.sleep(2)

# Check final state
print("\n" + "=" * 80)
print("✅ KIỂM TRA TRẠNG THÁI SAU KHI TẮT ALARM")
print("=" * 80)

conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        lifecycle_state,
        status,
        escalated_at,
        last_action_at,
        notes
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

final = cursor.fetchone()

if final:
    final_state = final['lifecycle_state']
    final_action = final['last_action_at']
    final_notes = final.get('notes', '')
    
    print(f"\n📊 Final State:")
    print(f"   Event ID: {event_id}")
    print(f"   Lifecycle State: {final_state}")
    print(f"   Status: {final['status']}")
    print(f"   Last Action At: {final_action}")
    
    if final_notes:
        print(f"\n   📝 Notes:")
        for line in final_notes.split('\n'):
            if 'RESOLVED' in line or 'stopped' in line.lower():
                print(f"      {line.strip()}")
    
    print("\n" + "=" * 80)
    
    if final_state == 'RESOLVED':
        print("✅ ✅ ✅ TEST PASSED! ✅ ✅ ✅")
        print("🎉 ALARM STOP → RESOLVED HOẠT ĐỘNG ĐÚNG!")
        print("\n💡 Quy trình:")
        print("   1. API nhận request stop alarm (enabled=false)")
        print("   2. Gửi NOTIFY qua PostgreSQL channel → stop còi")
        print("   3. emergency_alarm_handler nhận → tắt audio")
        print("   4. alarm_api._log_alarm_stop() → update event RESOLVED")
        print("   5. Event chuyển trạng thái: ALARM_ACTIVATED → RESOLVED")
    else:
        print("❌ ❌ ❌ TEST FAILED! ❌ ❌ ❌")
        print(f"🚫 Event vẫn ở state: {final_state}")
        print("\n💡 Expected: RESOLVED")
        print(f"   Actual: {final_state}")
        print("\n🔍 Check:")
        print("   - API server đang chạy? (port 8000)")
        print("   - PostgreSQL connection OK?")
        print("   - Xem log trong terminal main.py")

db.return_connection(conn)

print("=" * 80)
print("🏁 TEST HOÀN TẤT")
print("=" * 80)
