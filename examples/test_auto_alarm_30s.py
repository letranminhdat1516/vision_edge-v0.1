"""
Test Auto-Alarm Logic - Tạo event DANGER/WARNING và chờ 30s để alarm tự động kích hoạt
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime
import time

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🧪 TEST AUTO-ALARM LOGIC (30 GIÂY)")
print("=" * 80)
print("\n📋 Logic:")
print("   1. Tạo event DANGER/WARNING")
print("   2. Event có lifecycle_state = 'NOTIFIED'")
print("   3. Chờ 30 giây")
print("   4. EventLifecycleWorker tự động chuyển thành 'ALARM_ACTIVATED'")
print("   5. Alarm tự động phát (nếu main.py đang chạy)")
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
db.return_connection(conn)

print(f"\n👤 User ID: {user_id}")
print(f"📷 Camera ID: {camera_id}")

# Create test event
event_data = {
    'event_type': 'fall',
    'user_id': user_id,
    'camera_id': camera_id,
    'confidence': 0.68,
    'status': 'danger',
    'bounding_boxes': [],
    'context': {
        'description': '🆘 KHẨN CẤP - TÉ NGÃ: Phát hiện té ngã nghiêm trọng - Người nằm bất động trên sàn nhà',
        'test_case': True
    },
    'frame': None
}

print("\n" + "=" * 80)
print("📝 TẠO EVENT TEST")
print("=" * 80)
print(f"Event Type: {event_data['event_type']}")
print(f"Status: {event_data['status'].upper()}")
print(f"Confidence: {event_data['confidence']:.2%}")
print(f"Description: {event_data['context']['description'][:80]}...")
print("=" * 80)

print("\n⚡ Đang tạo event trong database...")

try:
    result = db.publish_event_detection(event_data)
    
    if result is None:
        print("\n❌ FAILED: Event bị reject (filtered)")
        exit(1)
    elif isinstance(result, dict) and result.get('filtered'):
        print(f"\n❌ FAILED: Event bị filter")
        print(f"   Reason: {result.get('reason')}")
        exit(1)
    elif isinstance(result, dict) and result.get('event_id'):
        event_id = result.get('event_id')
        print(f"\n✅ SUCCESS: Event đã được tạo!")
        print(f"   Event ID: {event_id}")
    else:
        print(f"\n❓ UNKNOWN: {result}")
        exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("⏱️  KIỂM TRA LIFECYCLE_STATE")
print("=" * 80)

# Check initial state
conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        lifecycle_state,
        acknowledged_at,
        escalated_at,
        created_at,
        last_action_at
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

event = cursor.fetchone()

if event:
    if isinstance(event, dict):
        state = event['lifecycle_state']
        created = event['created_at']
        ack = event.get('acknowledged_at')
        esc = event.get('escalated_at')
    else:
        state = event[1]
        created = event[4]
        ack = event[2]
        esc = event[3]
    
    print(f"\n📊 Trạng thái ban đầu:")
    print(f"   Lifecycle State: {state}")
    print(f"   Created At: {created}")
    print(f"   Acknowledged At: {ack}")
    print(f"   Escalated At: {esc}")
    
    if state != 'NOTIFIED':
        print(f"\n⚠️ WARNING: State không phải 'NOTIFIED' (là '{state}')")
        print("   Auto-alarm có thể không hoạt động!")

db.return_connection(conn)

print("\n" + "=" * 80)
print("⏳ CHỜ 45 GIÂY ĐỂ AUTO-ALARM KÍCH HOẠT")
print("=" * 80)
print("\n💡 EventLifecycleWorker hoạt động:")
print("   - Chạy mỗi 10 giây")
print("   - Sau 30s: Event đủ điều kiện auto-alarm")
print("   - Worker phải đợi đến lần check tiếp theo (có thể 30-40s)")
print("   - Alarm sẽ phát nếu main.py đang chạy")
print("\n⚠️ ĐẢM BẢO main.py ĐANG CHẠY trong terminal khác!")
print("\n" + "=" * 80)

# Countdown with real-time checking
alarm_activated = False
for i in range(45, 0, -1):
    print(f"\r⏱️  Còn lại: {i:2d} giây... ", end='', flush=True)
    time.sleep(1)
    
    # Check every 5 seconds (more frequent)
    if i % 5 == 0 or i <= 10:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT lifecycle_state, escalated_at
            FROM event_detections
            WHERE event_id = %s
        """, (event_id,))
        
        check = cursor.fetchone()
        if check:
            if isinstance(check, dict):
                check_state = check['lifecycle_state']
                check_esc = check.get('escalated_at')
            else:
                check_state = check[0]
                check_esc = check[1]
            
            if check_state == 'ALARM_ACTIVATED' and not alarm_activated:
                alarm_activated = True
                elapsed = 45 - i
                print(f"\n\n   🎉 🎉 🎉 ALARM ĐÃ ĐƯỢC KÍCH HOẠT! 🎉 🎉 🎉")
                print(f"   ⏱️  Thời gian: {elapsed} giây sau khi tạo event")
                print(f"   📊 State: {check_state}")
                print(f"   ⏰ Escalated At: {check_esc}")
                print(f"\n   💡 Tiếp tục chờ để hoàn tất test...")
        
        db.return_connection(conn)

print("\n\n" + "=" * 80)
print("✅ ĐÃ CHỜ 45 GIÂY - KIỂM TRA KẾT QUẢ CUỐI CÙNG")
print("=" * 80)

# Final check
conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        lifecycle_state,
        escalated_at,
        auto_escalation_reason,
        notes,
        created_at
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

final_event = cursor.fetchone()

if final_event:
    if isinstance(final_event, dict):
        final_state = final_event['lifecycle_state']
        final_esc = final_event.get('escalated_at')
        final_reason = final_event.get('auto_escalation_reason')
        final_notes = final_event.get('notes')
        final_created = final_event['created_at']
    else:
        final_state = final_event[1]
        final_esc = final_event[2]
        final_reason = final_event[3]
        final_notes = final_event[4]
        final_created = final_event[5]
    
    print(f"\n📊 Trạng thái cuối cùng:")
    print(f"   Event ID: {event_id}")
    print(f"   Lifecycle State: {final_state}")
    print(f"   Created At: {final_created}")
    print(f"   Escalated At: {final_esc}")
    print(f"   Auto Escalation Reason: {final_reason}")
    
    if final_notes:
        print(f"\n   Notes:")
        for line in final_notes.split('\n'):
            if line.strip():
                print(f"      {line.strip()}")
    
    print("\n" + "=" * 80)
    
    if final_state == 'ALARM_ACTIVATED':
        print("✅ ✅ ✅ TEST PASSED! ✅ ✅ ✅")
        print("🔊 AUTO-ALARM HOẠT ĐỘNG ĐÚNG!")
        print("\n💡 Kiểm tra:")
        print("   - Terminal chạy main.py → Xem log 'ALARM PLAYING'")
        print("   - Nghe tiếng còi báo động")
    elif final_state == 'NOTIFIED':
        print("❌ ❌ ❌ TEST FAILED! ❌ ❌ ❌")
        print("🚫 AUTO-ALARM CHƯA KÍCH HOẠT!")
        print("\n💡 Nguyên nhân có thể:")
        print("   1. EventLifecycleWorker chưa chạy")
        print("   2. main.py không đang chạy")
        print("   3. Worker bị lỗi")
        print("\n🔍 Kiểm tra:")
        print("   - Chạy main.py: cd src && python main.py")
        print("   - Xem log worker trong terminal main.py")
    else:
        print(f"⚠️ State không như mong đợi: {final_state}")

db.return_connection(conn)

print("=" * 80)
print("🏁 TEST HOÀN TẤT")
print("=" * 80)
