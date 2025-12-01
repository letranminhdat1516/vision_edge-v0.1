"""
Test Auto-Called Logic - Event ALARM_ACTIVATED sau 3 phút không xử lý → AUTO_CALLED
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import time

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🧪 TEST AUTO-CALLED LOGIC (3 PHÚT)")
print("=" * 80)
print("\n📋 Logic:")
print("   1. Tạo event với lifecycle_state = 'ALARM_ACTIVATED'")
print("   2. Set escalated_at = 3 phút trước (giả lập alarm đã kích hoạt 3 phút)")
print("   3. acknowledged_at = NULL (chưa ai xử lý)")
print("   4. Chờ 10-20 giây để worker check")
print("   5. Worker tự động chuyển → 'AUTOCALLED'")
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

# Calculate time 3 minutes ago
three_minutes_ago = datetime.now() - timedelta(minutes=3)

print("\n" + "=" * 80)
print("📝 TẠO EVENT TEST TRỰC TIẾP TRONG DATABASE")
print("=" * 80)

# Get snapshot_id
snapshot_id = db._create_minimal_snapshot(camera_id, user_id)
if not snapshot_id:
    snapshot_id = str(uuid.uuid4())
    print("⚠️ Using dummy snapshot_id")

# Insert event directly with ALARM_ACTIVATED state and backdated escalated_at
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

event_data = (
    event_id,
    user_id,
    camera_id,
    snapshot_id,
    'fall',  # event_type
    '🚨 KHẨN CẤP - TÉ NGÃ: Test event for AUTO_CALLED logic',  # description
    0.75,  # confidence
    'danger',  # status
    'ALARM_ACTIVATED',  # lifecycle_state - ĐÃ ALARM
    three_minutes_ago,  # escalated_at - 3 PHÚT TRƯỚC
    'alarm_timeout',  # auto_escalation_reason
    None,  # acknowledged_at - CHƯA XỬ LÝ
    False,  # is_canceled
    three_minutes_ago,  # detected_at
    three_minutes_ago,  # created_at
    three_minutes_ago,  # last_action_at
    json.dumps({'test_case': True, 'auto_called_test': True}),  # detection_data
    f'[{three_minutes_ago.isoformat()}] Test event for AUTO_CALLED - Alarm activated 3 minutes ago'  # notes
)

try:
    cursor.execute(insert_query, event_data)
    conn.commit()
    
    print(f"✅ Event created with ALARM_ACTIVATED state!")
    print(f"   🆔 Event ID: {event_id}")
    print(f"   📊 Lifecycle State: ALARM_ACTIVATED")
    print(f"   ⏰ Escalated At: {three_minutes_ago} (3 minutes ago)")
    print(f"   ⚠️ Acknowledged At: NULL (no response)")
    print(f"   🎯 Status: danger")
    
except Exception as e:
    print(f"❌ Failed to create event: {e}")
    import traceback
    traceback.print_exc()
    db.return_connection(conn)
    exit(1)

db.return_connection(conn)

print("\n" + "=" * 80)
print("⏱️  KIỂM TRA TRẠNG THÁI BAN ĐẦU")
print("=" * 80)

conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        lifecycle_state,
        escalated_at,
        acknowledged_at,
        auto_escalation_reason,
        EXTRACT(EPOCH FROM (NOW() - escalated_at)) as seconds_since_alarm
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

event = cursor.fetchone()

if event:
    state = event['lifecycle_state']
    escalated = event['escalated_at']
    acknowledged = event.get('acknowledged_at')
    reason = event.get('auto_escalation_reason')
    seconds_since = event['seconds_since_alarm']
    
    print(f"\n📊 Current State:")
    print(f"   Lifecycle State: {state}")
    print(f"   Escalated At: {escalated}")
    print(f"   Acknowledged At: {acknowledged}")
    print(f"   Seconds Since Alarm: {seconds_since:.0f}s ({seconds_since/60:.1f} minutes)")
    print(f"   Auto Escalation Reason: {reason}")
    
    if seconds_since < 180:
        print(f"\n⚠️ WARNING: Event chưa đủ 3 phút (còn {180-seconds_since:.0f}s)")
        print("   Worker có thể chưa promote ngay lập tức")
    else:
        print(f"\n✅ Event đã đủ 3 phút - Worker sẽ promote trong lần check tiếp theo")

db.return_connection(conn)

print("\n" + "=" * 80)
print("⏳ CHỜ 25 GIÂY ĐỂ WORKER CHECK VÀ PROMOTE")
print("=" * 80)
print("\n💡 EventLifecycleWorker:")
print("   - Chạy mỗi 10 giây")
print("   - Kiểm tra events ALARM_ACTIVATED > 3 phút")
print("   - Tự động chuyển → AUTO_CALLED")
print("\n⚠️ ĐẢM BẢO main.py ĐANG CHẠY trong terminal khác!")
print("\n" + "=" * 80)

# Countdown with checking
auto_called = False
for i in range(25, 0, -1):
    print(f"\r⏱️  Còn lại: {i:2d} giây... ", end='', flush=True)
    time.sleep(1)
    
    # Check every 5 seconds
    if i % 5 == 0 or i <= 5:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT lifecycle_state, last_action_at, notes
            FROM event_detections
            WHERE event_id = %s
        """, (event_id,))
        
        check = cursor.fetchone()
        if check:
            check_state = check['lifecycle_state']
            
            if check_state == 'AUTO_CALLED' and not auto_called:
                auto_called = True
                elapsed = 25 - i
                print(f"\n\n   📞 📞 📞 AUTOCALLED TRIGGERED! 📞 📞 📞")
                print(f"   ⏱️  Detected after: {elapsed} seconds")
                print(f"   📊 New State: {check_state}")
                print(f"   ⏰ Action Time: {check['last_action_at']}")
                
                # Show notes
                notes = check.get('notes', '')
                if 'Auto-called' in notes:
                    print(f"\n   📝 Notes:")
                    for line in notes.split('\n'):
                        if 'Auto-called' in line:
                            print(f"      {line.strip()}")
                
                print(f"\n   💡 Tiếp tục chờ để hoàn tất test...")
        
        db.return_connection(conn)

print("\n\n" + "=" * 80)
print("✅ ĐÃ CHỜ 25 GIÂY - KIỂM TRA KẾT QUẢ CUỐI CÙNG")
print("=" * 80)

# Final check
conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        lifecycle_state,
        escalated_at,
        acknowledged_at,
        last_action_at,
        auto_escalation_reason,
        notes,
        EXTRACT(EPOCH FROM (last_action_at - escalated_at)) as time_to_auto_call
    FROM event_detections
    WHERE event_id = %s
""", (event_id,))

final_event = cursor.fetchone()

if final_event:
    final_state = final_event['lifecycle_state']
    final_escalated = final_event['escalated_at']
    final_acknowledged = final_event.get('acknowledged_at')
    final_action = final_event['last_action_at']
    final_reason = final_event.get('auto_escalation_reason')
    final_notes = final_event.get('notes')
    time_to_call = final_event.get('time_to_auto_call')
    
    print(f"\n📊 Final State:")
    print(f"   Event ID: {event_id}")
    print(f"   Lifecycle State: {final_state}")
    print(f"   Escalated At: {final_escalated}")
    print(f"   Acknowledged At: {final_acknowledged}")
    print(f"   Last Action At: {final_action}")
    print(f"   Auto Escalation Reason: {final_reason}")
    
    if time_to_call:
        print(f"   Time to Auto-Call: {time_to_call:.0f}s ({time_to_call/60:.1f} minutes)")
    
    if final_notes:
        print(f"\n   📝 Notes:")
        for line in final_notes.split('\n'):
            if line.strip():
                print(f"      {line.strip()}")
    
    print("\n" + "=" * 80)
    
    if final_state == 'AUTOCALLED':
        print("✅ ✅ ✅ TEST PASSED! ✅ ✅ ✅")
        print("📞 AUTO-CALL LOGIC HOẠT ĐỘNG ĐÚNG!")
        print("\n💡 Event flow:")
        print("   NOTIFIED → (30s) → ALARM_ACTIVATED → (3min) → AUTOCALLED")
        print("\n🚨 In production:")
        print("   - Trigger emergency call to ambulance/police")
        print("   - Send SMS/call to emergency contacts")
        print("   - Escalate to emergency services")
    elif final_state == 'ALARM_ACTIVATED':
        print("❌ ❌ ❌ TEST FAILED! ❌ ❌ ❌")
        print("📞 AUTOCALL CHƯA ĐƯỢC KÍCH HOẠT!")
        print("\n💡 Possible reasons:")
        print("   1. EventLifecycleWorker chưa chạy")
        print("   2. Worker chưa đến lần check tiếp theo")
        print("   3. Event chưa đủ 3 phút (check time)")
        print("\n🔍 Check:")
        print("   - main.py đang chạy?")
        print("   - Xem log worker trong terminal main.py")
        print("   - Chờ thêm 10-20 giây và check lại")
    else:
        print(f"⚠️ UNEXPECTED STATE: {final_state}")

db.return_connection(conn)

print("=" * 80)
print("🏁 TEST HOÀN TẤT")
print("=" * 80)
