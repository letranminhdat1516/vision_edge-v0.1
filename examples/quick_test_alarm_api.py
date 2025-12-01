"""
Quick Test - Lấy event từ DB và test API alarm
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

print("=" * 80)
print("🧪 QUICK TEST ALARM API")
print("=" * 80)

# Connect DB
print("\n💾 Kết nối database...")
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

db = PostgreSQLHealthcareService()
conn = db.get_connection()
cursor = conn.cursor()

user_id = os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863')
print(f"👤 User ID: {user_id}")

# Get recent DANGER/WARNING events
print("\n🔍 Tìm events DANGER/WARNING gần nhất...")
cursor.execute("""
    SELECT 
        event_id, 
        event_type, 
        status,
        lifecycle_state,
        confidence_score,
        camera_id,
        detected_at
    FROM event_detections
    WHERE user_id = %s
      AND status IN ('danger', 'warning')
      AND detected_at > NOW() - INTERVAL '7 days'
    ORDER BY detected_at DESC
    LIMIT 15
""", (user_id,))

events = cursor.fetchall()

if not events:
    print("\n❌ Không tìm thấy event DANGER/WARNING!")
    print("\n💡 Tạo event test:")
    print("   1. cd src && python main.py")
    print("   2. Nhấn 'e' để tạo event")
    db.return_connection(conn)
    exit(1)

print(f"\n📋 Tìm thấy {len(events)} events:")
print("-" * 80)
print(f"{'#':<3} {'Event ID':<10} {'Type':<15} {'Status':<10} {'State':<25} {'Conf':<6} {'Time'}")
print("-" * 80)

for i, ev in enumerate(events, 1):
    if isinstance(ev, dict):
        ev_id = str(ev['event_id'])[:8]
        ev_type = ev['event_type']
        status = ev['status']
        state = ev['lifecycle_state']
        conf = float(ev.get('confidence_score', 0))
        cam_id = str(ev.get('camera_id', ''))
        detected = str(ev['detected_at'])[:19]
    else:
        ev_id = str(ev[0])[:8]
        ev_type = ev[1]
        status = ev[2]
        state = ev[3]
        conf = float(ev[4]) if ev[4] else 0
        cam_id = str(ev[5])
        detected = str(ev[6])[:19]
    
    status_icon = "🔴" if status == 'danger' else "🟠"
    print(f"{i:<3} {ev_id:<10} {ev_type:<15} {status_icon}{status:<9} {state:<25} {conf:.2f}   {detected}")

print("-" * 80)

# Select event
choice = input("\n👉 Chọn event (1-15) hoặc Enter để chọn event đầu: ").strip()

if not choice:
    idx = 0
    print("   ➜ Đã chọn event #1")
else:
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(events):
            print("❌ Lựa chọn không hợp lệ!")
            db.return_connection(conn)
            exit(1)
    except ValueError:
        print("❌ Phải nhập số!")
        db.return_connection(conn)
        exit(1)

selected = events[idx]

if isinstance(selected, dict):
    event_id = str(selected['event_id'])
    event_type = selected['event_type']
    status = selected['status']
    state = selected['lifecycle_state']
    camera_id = str(selected['camera_id'])
    conf = float(selected.get('confidence_score', 0))
else:
    event_id = str(selected[0])
    event_type = selected[1]
    status = selected[2]
    state = selected[3]
    conf = float(selected[4]) if selected[4] else 0
    camera_id = str(selected[5])

db.return_connection(conn)

print("\n" + "=" * 80)
print("🎯 EVENT ĐÃ CHỌN")
print("=" * 80)
print(f"📌 Event ID: {event_id}")
print(f"📌 Type: {event_type}")
print(f"📌 Status: {status}")
print(f"📌 State: {state}")
print(f"📌 Confidence: {conf:.2%}")
print(f"📌 User ID: {user_id}")
print(f"📌 Camera ID: {camera_id}")
print("=" * 80)

# API endpoint
API_URL = "http://localhost:8000/api/alarm/control"

# Menu
print("\n📋 CHỌN HÀNH ĐỘNG:")
print("   1. 🔊 BẬT ALARM (enabled: true)")
print("   2. 🔇 TẮT ALARM (enabled: false)")
print("   3. ❌ Hủy")

action = input("\n👉 Chọn (1/2/3): ").strip()

if action == '3' or action.lower() == 'q':
    print("❌ Đã hủy")
    exit(0)

if action not in ['1', '2']:
    print("❌ Lựa chọn không hợp lệ!")
    exit(1)

# Prepare payload
enabled = True if action == '1' else False
action_text = "BẬT" if enabled else "TẮT"

payload = {
    "event_id": event_id,
    "user_id": user_id,
    "camera_id": camera_id,
    "enabled": enabled
}

print(f"\n🔥 GỌI API ĐỂ {action_text} ALARM...")
print(f"POST {API_URL}")
print("\nPayload:")
print(json.dumps(payload, indent=2))

# Confirm
confirm = input(f"\n✅ Xác nhận {action_text} alarm? (y/n): ").strip().lower()

if confirm != 'y':
    print("❌ Đã hủy")
    exit(0)

# Call API
print("\n📡 Đang gọi API...")

try:
    response = requests.post(API_URL, json=payload, timeout=10)
    
    print(f"\n{'='*80}")
    print(f"📡 RESPONSE (Status: {response.status_code})")
    print(f"{'='*80}")
    
    try:
        response_data = response.json()
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
    except:
        print(response.text)
    
    print(f"{'='*80}")
    
    if response.status_code == 200:
        print("\n✅ ✅ ✅ THÀNH CÔNG! ✅ ✅ ✅")
        if enabled:
            print("🔊 ALARM ĐÃ ĐƯỢC BẬT!")
            print("\n💡 Kiểm tra:")
            print("   - Terminal chạy main.py → Xem log alarm handler")
            print("   - Nghe tiếng còi báo động (nếu có loa)")
        else:
            print("🔇 ALARM ĐÃ ĐƯỢC TẮT!")
            print("\n💡 Kiểm tra:")
            print("   - Terminal chạy main.py → Xem log 'ALARM STOPPED'")
            print("   - Còi đã ngừng phát")
    else:
        print(f"\n❌ LỖI! Status code: {response.status_code}")

except requests.exceptions.ConnectionError:
    print("\n❌ KHÔNG KẾT NỐI ĐƯỢC API!")
    print("\n💡 Giải pháp:")
    print("   1. Chạy main.py trong terminal khác:")
    print("      cd d:\\FPT\\Capstone\\vision_edge-v0.1\\src")
    print("      python main.py")
    print("   2. Đợi đến khi thấy: '✅ Alarm API Server ready on port 8000'")
    print("   3. Chạy lại script này")

except requests.exceptions.Timeout:
    print("\n❌ TIMEOUT! API không phản hồi trong 10 giây")

except Exception as e:
    print(f"\n❌ LỖI: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🏁 HOÀN TẤT")
print("=" * 80)
