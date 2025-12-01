"""
Test tạo event giả với caption "đứng" để kiểm tra filter
"""

import sys
from pathlib import Path
import random
import uuid
from datetime import datetime

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🧪 TEST TẠO EVENT GIẢ VỚI CAPTION 'ĐỨNG'")
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

# Test cases với caption có "đứng"
test_cases = [
    {
        'event_type': 'fall',
        'status': 'danger',
        'confidence': 0.65,
        'description': '🆘 KHẨN CẤP - TÉ NGÃ: Một người phụ nữ đang đứng trước cửa sổ trong phòng khách - Phát hiện ngã đổ',
        'expected': 'FILTERED (có từ đứng)'
    },
    {
        'event_type': 'fall',
        'status': 'warning',
        'confidence': 0.45,
        'description': '⚠️ CẢNH BÁO TÉ NGÃ: Người già đang đứng không vững gần bàn - Cần theo dõi',
        'expected': 'FILTERED (có từ đứng)'
    },
    {
        'event_type': 'seizure',
        'status': 'danger',
        'confidence': 0.70,
        'description': '🆘 KHẨN CẤP - ĐỘT QUỴ: Phát hiện đột quỵ nghiêm trọng - Người nằm trên sàn',
        'expected': 'SAVED (có đột quỵ, không có đứng)'
    },
    {
        'event_type': 'fall',
        'status': 'danger',
        'confidence': 0.68,
        'description': '🆘 KHẨN CẤP: Phát hiện té ngã nghiêm trọng - Người nằm bất động trên sàn nhà',
        'expected': 'SAVED (có ngã, không có đứng)'
    },
]

print("\n" + "=" * 80)
print("📋 BẮT ĐẦU TEST")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(test_cases)}")
    print(f"{'='*80}")
    print(f"Event Type: {test['event_type']}")
    print(f"Status: {test['status'].upper()}")
    print(f"Confidence: {test['confidence']:.2%}")
    print(f"Description: {test['description'][:80]}...")
    print(f"Expected: {test['expected']}")
    print(f"{'-'*80}")
    
    # Create event data
    event_data = {
        'event_type': test['event_type'],
        'user_id': user_id,
        'camera_id': camera_id,
        'confidence': test['confidence'],
        'status': test['status'],
        'bounding_boxes': [],
        'context': {
            'description': test['description'],
            'test_case': True
        },
        'frame': None  # No frame for test
    }
    
    # Try to publish
    print("\n⚡ Đang gọi publish_event_detection()...")
    
    try:
        result = db.publish_event_detection(event_data)
        
        if result is None:
            print("❌ RESULT: None - Event bị REJECT (không lưu DB)")
            print("   ✅ Filter hoạt động ĐÚNG!")
            actual = 'FILTERED'
        elif isinstance(result, dict) and result.get('filtered'):
            print(f"🚫 RESULT: Filtered")
            print(f"   Reason: {result.get('reason')}")
            print(f"   Description: {result.get('description', '')[:80]}...")
            print("   ✅ Filter hoạt động ĐÚNG!")
            actual = 'FILTERED'
        elif isinstance(result, dict) and result.get('event_id'):
            event_id = result.get('event_id')
            print(f"✅ RESULT: Saved to DB")
            print(f"   Event ID: {event_id}")
            print("   ⚠️ Event đã được LƯU vào database")
            actual = 'SAVED'
        else:
            print(f"❓ RESULT: Unknown - {result}")
            actual = 'UNKNOWN'
        
        # Check expectation
        expected_result = 'FILTERED' if 'FILTERED' in test['expected'] else 'SAVED'
        
        if actual == expected_result:
            print(f"\n✅ ✅ ✅ TEST PASSED! ✅ ✅ ✅")
            print(f"   Expected: {expected_result}")
            print(f"   Actual: {actual}")
        else:
            print(f"\n❌ ❌ ❌ TEST FAILED! ❌ ❌ ❌")
            print(f"   Expected: {expected_result}")
            print(f"   Actual: {actual}")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Delay between tests
    import time
    time.sleep(1)

print("\n" + "=" * 80)
print("🏁 TEST HOÀN TẤT")
print("=" * 80)

# Check recent events in database
print("\n📊 KIỂM TRA DATABASE:")
print("-" * 80)

conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        event_id,
        event_type,
        status,
        confidence_score,
        LEFT(event_description, 80) as short_desc,
        detected_at
    FROM event_detections
    WHERE user_id = %s
      AND detected_at > NOW() - INTERVAL '5 minutes'
    ORDER BY detected_at DESC
    LIMIT 10
""", (user_id,))

events = cursor.fetchall()

if events:
    print(f"\nTìm thấy {len(events)} events trong 5 phút gần đây:\n")
    for ev in events:
        if isinstance(ev, dict):
            ev_id = str(ev['event_id'])[:8]
            ev_type = ev['event_type']
            status = ev['status']
            conf = float(ev.get('confidence_score', 0))
            desc = ev.get('short_desc', '')
            detected = str(ev['detected_at'])[:19]
        else:
            ev_id = str(ev[0])[:8]
            ev_type = ev[1]
            status = ev[2]
            conf = float(ev[3]) if ev[3] else 0
            desc = ev[4] if len(ev) > 4 else ''
            detected = str(ev[5])[:19] if len(ev) > 5 else ''
        
        has_standing = 'đứng' in desc.lower() if desc else False
        icon = "⚠️" if has_standing else "✅"
        
        print(f"{icon} {ev_id}... | {ev_type:10s} | {status:10s} | {conf:.2f}")
        print(f"   {desc}")
        print(f"   {detected}")
        if has_standing:
            print(f"   🚫 WARNING: Có từ 'đứng' nhưng vẫn lưu DB!")
        print()
else:
    print("\n❌ Không có event nào trong 5 phút gần đây")

db.return_connection(conn)

print("=" * 80)
print("💡 KẾT LUẬN:")
print("   - Nếu events có 'đứng' bị FILTERED → Filter hoạt động ✅")
print("   - Nếu events có 'đứng' vẫn lưu DB → Filter CHƯA hoạt động ❌")
print("=" * 80)
