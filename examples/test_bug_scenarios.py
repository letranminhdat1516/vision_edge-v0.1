"""
Test Bug Scenarios - Dự đoán các trường hợp có thể xảy ra lỗi
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def log(msg, color=Colors.BLUE):
    print(f"{color}[{datetime.now().strftime('%H:%M:%S')}] {msg}{Colors.END}")

def test_case(name):
    print(f"\n{'='*80}")
    print(f"🧪 TEST CASE: {name}")
    print(f"{'='*80}")

# ============================================================================
# BUG SCENARIO 1: Double Trigger (Race Condition)
# ============================================================================
def test_bug_1_double_trigger():
    """
    BUG: Gọi API enabled=true 2 lần liên tục
    
    Kỳ vọng:
    - Lần 1: NOTIFIED → ALARM_ACTIVATED ✅
    - Lần 2: ALARM_ACTIVATED → ALARM_ACTIVATED (idempotent) ✅
    
    Có thể xảy ra:
    - ❌ Database deadlock (2 UPDATE cùng lúc)
    - ❌ Alarm phát 2 lần đồng thời (audio overlap)
    - ❌ escalated_at bị ghi đè (mất track lần trigger đầu)
    """
    test_case("Double Trigger - Race Condition")
    
    event_id = "test-bug-1"
    payload = {
        "event_id": event_id,
        "user_id": "user-123",
        "camera_id": "cam-456",
        "enabled": True
    }
    
    log("🔥 Trigger 1st time...", Colors.YELLOW)
    r1 = requests.post(f"{BASE_URL}/api/alarm/control", json=payload)
    log(f"Response 1: {r1.status_code} - {r1.json()}", Colors.GREEN)
    
    log("🔥 Trigger 2nd time IMMEDIATELY (no delay)...", Colors.YELLOW)
    r2 = requests.post(f"{BASE_URL}/api/alarm/control", json=payload)
    log(f"Response 2: {r2.status_code} - {r2.json()}", Colors.GREEN)
    
    log("⚠️ Check: Có audio overlap không? escalated_at có đúng không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 2: Stop Non-Existent Event
# ============================================================================
def test_bug_2_stop_nonexistent():
    """
    BUG: Gọi API enabled=false cho event không tồn tại
    
    Kỳ vọng:
    - ✅ API trả về success (vì không có WHERE condition fail)
    - ⚠️ Nhưng không có row nào được update
    
    Có thể xảy ra:
    - ❌ API báo success nhưng thực tế không làm gì
    - ❌ Không có error handling cho "event not found"
    - ❌ rows_updated = 0 nhưng vẫn log "RESOLVED"
    """
    test_case("Stop Non-Existent Event")
    
    payload = {
        "event_id": "non-existent-event-xyz",
        "user_id": "user-123",
        "camera_id": "cam-456",
        "enabled": False
    }
    
    log("🔥 Trying to stop non-existent event...", Colors.YELLOW)
    response = requests.post(f"{BASE_URL}/api/alarm/control", json=payload)
    log(f"Response: {response.status_code} - {response.json()}", Colors.GREEN)
    
    log("⚠️ Check: API có báo lỗi 'event not found' không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 3: Trigger Cancelled Event
# ============================================================================
def test_bug_3_trigger_cancelled():
    """
    BUG: Gọi API enabled=true cho event đã bị cancel (is_canceled=TRUE)
    
    Kỳ vọng:
    - ⚠️ Có nên trigger alarm cho cancelled event không?
    
    Có thể xảy ra:
    - ❌ API trigger thành công cho cancelled event
    - ❌ Lifecycle state đổi nhưng event vẫn là cancelled
    - ❌ Conflict: is_canceled=TRUE nhưng lifecycle_state=ALARM_ACTIVATED
    """
    test_case("Trigger Cancelled Event")
    
    # Giả sử có event đã cancelled trong DB
    payload = {
        "event_id": "cancelled-event-001",
        "user_id": "user-123",
        "camera_id": "cam-456",
        "enabled": True
    }
    
    log("🔥 Trying to trigger cancelled event...", Colors.YELLOW)
    response = requests.post(f"{BASE_URL}/api/alarm/control", json=payload)
    log(f"Response: {response.status_code} - {response.json()}", Colors.GREEN)
    
    log("⚠️ Check: Có kiểm tra is_canceled trước khi trigger không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 4: Rapid Toggle (On-Off-On-Off)
# ============================================================================
def test_bug_4_rapid_toggle():
    """
    BUG: Toggle alarm liên tục: ON → OFF → ON → OFF
    
    Kỳ vọng:
    - Mỗi lần đổi state đúng
    
    Có thể xảy ra:
    - ❌ State transitions bị lộn xộn (race condition)
    - ❌ Audio service crash (start/stop quá nhanh)
    - ❌ PostgreSQL NOTIFY queue overflow
    - ❌ Emergency handler không kịp xử lý
    """
    test_case("Rapid Toggle - Stress Test")
    
    event_id = "test-bug-4"
    
    for i in range(5):
        # BẬT
        log(f"🔥 Round {i+1}: Trigger ON...", Colors.YELLOW)
        requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": event_id,
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": True
        })
        time.sleep(0.1)  # 100ms delay
        
        # TẮT
        log(f"🔥 Round {i+1}: Trigger OFF...", Colors.YELLOW)
        requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": event_id,
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": False
        })
        time.sleep(0.1)
    
    log("⚠️ Check: Audio có bị crash không? State có đúng không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 5: Missing Required Fields
# ============================================================================
def test_bug_5_missing_fields():
    """
    BUG: Gọi API thiếu các field bắt buộc
    
    Kỳ vọng:
    - ✅ FastAPI validation reject (422 Unprocessable Entity)
    
    Có thể xảy ra:
    - ❌ Backend crash nếu không có validation
    - ❌ PostgreSQL error nếu field NULL
    """
    test_case("Missing Required Fields")
    
    # Missing event_id
    log("🔥 Test 1: Missing event_id...", Colors.YELLOW)
    try:
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": True
        })
        log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
    except Exception as e:
        log(f"Error: {e}", Colors.RED)
    
    # Missing user_id
    log("🔥 Test 2: Missing user_id...", Colors.YELLOW)
    try:
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": "test",
            "camera_id": "cam-456",
            "enabled": True
        })
        log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
    except Exception as e:
        log(f"Error: {e}", Colors.RED)
    
    # Missing enabled
    log("🔥 Test 3: Missing enabled field...", Colors.YELLOW)
    try:
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": "test",
            "user_id": "user-123",
            "camera_id": "cam-456"
        })
        log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
    except Exception as e:
        log(f"Error: {e}", Colors.RED)

# ============================================================================
# BUG SCENARIO 6: Invalid Data Types
# ============================================================================
def test_bug_6_invalid_types():
    """
    BUG: Gọi API với sai data type
    
    Kỳ vọng:
    - ✅ FastAPI validation reject
    
    Có thể xảy ra:
    - ❌ enabled="yes" thay vì boolean
    - ❌ event_id là number thay vì string
    """
    test_case("Invalid Data Types")
    
    # enabled as string
    log("🔥 Test 1: enabled='yes' (should be boolean)...", Colors.YELLOW)
    try:
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": "test",
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": "yes"  # ❌ String instead of boolean
        })
        log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
    except Exception as e:
        log(f"Error: {e}", Colors.RED)
    
    # event_id as number
    log("🔥 Test 2: event_id=123 (should be string)...", Colors.YELLOW)
    try:
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": 123,  # ❌ Number instead of string
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": True
        })
        log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
    except Exception as e:
        log(f"Error: {e}", Colors.RED)

# ============================================================================
# BUG SCENARIO 7: PostgreSQL Connection Lost
# ============================================================================
def test_bug_7_db_connection_lost():
    """
    BUG: PostgreSQL service disconnect giữa chừng
    
    Kỳ vọng:
    - ✅ API trả về error message rõ ràng
    
    Có thể xảy ra:
    - ❌ API crash với 500 Internal Server Error
    - ❌ Alarm phát nhưng không update DB
    - ❌ Không có error logging
    """
    test_case("PostgreSQL Connection Lost")
    
    log("⚠️ Manual test: Stop PostgreSQL service và thử gọi API", Colors.RED)
    log("   Expected: API should return clear error message", Colors.YELLOW)

# ============================================================================
# BUG SCENARIO 8: Worker and API Conflict
# ============================================================================
def test_bug_8_worker_api_conflict():
    """
    BUG: Worker auto-alarm và API manual trigger cùng lúc
    
    Scenario:
    1. Event created → NOTIFIED
    2. Đợi 29s
    3. User gọi API enabled=true (29s) → ALARM_ACTIVATED
    4. Worker check (30s) → thấy NOTIFIED... nhưng đã ALARM_ACTIVATED rồi!
    
    Có thể xảy ra:
    - ❌ Worker trigger lại (duplicate alarm)
    - ❌ State bị ghi đè: ALARM_ACTIVATED → ALARM_ACTIVATED (mất track manual trigger)
    - ❌ escalated_at bị ghi đè
    """
    test_case("Worker-API Conflict")
    
    log("⚠️ Manual test:", Colors.RED)
    log("   1. Tạo event danger → NOTIFIED", Colors.YELLOW)
    log("   2. Đợi 29s", Colors.YELLOW)
    log("   3. Gọi API enabled=true", Colors.YELLOW)
    log("   4. Đợi 1s → Worker check (30s)", Colors.YELLOW)
    log("   5. Check: Worker có trigger lại không?", Colors.YELLOW)

# ============================================================================
# BUG SCENARIO 9: Concurrent Requests - Multiple Users
# ============================================================================
def test_bug_9_concurrent_users():
    """
    BUG: Nhiều users trigger alarm cho cùng 1 event đồng thời
    
    Kỳ vọng:
    - ✅ Tất cả requests đều success (idempotent)
    
    Có thể xảy ra:
    - ❌ Database lock timeout
    - ❌ Last-write-wins (mất track user nào trigger trước)
    - ❌ notes field bị corrupt (concurrent append)
    """
    test_case("Concurrent Users - Multiple Triggers")
    
    import threading
    
    event_id = "test-bug-9"
    results = []
    
    def trigger(user):
        r = requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": event_id,
            "user_id": user,
            "camera_id": "cam-456",
            "enabled": True
        })
        results.append((user, r.status_code, r.json()))
    
    log("🔥 Simulating 5 users triggering same event simultaneously...", Colors.YELLOW)
    threads = []
    for i in range(5):
        t = threading.Thread(target=trigger, args=(f"user-{i}",))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    for user, status, data in results:
        log(f"{user}: {status} - {data}", Colors.GREEN)
    
    log("⚠️ Check: Database có bị deadlock không? notes có đủ 5 entries không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 10: Long Event ID (SQL Injection?)
# ============================================================================
def test_bug_10_sql_injection():
    """
    BUG: Thử SQL injection qua event_id
    
    Kỳ vọng:
    - ✅ Parameterized query → không bị injection
    
    Có thể xảy ra:
    - ❌ Nếu không dùng parameterized query
    """
    test_case("SQL Injection Test")
    
    malicious_payloads = [
        "'; DROP TABLE event_detections; --",
        "abc' OR '1'='1",
        "abc'; UPDATE event_detections SET lifecycle_state='HACKED'; --"
    ]
    
    for payload in malicious_payloads:
        log(f"🔥 Testing payload: {payload}", Colors.YELLOW)
        try:
            r = requests.post(f"{BASE_URL}/api/alarm/control", json={
                "event_id": payload,
                "user_id": "user-123",
                "camera_id": "cam-456",
                "enabled": True
            })
            log(f"Response: {r.status_code} - {r.json()}", Colors.GREEN)
        except Exception as e:
            log(f"Error: {e}", Colors.RED)
    
    log("✅ If all requests handled safely → SQL injection protected", Colors.GREEN)

# ============================================================================
# BUG SCENARIO 11: Very Long Notes Field
# ============================================================================
def test_bug_11_long_notes():
    """
    BUG: Trigger/stop nhiều lần → notes field quá dài
    
    Có thể xảy ra:
    - ❌ Database field overflow (nếu có VARCHAR limit)
    - ❌ Query timeout (UPDATE quá chậm với long string)
    """
    test_case("Long Notes Field")
    
    event_id = "test-bug-11"
    
    log("🔥 Triggering alarm 100 times to overflow notes field...", Colors.YELLOW)
    for i in range(100):
        requests.post(f"{BASE_URL}/api/alarm/control", json={
            "event_id": event_id,
            "user_id": "user-123",
            "camera_id": "cam-456",
            "enabled": i % 2 == 0  # Toggle ON/OFF
        })
        if i % 10 == 0:
            log(f"   Progress: {i}/100", Colors.BLUE)
    
    log("⚠️ Check: notes field có bị truncate không? Performance có chậm không?", Colors.RED)

# ============================================================================
# BUG SCENARIO 12: API Server Restart During Request
# ============================================================================
def test_bug_12_server_restart():
    """
    BUG: API server restart giữa chừng transaction
    
    Có thể xảy ra:
    - ❌ NOTIFY sent nhưng DB không update (transaction không commit)
    - ❌ DB updated nhưng NOTIFY không send (handler chưa start)
    """
    test_case("Server Restart During Request")
    
    log("⚠️ Manual test:", Colors.RED)
    log("   1. Gọi API enabled=true", Colors.YELLOW)
    log("   2. NGAY LẬP TỨC restart API server (Ctrl+C)", Colors.YELLOW)
    log("   3. Check: alarm có phát không? DB có update không?", Colors.YELLOW)

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🐛 BUG HUNTING - Test Scenarios")
    print("="*80)
    
    try:
        # Automated tests
        test_bug_1_double_trigger()
        time.sleep(2)
        
        test_bug_2_stop_nonexistent()
        time.sleep(2)
        
        test_bug_4_rapid_toggle()
        time.sleep(2)
        
        test_bug_5_missing_fields()
        time.sleep(2)
        
        test_bug_6_invalid_types()
        time.sleep(2)
        
        test_bug_9_concurrent_users()
        time.sleep(2)
        
        test_bug_10_sql_injection()
        time.sleep(2)
        
        # Manual tests (require human intervention)
        print("\n" + "="*80)
        print("📋 MANUAL TESTS (Cần test thủ công)")
        print("="*80)
        test_bug_3_trigger_cancelled()
        test_bug_7_db_connection_lost()
        test_bug_8_worker_api_conflict()
        test_bug_12_server_restart()
        
        print("\n" + "="*80)
        print("✅ ALL AUTOMATED TESTS COMPLETED!")
        print("⚠️  Please run manual tests as described above")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        log("❌ Cannot connect to API server!", Colors.RED)
        log("💡 Make sure to run: python src/main.py", Colors.YELLOW)
    except Exception as e:
        log(f"❌ Test failed: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
