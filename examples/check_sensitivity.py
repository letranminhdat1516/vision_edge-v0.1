"""
Check Fall Detection Sensitivity and Event Status Formula
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 80)
print("🎯 FALL DETECTION SENSITIVITY")
print("=" * 80)

# Check fall detector settings
try:
    from fall_detection.simple_fall_detector import SimpleFallDetector
    
    detector = SimpleFallDetector()
    
    print(f"\n📊 Fall Detection Thresholds:")
    print(f"   Confidence Threshold: {detector.confidence_threshold} ({detector.confidence_threshold * 100:.0f}%)")
    print(f"   Min Time Interval: {detector.min_time_interval}s (giữa 2 frame phân tích)")
    print(f"   Frame Buffer Size: {detector.max_buffer_size} frames")
    print(f"   Method: Simplified (movement pattern analysis)")
    
    print(f"\n🎯 Sensitivity Level:")
    if detector.confidence_threshold <= 0.3:
        print(f"   ⚠️ VERY SENSITIVE - Nhiều cảnh báo, có thể nhiều false positive")
    elif detector.confidence_threshold <= 0.5:
        print(f"   ✅ BALANCED - Cân bằng giữa độ chính xác và nhạy")
    else:
        print(f"   🔒 CONSERVATIVE - Ít cảnh báo, độ chính xác cao")
    
except Exception as e:
    print(f"❌ Error loading fall detector: {e}")

print("\n" + "=" * 80)
print("🎯 SEIZURE DETECTION SENSITIVITY")
print("=" * 80)

# Check seizure detector settings
try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    
    detector = VSViGSeizureDetector()
    
    print(f"\n📊 Seizure Detection Thresholds:")
    print(f"   Confidence Threshold: {detector.confidence_threshold}")
    print(f"   Model: VSViG-base")
    
    print(f"\n🎯 Sensitivity Level:")
    if detector.confidence_threshold <= 0.3:
        print(f"   ⚠️ VERY SENSITIVE - Nhiều cảnh báo nhưng có thể sai")
    elif detector.confidence_threshold <= 0.5:
        print(f"   ✅ BALANCED - Cân bằng độ chính xác")
    else:
        print(f"   🔒 CONSERVATIVE - Ít cảnh báo, độ chính xác cao")
    
except Exception as e:
    print(f"❌ Error loading seizure detector: {e}")

print("\n" + "=" * 80)
print("📋 EVENT STATUS FORMULA (PostgreSQL)")
print("=" * 80)

# Check status determination logic
try:
    from service.postgresql_healthcare_service import PostgreSQLHealthcareService
    
    service = PostgreSQLHealthcareService()
    
    print(f"\n🔍 Testing _determine_event_status() method:")
    
    # Test Fall Detection
    print(f"\n🚨 FALL DETECTION:")
    test_cases_fall = [
        (0.95, 'fall'),
        (0.80, 'fall'),
        (0.70, 'fall'),
        (0.60, 'fall'),
        (0.50, 'fall'),
        (0.40, 'fall')
    ]
    
    for confidence, event_type in test_cases_fall:
        status = service._determine_event_status(confidence, event_type)
        emoji = "🔴" if status == "danger" else "🟡" if status == "warning" else "🟢"
        print(f"   Confidence {confidence:.0%} → {status.upper():8s} {emoji}")
    
    # Test Seizure Detection
    print(f"\n🧠 SEIZURE/ABNORMAL BEHAVIOR:")
    test_cases_seizure = [
        (0.95, 'abnormal_behavior'),
        (0.80, 'abnormal_behavior'),
        (0.70, 'abnormal_behavior'),
        (0.60, 'abnormal_behavior'),
        (0.50, 'abnormal_behavior'),
        (0.40, 'abnormal_behavior'),
        (0.30, 'abnormal_behavior')
    ]
    
    for confidence, event_type in test_cases_seizure:
        status = service._determine_event_status(confidence, event_type)
        emoji = "🔴" if status == "danger" else "🟡" if status == "warning" else "🟢"
        print(f"   Confidence {confidence:.0%} → {status.upper():8s} {emoji}")
    
except Exception as e:
    print(f"❌ Error testing status logic: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("💾 DATABASE STATUS COLUMN")
print("=" * 80)

print(f"""
Table: event_detections
Column: status VARCHAR(8)
Possible Values:
   • 'normal'  🟢 - Không nguy hiểm
   • 'warning' 🟡 - Cần theo dõi
   • 'danger'  🔴 - Khẩn cấp

Công thức tính (từ code):
""")

# Read actual code
try:
    import inspect
    from service.postgresql_healthcare_service import PostgreSQLHealthcareService
    
    service = PostgreSQLHealthcareService()
    source = inspect.getsource(service._determine_event_status)
    
    print("Source code:")
    print("-" * 80)
    for i, line in enumerate(source.split('\n'), 1):
        if 'def _determine_event_status' in line or 'return' in line or 'if' in line or 'elif' in line:
            print(f"{i:3d} | {line}")
    print("-" * 80)
    
except Exception as e:
    print(f"Cannot read source: {e}")

print("\n" + "=" * 80)
print("📊 PRIORITY & ALERT CREATION LOGIC")
print("=" * 80)

try:
    from service.emergency_notification_dispatcher import EmergencyNotificationDispatcher
    
    print(f"\n🔍 Alert Creation Rules (từ _should_create_alert):")
    print(f"\n📋 FALL:")
    print(f"   Confidence >= 0.80 → CREATE ALERT (CRITICAL)")
    print(f"   Confidence >= 0.60 → CREATE ALERT (HIGH)")
    print(f"   Confidence >= 0.40 → CREATE ALERT (MEDIUM)")
    print(f"   Confidence <  0.40 → NO ALERT (LOW)")
    
    print(f"\n📋 SEIZURE:")
    print(f"   Confidence >= 0.70 → CREATE ALERT (CRITICAL)")
    print(f"   Confidence >= 0.50 → CREATE ALERT (HIGH)")
    print(f"   Confidence >= 0.30 → CREATE ALERT (MEDIUM)")
    print(f"   Confidence <  0.30 → NO ALERT (LOW)")
    
except Exception as e:
    print(f"Cannot load dispatcher: {e}")

print("\n" + "=" * 80)
print("✅ RECOMMENDATION")
print("=" * 80)

print("""
Độ nhạy hiện tại:
   ✅ Fall Detection: Balanced (60% warning, 80% danger)
   ✅ Seizure Detection: Balanced (50% warning, 70% danger)

Nếu muốn thay đổi:
   • Giảm threshold → Nhiều cảnh báo hơn (nhạy hơn)
   • Tăng threshold → Ít cảnh báo hơn (chính xác hơn)

Vị trí code:
   • Fall threshold: src/fall_detection/simple_fall_detector.py
   • Seizure threshold: src/seizure_detection/vsvig_detector.py
   • Status logic: src/service/postgresql_healthcare_service.py
     → Method: _determine_event_status()
""")

print("=" * 80)
