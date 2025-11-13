# 📊 TÓM TẮT TẤT CẢ SEIZURE THRESHOLDS

## 🎯 **1. CORE DETECTION THRESHOLD (Quan trọng nhất)**

### `src/seizure_detection/vsvig_detector.py`

```python
Line 42: confidence_threshold: float = 0.01  # ⚠️ CỰC THẤP (1%)
Line 258: if seizure_confidence < 0.3:  # Very low confidence = normal
Line 263: elif seizure_confidence >= self.confidence_threshold:  # Detected
Line 272: elif seizure_confidence >= self.confidence_threshold * 0.85:  # Warning
Line 400: if seizure_confidence < 0.1:  # Giảm xuống 0.1 để cực kỳ nhạy
```

**Recommend:** Tăng `confidence_threshold` từ **0.01** → **0.30** hoặc **0.50**

---

## 🚨 **2. ALERT LEVEL THRESHOLDS**

### `src/service/advanced_healthcare_pipeline.py`

```python
Line 346: seizure_threshold = 0.02  # ⚠️ CỰC THẤP (2%)
Line 349: if final_seizure_confidence > seizure_threshold:  # Critical
Line 353: elif final_seizure_confidence > warning_threshold:  # Warning
Line 425: elif final_seizure_confidence > warning_threshold and motion_level > 0.2:
Line 453: elif result['seizure_confidence'] > 0.45 and motion_level > 0.7:  # Giảm từ 0.5 xuống 0.45
```

**Recommend:**

- `seizure_threshold` từ **0.02** → **0.50** (critical)
- `warning_threshold` từ **0.01** → **0.30** (warning)

---

## 💾 **3. DATABASE STATUS THRESHOLDS**

### `src/service/postgresql_healthcare_service.py`

```python
Line 445: if confidence >= 0.50:  # high threshold for seizures → 'danger'
Line 447: elif confidence >= 0.30:  # medium threshold for seizures → 'warning'
```

**Status:** ✅ OK - Phù hợp

---

## 🔔 **4. NOTIFICATION THRESHOLDS**

### `src/service/emergency_notification_dispatcher.py`

```python
Line 556: notification_threshold = seizure_threshold_config.get('notification_threshold', 0.60)
```

### `src/service/database_config_service.py`

```python
Line 62: 'confidence_threshold': float(os.getenv('SEIZURE_DETECTION_CONFIDENCE', '0.6'))
Line 103: "high": float(os.getenv('SEIZURE_THRESHOLD_HIGH', '0.30'))
Line 104: "medium": float(os.getenv('SEIZURE_THRESHOLD_MEDIUM', '0.20'))
Line 105: "low": float(os.getenv('SEIZURE_THRESHOLD_LOW', '0.12'))
```

**Status:** ⚠️ Có thể cần điều chỉnh tuỳ mức độ nhạy mong muốn

---

## 🎥 **5. DUAL CAMERA SYSTEM THRESHOLDS**

### `src/service/dual_camera_surveillance_system.py`

```python
Line 176: if result.fall_confidence > 0.3 or result.seizure_confidence > 0.3:
Line 213: if fused_result.max_seizure_confidence > 0.20:  # ⚠️ THẤP (20%)
Line 579: if seizure_confidence > 0.1:  # Print debug
Line 619: if seizure_confidence > 0.3:  # Buffer
Line 714: fused_result.max_seizure_confidence > 0.3:
Line 816: if max_seizure_confidence > 0.3:
Line 902: seizure_agreement = calculate_confidence_agreement(seizure_confidences, 0.3)
Line 1021: if detection_result.fall_confidence > 0.6 or detection_result.seizure_confidence > 0.6:
Line 1023: elif detection_result.fall_confidence > 0.3 or detection_result.seizure_confidence > 0.3:
Line 1053: if detection_result.seizure_confidence > 0.6:  # Critical display
Line 1056: elif detection_result.seizure_confidence > 0.3:  # Warning display
```

**Recommend:** Tăng `> 0.20` → `> 0.50`

---

## 📋 **RECOMMENDED CHANGES:**

### ✅ **Option 1: BALANCED (Recommend)**

```python
# vsvig_detector.py
confidence_threshold = 0.30  # 30%

# advanced_healthcare_pipeline.py
seizure_threshold = 0.50    # 50% for critical
warning_threshold = 0.30     # 30% for warning

# dual_camera_surveillance_system.py
max_seizure_confidence > 0.40  # 40% for dual camera
```

### ✅ **Option 2: CONSERVATIVE (Chính xác cao)**

```python
# vsvig_detector.py
confidence_threshold = 0.50  # 50%

# advanced_healthcare_pipeline.py
seizure_threshold = 0.60    # 60% for critical
warning_threshold = 0.40     # 40% for warning

# dual_camera_surveillance_system.py
max_seizure_confidence > 0.50  # 50% for dual camera
```

---

## 📂 **FILES NEED TO MODIFY:**

1. ✅ `src/seizure_detection/vsvig_detector.py` - Line 42
2. ✅ `src/service/advanced_healthcare_pipeline.py` - Line 346
3. ✅ `src/service/dual_camera_surveillance_system.py` - Line 213
4. ⚠️ (Optional) `src/service/database_config_service.py` - Lines 103-105
5. ⚠️ (Optional) `.env` file - SEIZURE*THRESHOLD*\* variables

---

## 🎯 **CURRENT PROBLEM:**

**TOO SENSITIVE!** Với threshold 0.01-0.02, hệ thống sẽ:

- ✅ Phát hiện mọi động thái nhỏ
- ❌ Rất nhiều FALSE POSITIVE
- ❌ Gây phiền hà cho user
- ❌ Giảm độ tin cậy của hệ thống

**SOLUTION:** Tăng threshold lên ít nhất 0.30-0.50 để cân bằng!
