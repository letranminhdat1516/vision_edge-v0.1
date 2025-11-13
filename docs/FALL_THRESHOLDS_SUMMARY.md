# 📊 TÓM TẮT TẤT CẢ FALL DETECTION THRESHOLDS

## 🎯 **1. CORE FALL DETECTOR**

### `src/fall_detection/simple_fall_detector.py`

```python
Line 18: confidence_threshold=0.4  # Giảm từ 0.7 xuống 0.4 để nhạy hơn
Line 27: min_time_interval = 0.8   # Giảm từ 1.0 xuống 0.8 giây
Line 29: max_buffer_size = 3       # Frame buffer
```

**Current:** 0.4 (40%) - BALANCED
**Recommend:** ✅ OK hoặc tăng lên 0.5 (50%) nếu muốn conservative

---

## 🏥 **2. ADVANCED HEALTHCARE PIPELINE**

### `src/service/advanced_healthcare_pipeline.py`

#### Direct Detection (Line 188):

```python
Line 188: if base_fall_confidence >= 0.7:  # ⚠️ RẤT CAO - Direct detection
Line 242: fall_threshold = 0.4             # Enhanced detection threshold
Line 248: min_confirmation_frames = 4      # Cần 4 frames
Line 174: cooldown = 8.0 seconds           # Fall cooldown
```

#### Alert Display (Line 458, 681):

```python
Line 458: elif result['fall_confidence'] > 0.50:  # Warning alert
Line 681: elif fall_confidence > 0.25:            # Display warning
```

**Issue:**

- Direct detection cần **0.7 (70%)** - QUÁ CAO!
- Enhanced detection cần **0.4 (40%)** - Hợp lý
- Cần **4 frames** confirm - Khá cao
- Cooldown **8s** - Hợp lý

---

## 💾 **3. DATABASE STATUS THRESHOLDS**

### `src/service/postgresql_healthcare_service.py`

```python
Line 437: if confidence >= 0.60:  # 'danger' status
Line 439: elif confidence >= 0.40:  # 'warning' status
```

**Status:** ✅ OK - Phù hợp

---

## 🔔 **4. NOTIFICATION THRESHOLDS**

### `src/service/emergency_notification_dispatcher.py`

```python
Line 437: notification_threshold = 0.70  # ⚠️ CAO - Notification threshold
```

### `src/service/database_config_service.py`

```python
Line 56:  'confidence_threshold': 0.7  # Fall detection confidence
Line 95:  "high": 0.35                  # Alert threshold high
Line 96:  "medium": 0.25                # Alert threshold medium
Line 97:  "low": 0.15                   # Alert threshold low
Line 99:  "notification_threshold": 0.40
```

**Issue:** Có 2 notification thresholds khác nhau (0.70 vs 0.40)

---

## 🎥 **5. DUAL CAMERA SYSTEM**

### `src/service/dual_camera_surveillance_system.py`

```python
Line 191: if max_fall_confidence > 0.25:   # Dual camera fusion
Line 563: if fall_confidence > 0.1:        # Debug print
Line 617: if fall_confidence > 0.3:        # Buffer threshold
Line 713: if max_fall_confidence > 0.3:    # Alert check
Line 814: if max_fall_confidence > 0.3:    # Boost threshold
Line 901: fall_agreement(..., 0.3)         # Agreement threshold
Line 1021: if fall_confidence > 0.6:       # Critical display
Line 1023: elif fall_confidence > 0.3:     # Warning display
```

**Status:** ✅ Hợp lý - Dual camera có threshold thấp hơn (0.25)

---

## 📋 **PROBLEM ANALYSIS:**

### ❌ **QUÁ KHẮT KHE:**

1. **Direct Detection: 0.7 (70%)** - QUÁ CAO!

   - File: `advanced_healthcare_pipeline.py` Line 188
   - Khó phát hiện fall trực tiếp

2. **Notification: 0.7 (70%)** - QUÁ CAO!

   - File: `emergency_notification_dispatcher.py` Line 437
   - Khó gửi notification

3. **Min Confirmation Frames: 4** - Hơi cao
   - File: `advanced_healthcare_pipeline.py` Line 248
   - Cần nhiều frames để confirm

### ✅ **HỢP LÝ:**

- Core detector: 0.4 (40%)
- Enhanced threshold: 0.4 (40%)
- Database status: 0.6 danger, 0.4 warning
- Dual camera: 0.25 (25%)

---

## 💡 **RECOMMENDED CHANGES:**

### **Option 1: BALANCED (Recommend)**

```python
# advanced_healthcare_pipeline.py
Line 188: if base_fall_confidence >= 0.50:  # Giảm từ 0.7 → 0.50
Line 248: min_confirmation_frames = 3       # Giảm từ 4 → 3

# emergency_notification_dispatcher.py
Line 437: notification_threshold = 0.50     # Giảm từ 0.7 → 0.50
```

### **Option 2: SENSITIVE (Nhạy hơn)**

```python
# advanced_healthcare_pipeline.py
Line 188: if base_fall_confidence >= 0.40:  # Giảm từ 0.7 → 0.40
Line 248: min_confirmation_frames = 2       # Giảm từ 4 → 2

# emergency_notification_dispatcher.py
Line 437: notification_threshold = 0.40     # Giảm từ 0.7 → 0.40
```

---

## 📂 **FILES NEED TO MODIFY:**

1. ✅ `src/service/advanced_healthcare_pipeline.py` - Lines 188, 248
2. ✅ `src/service/emergency_notification_dispatcher.py` - Line 437
3. ⚠️ (Optional) `src/fall_detection/simple_fall_detector.py` - Line 18
4. ⚠️ (Optional) `.env` - FALL*THRESHOLD*\* variables

---

## 🎯 **CURRENT FLOW:**

```
Fall Detector (0.4)
    ↓
Direct Check (0.7) ← ❌ QUÁ CAO
    ↓ NO
Enhanced Check (0.4)
    ↓
Confirmation (4 frames) ← ⚠️ Hơi cao
    ↓
Cooldown (8s) ← ✅ OK
    ↓
Notification (0.7) ← ❌ QUÁ CAO
```

## 🚀 **PROPOSED FLOW (BALANCED):**

Fall Detector (0.4) ✅ OK
↓
Direct Check (0.45) ✅ DỄ DEMO - nhạy vừa
↓ NO
Enhanced Check (0.4) ✅ OK
↓
Confirmation (3 frames) ✅ NHANH HƠN
↓
Cooldown (8s) ✅ OK
↓
Notification (0.45) ✅ DỄ DEMO
