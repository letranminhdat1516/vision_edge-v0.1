# 🎯 THRESHOLD CONFIGURATION GUIDE

**Hướng dẫn điều chỉnh độ nhạy phát hiện Fall & Seizure**

---

## 📊 **TỔNG QUAN CÁC THRESHOLD**

### 🔴 **1. FALL DETECTION (Phát hiện Té Ngã)**

| File                               | Line | Giá trị                            | Mô tả                            | Cách điều chỉnh        |
| ---------------------------------- | ---- | ---------------------------------- | -------------------------------- | ---------------------- |
| `.env`                             | 29   | `FALL_THRESHOLD_HIGH=0.25`         | Severity HIGH threshold          | **GIẢM = nhạy hơn**    |
| `.env`                             | 30   | `FALL_THRESHOLD_MEDIUM=0.18`       | Severity MEDIUM threshold        | GIẢM = nhạy hơn        |
| `.env`                             | 31   | `FALL_THRESHOLD_LOW=0.12`          | Severity LOW threshold           | GIẢM = nhạy hơn        |
| `.env`                             | 32   | `FALL_NOTIFICATION_THRESHOLD=0.30` | Mobile notification threshold    | GIẢM = nhạy hơn        |
| `advanced_healthcare_pipeline.py`  | 188  | `>= 0.30`                          | **Direct detection threshold**   | **GIẢM = nhạy hơn** ⭐ |
| `fall_detection_service.py`        | 4    | `confidence_threshold=0.15`        | Base detector threshold          | GIẢM = nhạy hơn        |
| `postgresql_healthcare_service.py` | 437  | `>= 0.60`                          | Database severity HIGH mapping   | GIẢM = nhạy hơn        |
| `postgresql_healthcare_service.py` | 439  | `>= 0.40`                          | Database severity MEDIUM mapping | GIẢM = nhạy hơn        |

**⭐ Threshold quan trọng nhất:**

- `advanced_healthcare_pipeline.py` line 188: `>= 0.30`
  - **Giảm xuống (0.25, 0.20)** = Nhạy hơn, dễ detect fall
  - **Tăng lên (0.35, 0.40)** = Ít nhạy hơn, giảm false positive

---

### 🟠 **2. SEIZURE DETECTION (Phát hiện Co Giật)**

| File                               | Line | Giá trị                               | Mô tả                            | Cách điều chỉnh           |
| ---------------------------------- | ---- | ------------------------------------- | -------------------------------- | ------------------------- |
| `.env`                             | 35   | `SEIZURE_THRESHOLD_HIGH=0.80`         | Severity HIGH threshold          | **TĂNG = ít nhạy hơn**    |
| `.env`                             | 36   | `SEIZURE_THRESHOLD_MEDIUM=0.65`       | Severity MEDIUM threshold        | TĂNG = ít nhạy hơn        |
| `.env`                             | 37   | `SEIZURE_THRESHOLD_LOW=0.50`          | Severity LOW threshold           | TĂNG = ít nhạy hơn        |
| `.env`                             | 38   | `SEIZURE_NOTIFICATION_THRESHOLD=0.85` | Mobile notification threshold    | TĂNG = ít nhạy hơn        |
| `main.py`                          | 234  | `alert_threshold=0.70`                | **Seizure Predictor alert**      | **TĂNG = ít nhạy hơn** ⭐ |
| `main.py`                          | 235  | `warning_threshold=0.55`              | **Seizure Predictor warning**    | **TĂNG = ít nhạy hơn** ⭐ |
| `seizure_detection_service.py`     | 26   | `confidence_threshold=0.50`           | Base detector threshold          | TĂNG = ít nhạy hơn        |
| `vsvig_detector.py`                | 258  | `< 0.30`                              | Normal activity threshold        | Cố định                   |
| `vsvig_detector.py`                | 263  | `>= self.confidence_threshold`        | Detection threshold              | Dùng từ service           |
| `postgresql_healthcare_service.py` | 445  | `>= 0.50`                             | Database severity HIGH mapping   | TĂNG = ít nhạy hơn        |
| `postgresql_healthcare_service.py` | 447  | `>= 0.30`                             | Database severity MEDIUM mapping | TĂNG = ít nhạy hơn        |

**⭐ Threshold quan trọng nhất:**

- `main.py` line 234-235:
  ```python
  alert_threshold=0.70,      # 70% confidence mới báo động
  warning_threshold=0.55     # 55% confidence chỉ cảnh báo
  ```
  - **Tăng lên (0.75, 0.80)** = Ít nhạy hơn, giảm spam detection
  - **Giảm xuống (0.65, 0.60)** = Nhạy hơn, detect dễ hơn

---

## 🎯 **HƯỚNG DẪN ĐIỀU CHỈNH NHANH**

### ✅ **Muốn TĂNG NHẠY Fall (detect dễ hơn):**

1. **File: `advanced_healthcare_pipeline.py`** (Line 188)

   ```python
   # Từ:
   if base_fall_confidence >= 0.30:
   # Xuống:
   if base_fall_confidence >= 0.25:  # hoặc 0.20
   ```

2. **File: `.env`** (Line 29-32)
   ```bash
   FALL_THRESHOLD_HIGH=0.20          # Từ 0.25 → 0.20
   FALL_NOTIFICATION_THRESHOLD=0.25   # Từ 0.30 → 0.25
   ```

### ✅ **Muốn GIẢM NHẠY Seizure (ít spam hơn):**

1. **File: `main.py`** (Line 234-235)

   ```python
   # Từ:
   individual_seizure_predictor = SeizurePredictor(
       temporal_window=3,
       alert_threshold=0.70,
       warning_threshold=0.55
   )
   # Tăng lên:
   individual_seizure_predictor = SeizurePredictor(
       temporal_window=3,
       alert_threshold=0.75,      # 0.70 → 0.75
       warning_threshold=0.60     # 0.55 → 0.60
   )
   ```

2. **File: `.env`** (Line 35-38)
   ```bash
   SEIZURE_THRESHOLD_HIGH=0.85       # Từ 0.80 → 0.85
   SEIZURE_NOTIFICATION_THRESHOLD=0.90  # Từ 0.85 → 0.90
   ```

---

## 📝 **CÁC NGƯỠNG HIỆN TẠI (Sau khi điều chỉnh)**

### **Fall Detection:**

- **Direct Detection:** `>= 0.30` (30% confidence)
- **Base Detector:** `>= 0.15` (15% confidence)
- **Severity HIGH:** `>= 0.25` (.env)
- **Database HIGH:** `>= 0.60` (postgresql_healthcare_service.py)

### **Seizure Detection:**

- **Alert Threshold:** `>= 0.70` (70% confidence)
- **Warning Threshold:** `>= 0.55` (55% confidence)
- **Base Detector:** `>= 0.50` (50% confidence)
- **Severity HIGH:** `>= 0.80` (.env)
- **Database HIGH:** `>= 0.50` (postgresql_healthcare_service.py)

---

## ⏱️ **COOLDOWN TIMERS (Anti-Spam)**

**Mục đích**: Tránh spam notifications khi event liên tục xảy ra

| Event Type  | Cooldown    | File                              | Line | Mô tả                                                              |
| ----------- | ----------- | --------------------------------- | ---- | ------------------------------------------------------------------ |
| **Fall**    | **3 giây**  | `advanced_healthcare_pipeline.py` | 184  | Sau 1 fall detection, phải đợi 3s mới detect fall tiếp theo        |
| **Seizure** | **10 giây** | `advanced_healthcare_pipeline.py` | 387  | Sau 1 seizure detection, phải đợi 10s mới detect seizure tiếp theo |

**Cách điều chỉnh**:

```python
# Line 184 - Fall cooldown
FALL_COOLDOWN = 3.0  # 3 giây → Có thể tăng lên 5.0, 7.0

# Line 387 - Seizure cooldown
SEIZURE_COOLDOWN = 10.0  # 10 giây → Có thể tăng lên 15.0, 20.0
```

**Lý do**:

- **Fall**: 3s đủ để tránh spam khi người té xuống rồi nằm
- **Seizure**: 10s vì co giật thường kéo dài, tránh tạo nhiều event liên tục

---

## ⚠️ **LƯU Ý QUAN TRỌNG**

1. **Sau khi sửa `.env`**: KHÔNG cần restart, config tự reload
2. **Sau khi sửa `.py` files**: BẮT BUỘC restart `main.py`
3. **Test từng threshold một**: Đừng sửa nhiều threshold cùng lúc
4. **Giá trị khuyến nghị:**
   - Fall: 0.25 - 0.35 (dễ detect)
   - Seizure: 0.65 - 0.75 (tránh spam)
5. **Cooldown khuyến nghị:**
   - Fall: 3-5 giây
   - Seizure: 10-15 giây

---

## 🔧 **QUICK REFERENCE**

```bash
# Fall: GIẢM số = nhạy hơn
FALL_THRESHOLD_HIGH=0.25

# Seizure: TĂNG số = ít nhạy hơn
SEIZURE_THRESHOLD_HIGH=0.80

# Main.py Seizure Predictor: TĂNG số = ít nhạy hơn
alert_threshold=0.70
warning_threshold=0.55

# Advanced Pipeline Fall: GIẢM số = nhạy hơn
if base_fall_confidence >= 0.30:
```

---

**Cập nhật lần cuối:** 2025-11-14  
**Phiên bản:** 1.0  
**Tác giả:** AI Assistant + User Configuration
