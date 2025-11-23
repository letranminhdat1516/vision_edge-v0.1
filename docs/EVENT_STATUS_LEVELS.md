# Event Status Levels - 5 Level Classification System

## Overview

Hệ thống phân loại 5 mức độ trạng thái sự kiện để xác định mức độ nguy hiểm và hành động cần thiết.

## 5 Status Levels

### 1. 🔴 DANGER (Nguy hiểm cao)

**Định nghĩa:** Nằm ngã bất động, tình trạng nguy hiểm cần cứu hộ ngay lập tức

**Điều kiện:**

- **Fall (Té ngã):**

  - Confidence ≥ 0.60 (60%)
  - Fall type = `slow_collapse` (đột quỵ) → EXTRA DANGER
  - Người nằm bất động trên sàn

- **Seizure (Co giật):**
  - Confidence ≥ 0.50 (50%)
  - Chuyển động co giật xác nhận

**Hành động:**

- ✅ Lưu vào database
- ✅ Gửi FCM notification (High Priority)
- ✅ Chụp 5 ảnh multi-angle
- ✅ Phát cảnh báo âm thanh
- ✅ Đề xuất gọi cấp cứu

**Event Types:**

- `fall` (confidence ≥ 0.60)
- `fall_stroke` (slow_collapse)
- `seizure` (confidence ≥ 0.50)

---

### 2. 🟠 WARNING (Cảnh báo)

**Định nghĩa:** Mang tính báo động, chưa tới mức nguy hiểm nhưng cần theo dõi

**Điều kiện:**

- **Fall Warning:**

  - Confidence: 0.40 - 0.59
  - Có dấu hiệu té nhưng chưa chắc chắn

- **Seizure Warning:**
  - Confidence ≥ 0.45
  - Motion level ≥ 0.70
  - Chuyển động bất thường nhưng chưa xác nhận co giật

**Hành động:**

- ✅ Lưu vào database
- ✅ Gửi FCM notification (Normal Priority)
- ✅ Theo dõi thêm
- ❌ KHÔNG phát cảnh báo âm thanh

**Event Types:**

- `fall_warning` (confidence 0.40-0.59)
- `seizure_warning` (confidence ≥ 0.45, motion ≥ 0.70)

---

### 3. 🟡 SUSPECT (Nghi ngờ)

**Định nghĩa:** Nghi ngờ hành động có thể xảy ra nguy hiểm, cần quan sát

**Điều kiện:**

- **Pre-fall Risk:**

  - Fall confidence: 0.20 - 0.39
  - Tư thế không ổn định, có thể sắp té

- **Abnormal Movement:**
  - Seizure confidence: 0.15 - 0.44
  - Chuyển động bất thường, không giống hoạt động bình thường

**Hành động:**

- ✅ Lưu vào database
- ❌ KHÔNG gửi notification
- ✅ Ghi log để phân tích

**Event Types:**

- `pre_fall_risk` (confidence 0.20-0.39)
- `abnormal_movement` (confidence 0.15-0.44)

---

### 4. 🟢 NORMAL (Bình thường)

**Định nghĩa:** Hoạt động bình thường, không có dấu hiệu bất thường

**Điều kiện:**

- Fall confidence < 0.20
- Seizure confidence < 0.15
- Motion level bình thường (thường < 0.3)
- Người đang đứng, ngồi, đi lại bình thường

**Hành động:**

- ✅ Lưu vào database (mỗi 1 giây = 30 frames)
- ✅ Chụp 1 ảnh (mỗi 1 giây)
- ❌ KHÔNG gửi notification
- ✅ Ghi log để tracking hoạt động hàng ngày

**Event Types:**

- `normal_activity` (hoạt động chung)
- `walking` (đi bộ)
- `sitting` (ngồi)
- `standing` (đứng)

---

## 🔍 NORMAL LOGIC - Chi tiết

### **Tại sao cần log NORMAL?**

1. **Timeline tracking:** Xem lịch sử hoạt động của người già trong ngày
2. **Pattern analysis:** Phân tích thói quen sinh hoạt (ngủ, ăn, đi lại)
3. **Baseline data:** Dữ liệu nền để so sánh khi có bất thường
4. **Absence detection:** Phát hiện khi không có hoạt động bình thường (có thể nguy hiểm)
5. **Activity report:** Báo cáo hoạt động hàng ngày cho người nhà

### **Logic hoạt động:**

```python
# Kiểm tra mỗi frame (30 FPS)
if fall_confidence < 0.20 AND seizure_confidence < 0.15:
    alert_level = 'normal'

    # Log mỗi 1 giây (30 frames)
    if current_frame % 30 == 0:
        # 📸 Chụp 1 ảnh snapshot
        snapshot = capture_single_image(frame)

        # 💾 Lưu vào database
        save_event({
            'status': 'normal',
            'event_type': 'normal_activity',
            'confidence': 0.1,
            'snapshot_id': snapshot.id,
            'timestamp': NOW()
        })
```

### **Tần suất lưu:**

- **30 FPS video** → 1 log/giây = 1 log/30 frames
- **1 giờ** = 3,600 logs (3.6K)
- **24 giờ** = 86,400 logs (86.4K)

### **Ví dụ Timeline:**

```
08:00:00 - NORMAL: standing (đứng)
08:00:01 - NORMAL: walking (đi bộ)
08:00:02 - NORMAL: walking
08:00:05 - NORMAL: sitting (ngồi)
08:00:15 - NORMAL: sitting
08:15:00 - SUSPECT: pre_fall_risk (tư thế không ổn định)
08:15:02 - WARNING: fall_warning (nghi ngờ té)
08:15:03 - DANGER: fall (té ngã xác nhận) ⚠️
```

### **Database Query - Hoạt động trong ngày:**

```sql
-- Tổng thời gian hoạt động bình thường
SELECT
    DATE(detected_at) as date,
    COUNT(*) as normal_logs,
    COUNT(*) / 3600.0 as hours_active  -- Giả sử 1 log/giây
FROM event_detections
WHERE status = 'normal'
  AND detected_at >= CURRENT_DATE
GROUP BY DATE(detected_at);
```

### **Optimization để giảm storage:**

```python
# Option 1: Tăng interval (2 giây thay vì 1 giây)
should_log_normal = self.stats['total_frames'] % 60 == 0  # Mỗi 2s

# Option 2: Chỉ log khi có thay đổi motion
if motion_level > 0.05:  # Có chuyển động
    should_log_normal = True

# Option 3: Aggregate hàng giờ
# Thay vì lưu mỗi giây, lưu summary mỗi giờ
```

### **Benefits:**

✅ **Detect absence:** Nếu không có log normal trong 5 phút → Người có thể nằm bất động

✅ **Activity patterns:** Phân tích giờ ngủ, giờ thức, tần suất đi lại

✅ **Health insights:** Giảm hoạt động đột ngột = dấu hiệu sức khỏe xấu đi

✅ **Caregiver report:** "Bà ngoại đã đi bộ 45 phút hôm nay"

---

### 5. ⚪ UNKNOWN (Không rõ)

**Định nghĩa:** Hành động không rõ ràng, phát hiện nhưng không phân loại được

**Điều kiện:**

- Confidence > 0 nhưng không đủ để phân loại
- Hành động không khớp với pattern nào
- Chất lượng detection thấp

**Hành động:**

- ✅ Lưu vào database
- ❌ KHÔNG gửi notification
- ✅ Ghi log để cải thiện model

**Event Types:**

- `unclear_activity`
- `ambiguous_detection`

---

## Database Schema

```sql
CREATE TYPE event_status_enum AS ENUM (
    'danger',    -- Nguy hiểm cao
    'warning',   -- Cảnh báo
    'suspect',   -- Nghi ngờ
    'normal',    -- Bình thường
    'unknown'    -- Không rõ
);
```

## Event Classification Logic

### Fall Detection

```
if confidence >= 0.60:
    if fall_type == 'slow_collapse':
        status = 'danger' (ĐỘT QUỴ - CRITICAL)
    else:
        status = 'danger' (TÉ NGÃ XÁC NHẬN)

elif confidence >= 0.40:
    status = 'warning' (NGHI NGỜ TÉ)

elif confidence >= 0.20:
    status = 'suspect' (CÓ THỂ SẮP TÉ)

else:
    status = 'normal' or 'unknown'
```

### Seizure Detection

```
if confidence >= 0.50:
    status = 'danger' (CO GIẬT XÁC NHẬN)

elif confidence >= 0.30:
    status = 'warning' (NGHI NGỜ CO GIẬT)

elif confidence >= 0.15:
    status = 'suspect' (CHUYỂN ĐỘNG BẤT THƯỜNG)

else:
    status = 'normal' or 'unknown'
```

## Priority Matrix

| Status      | FCM Notification   | Audio Alert | Database Log | Snapshot Capture          |
| ----------- | ------------------ | ----------- | ------------ | ------------------------- |
| **danger**  | ✅ High Priority   | ✅ Yes      | ✅ Immediate | ✅ 5 images (multi-angle) |
| **warning** | ✅ Normal Priority | ❌ No       | ✅ Immediate | ✅ 5 images               |
| **suspect** | ❌ No              | ❌ No       | ✅ Immediate | ✅ 1 image                |
| **normal**  | ❌ No              | ❌ No       | ✅ Every 1s  | ✅ 1 image (every 1s)     |
| **unknown** | ❌ No              | ❌ No       | ✅ Immediate | ✅ 1 image                |

## Notification Examples

### DANGER

```json
{
  "title": "🚨 EMERGENCY: Fall Detected",
  "body": "SLOW COLLAPSE (Possible Stroke) - Duration: 1.25s",
  "priority": "high",
  "data": {
    "alert_level": "danger",
    "fall_type": "slow_collapse",
    "confidence": 0.85,
    "action": "call_ambulance"
  }
}
```

### WARNING

```json
{
  "title": "⚠️ Warning: Fall Suspected",
  "body": "Person may be falling - Confidence: 45%",
  "priority": "normal",
  "data": {
    "alert_level": "warning",
    "confidence": 0.45,
    "action": "monitor"
  }
}
```

### SUSPECT

```json
// No notification sent - only logged to database
{
  "event_type": "pre_fall_risk",
  "alert_level": "suspect",
  "confidence": 0.28,
  "logged_at": "2024-11-23T10:30:45Z"
}
```

## Analytics Queries

### Get all dangerous events

```sql
SELECT * FROM event_detections
WHERE status = 'danger'
ORDER BY detected_at DESC;
```

### Get stroke cases (slow collapse)

```sql
SELECT * FROM event_detections
WHERE status = 'danger'
  AND context_data->>'fall_type' = 'slow_collapse'
ORDER BY detected_at DESC;
```

### Daily activity summary

```sql
SELECT
    status,
    COUNT(*) as count,
    AVG(confidence_score::numeric) as avg_confidence
FROM event_detections
WHERE detected_at >= NOW() - INTERVAL '1 day'
GROUP BY status
ORDER BY
    CASE status
        WHEN 'danger' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'suspect' THEN 3
        WHEN 'normal' THEN 4
        WHEN 'unknown' THEN 5
    END;
```

## Implementation Details

### Fall Velocity Analysis

Phân biệt té nhanh vs té chậm để xác định mức độ nguy hiểm:

- **Fast Fall** (< 0.5s): Té bình thường → `danger` (multiplier 1.0x)
- **Moderate Fall** (0.5-1.0s): Té vừa → `danger` (multiplier 1.1x)
- **Slow Collapse** (≥ 1.0s): Đột quỵ/yếu sức → `danger` (multiplier 1.3x)

### Context Metadata

```json
{
  "alert_level": "danger",
  "fall_type": "slow_collapse",
  "fall_duration": 1.25,
  "fall_velocity": 180.5,
  "motion_level": 0.15,
  "detection_method": "rapid_downward"
}
```

## Mobile App Integration

### UI Color Coding

- 🔴 **DANGER**: Red background, urgent animation
- 🟠 **WARNING**: Orange background, attention indicator
- 🟡 **SUSPECT**: Yellow background, observe label
- 🟢 **NORMAL**: Green background, no action
- ⚪ **UNKNOWN**: Gray background, info icon

### Action Buttons

- **DANGER**: "Call Ambulance" + "View Camera"
- **WARNING**: "Check Status" + "View Camera"
- **SUSPECT**: "View Details"
- **NORMAL**: "View Timeline"
- **UNKNOWN**: "Report Issue"

---

## Summary

✅ **5 mức độ rõ ràng**: danger → warning → suspect → normal → unknown

✅ **Tất cả được log vào database**: Tracking đầy đủ mọi hoạt động

✅ **Priority-based notification**: Chỉ gửi thông báo cho danger/warning

✅ **Fall velocity analysis**: Phân biệt té nhanh/chậm để xác định đột quỵ

✅ **Context-aware classification**: Dựa trên fall_type, duration, confidence
