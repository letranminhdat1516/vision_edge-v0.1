# Key Thresholds Summary (Quick Reference)

- **Mục đích**: Bảng tóm tắt nhanh tất cả ngưỡng/điều kiện quan trọng trong hệ thống để dễ tra cứu và tinh chỉnh.

---

## 1. Motion Detection (MOG2)

| Parameter             | Value | Module                 |
| --------------------- | ----- | ---------------------- |
| `motion_pixels`       | > 150 | `simple_processing.py` |
| Background subtractor | MOG2  | OpenCV                 |

---

## 2. Keyframe Detection

| Parameter          | Value               | Module                   |
| ------------------ | ------------------- | ------------------------ |
| `normalized_diff`  | > 0.01 (1% pixels)  | `SimpleKeyframeDetector` |
| History multiplier | > 1.5× mean(last 5) | `SimpleKeyframeDetector` |
| Gaussian blur      | 9×9 kernel          | `SimpleKeyframeDetector` |

**Logic**: `is_keyframe = (diff > 0.01) AND (diff > 1.5 × avg_recent)`

---

## 3. YOLOv8-Pose Person Selection

| Parameter              | Value                    | Module                  |
| ---------------------- | ------------------------ | ----------------------- |
| `confidence_threshold` | ≥ 0.5                    | `YOLOv8PoseEstimator`   |
| Min visible keypoints  | ≥ 5                      | `_validate_keypoints()` |
| Essential points       | ≥ 2/4 (shoulders + hips) | `_validate_keypoints()` |

---

## 4. Fall Detection

### 4.1 Primary Thresholds

| Parameter              | Value | Description                       |
| ---------------------- | ----- | --------------------------------- |
| `confidence_threshold` | 0.40  | Minimum để xác nhận fall          |
| `min_time_interval`    | 0.15s | Thời gian tối thiểu giữa 2 frames |
| `max_buffer_size`      | 5     | Số frames trong buffer            |

### 4.2 Movement Thresholds

| Condition        | Threshold               | Action                   |
| ---------------- | ----------------------- | ------------------------ |
| Standing up      | Δy > 300px + moving UP  | Reject                   |
| Small adjustment | Δy < 60px + moving DOWN | Reject (unless sideways) |
| Rapid fall       | Δy > 70px + moving DOWN | Analyze velocity         |
| Walking across   | Δx > 0.8 × Δy           | Reject                   |
| Depth movement   | size_change > 150%      | Reject                   |

### 4.3 Aspect Ratio Conditions

| Condition        | Values                                   | Meaning      |
| ---------------- | ---------------------------------------- | ------------ |
| Sideways fall    | Δx > 40, Δa > 1.2, a₂ > 1.4              | Té ngang     |
| Controlled lying | a₂ > 1.2, 0.85 ≤ Δa ≤ 1.15, v < 1500px/s | Nằm từ từ    |
| Already lying    | a₁ > 1.5                                 | Đã nằm sẵn   |
| Deep bending     | a₂ < 0.6                                 | Cúi sâu      |
| Definitely fall  | y/H ≥ 90% AND a₂ ≥ 1.4                   | Chắc chắn té |

### 4.4 Velocity Model

| Fall Type          | Velocity                    | Multiplier     |
| ------------------ | --------------------------- | -------------- |
| `fast_fall`        | > 400 px/s                  | 1.0×           |
| `moderate_fall`    | 150-400 px/s                | 1.1×           |
| `slow_collapse`    | < 150 px/s, duration ≥ 1.5s | 1.3× (stroke!) |
| Controlled descent | < 150 px/s, duration > 0.5s | Reject         |

### 4.5 Confidence Formula

```
conf = min(0.90, 0.50 + Δy/180) × severity_multiplier
Accept if: conf ≥ 0.50 AND (velocity ≥ 600 OR (y/H ≥ 0.90 AND a₂ ≥ 1.4))
```

### 4.6 Cooldowns

| Cooldown               | Duration | Purpose                |
| ---------------------- | -------- | ---------------------- |
| `danger_cooldown`      | 15s      | Giữa các DANGER events |
| `standing_up_cooldown` | 3s       | Sau khi đứng dậy       |

---

## 5. Seizure Predictor

| Parameter              | Value    | Description                |
| ---------------------- | -------- | -------------------------- |
| `smoothing_factor` (α) | 0.8      | EMA weight cho giá trị mới |
| `warning_threshold`    | ≥ 0.80   | Ngưỡng WARNING             |
| `alert_threshold`      | ≥ 0.90   | Ngưỡng CRITICAL            |
| `temporal_window`      | 5 frames | Cửa sổ phân tích           |

### Alert Logic

```
CRITICAL: smoothed ≥ 0.90 OR raw ≥ 1.00
WARNING: smoothed ≥ 0.80
NORMAL: otherwise

Enhanced: IF sustained_high AND trend=increasing → at least WARNING
```

### Trend Detection

| Slope   | Trend      |
| ------- | ---------- |
| > 0.01  | increasing |
| < -0.01 | decreasing |
| else    | stable     |

---

## 6. Pipeline Cooldowns

| Cooldown              | Duration | Scope                            |
| --------------------- | -------- | -------------------------------- |
| **Global**            | 45s      | Any event → block all new events |
| Fall-specific         | 10s      | Trong SimpleFallDetector         |
| Seizure-specific      | 30s      | Trong SeizurePredictor           |
| Normal-log throttle   | 10s      | Chỉ log NORMAL mỗi 10s           |
| DANGER → block NORMAL | 60s      | Sau DANGER, block NORMAL events  |

---

## 7. Event Severity Mapping

### Fall Detection

| Confidence | Severity | Mobile Status     |
| ---------- | -------- | ----------------- |
| ≥ 0.60     | high     | danger            |
| ≥ 0.40     | medium   | abnormal_behavior |
| < 0.40     | low      | normal            |

### Seizure Detection

| Confidence | Severity | Mobile Status     |
| ---------- | -------- | ----------------- |
| ≥ 0.50     | high     | danger            |
| ≥ 0.30     | medium   | abnormal_behavior |
| < 0.30     | low      | normal            |

---

## 8. Priority Levels

| Level | Severity | Status       | Action      |
| ----- | -------- | ------------ | ----------- |
| 4     | high     | active       | 🔴 CRITICAL |
| 3     | medium   | active       | 🟡 WARNING  |
| 2     | low      | active       | 🟢 INFO     |
| 2     | high     | acknowledged | Handling    |
| 1     | medium   | acknowledged | Handling    |
| 0     | any      | resolved     | Closed      |

**Gating Rule**: Create event only if `new_priority ≥ current_max_priority`

---

## 9. Keypoint Validation

| Check             | Threshold                            |
| ----------------- | ------------------------------------ |
| Total keypoints   | = 17                                 |
| Visible points    | ≥ 5 (conf > 0.3)                     |
| Essential visible | ≥ 2/4 (shoulders[5,6] + hips[11,12]) |
| Point confidence  | > 0.3 to be "visible"                |

---

## 10. Frame Buffer & Storage

| Parameter            | Value    | Purpose              |
| -------------------- | -------- | -------------------- |
| Pre-event buffer     | 5 frames | Capture before event |
| Snapshot count       | 5 images | Per event            |
| Max files per folder | 1000     | Auto-cleanup         |
| Keyframe history     | 50 → 30  | Memory management    |

---

## 11. Network & Database

| Parameter       | Value        | Notes                  |
| --------------- | ------------ | ---------------------- |
| PostgreSQL port | 5432         | Direct (LISTEN/NOTIFY) |
| Pooler port     | 6543         | PgBouncer (no LISTEN)  |
| Cache cleanup   | > 1000 items | Clear processed_events |
| Alarm duration  | 0 (infinite) | Until stop command     |

---

## Quick Tuning Guide

| Issue                    | Adjust                                 |
| ------------------------ | -------------------------------------- |
| Too many false falls     | ↑ `confidence_threshold` (0.40 → 0.50) |
| Missing real falls       | ↓ Δy threshold (70 → 60px)             |
| Sitting detected as fall | Tighten y/H (90% → 95%)                |
| Noisy keyframes          | ↑ `min_diff_threshold` (0.01 → 0.02)   |
| Seizure false positives  | ↑ `alert_threshold` (0.90 → 0.95)      |
| Event spam               | ↑ `GLOBAL_EVENT_COOLDOWN` (45s → 60s)  |
| Slow reaction            | ↓ cooldowns cautiously                 |

---

## Notes (VI)

Bảng tham chiếu nhanh tất cả ngưỡng quan trọng: motion (150px), keyframe (1% + 1.5× avg), YOLOv8-Pose (conf ≥ 0.5), fall detection (multi-condition + velocity model), seizure (EMA α=0.8, critical ≥ 0.90, warning ≥ 0.80), pipeline cooldowns (global 45s). Dùng để debug và tinh chỉnh hệ thống theo môi trường triển khai.
