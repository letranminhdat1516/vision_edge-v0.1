# 📊 TỔNG HỢP TẤT CẢ CÔNG THỨC VÀ THUẬT TOÁN

> **Tài liệu này tổng hợp tất cả ~40 công thức toán học và thuật toán được sử dụng trong hệ thống Healthcare Monitoring.**
>
> **Cập nhật:** 12/01/2026

---

## 📋 MỤC LỤC

1. [Fall Detection (10 công thức)](#1️⃣-fall-detection---phát-hiện-té-ngã)
2. [Seizure Detection (10 công thức)](#2️⃣-seizure-detection---phát-hiện-co-giật)
3. [Seizure Predictor (4 công thức)](#3️⃣-seizure-predictor---dự-đoán-co-giật)
4. [Reliability Score (4 thành phần)](#4️⃣-reliability-score---điểm-độ-tin-cậy)
5. [Motion Detection (2 công thức)](#5️⃣-motion-detection---phát-hiện-chuyển-động)
6. [Keyframe Detection (2 công thức)](#6️⃣-keyframe-detection---phát-hiện-khung-hình-chính)
7. [Exercise Detection (4 công thức)](#7️⃣-exercise-detection---phát-hiện-bài-tập)
8. [Status Thresholds (2 bảng)](#8️⃣-status-thresholds---ngưỡng-trạng-thái)
9. [Bảng tổng hợp Thresholds](#📋-bảng-tổng-hợp-tất-cả-thresholds)

---

## 1️⃣ FALL DETECTION - Phát hiện té ngã

**File:** `src/fall_detection/simple_fall_detector.py`

### 1.1 Aspect Ratio (Tỷ lệ khung hình)

```python
# Line 220-230
aspect_ratio = width / height
```

**Mục đích:** Xác định tư thế người

- `aspect_ratio > 1.4` → Nằm ngang (lying)
- `aspect_ratio < 1.0` → Đứng thẳng (standing)
- `1.0 < aspect_ratio < 1.4` → Ngồi/cúi (sitting/bending)

---

### 1.2 Aspect Change (Thay đổi tỷ lệ)

```python
# Line 230
aspect_change = aspect_ratio2 / aspect_ratio1
```

**Mục đích:** Phát hiện thay đổi tư thế

- `aspect_change > 1.15` → Moderate fall threshold
- `aspect_change > 1.6` → Dynamic fall threshold (đứng → nằm)

---

### 1.3 Center Position (Vị trí trung tâm)

```python
# Line 232-235
center_x = (bbox[0] + bbox[2]) / 2  # (x1 + x2) / 2
center_y = (bbox[1] + bbox[3]) / 2  # (y1 + y2) / 2
```

**Mục đích:** Theo dõi vị trí người trong frame

---

### 1.4 Movement Calculation (Tính toán chuyển động)

```python
# Line 237-239
vertical_movement = abs(center2_y - center1_y)
horizontal_movement = abs(center2_x - center1_x)
movement_ratio = vertical_movement / (horizontal_movement + 1)
```

**Mục đích:** Phân biệt té ngã vs đi bộ

- `movement_ratio > 1.25` → Chuyển động xuống (có thể té)
- `movement_ratio < 0.8` → Chuyển động ngang (đi bộ)

---

### 1.5 Fall Velocity (Vận tốc té)

```python
# Line 702-703
fall_duration = current_time - fall_start_time
fall_velocity = total_fall_distance / fall_duration  # pixels/second
```

**Phân loại theo vận tốc:**
| Velocity | Type | Severity |
|----------|------|----------|
| > 400 px/s | Fast fall | Multiplier = 1.0 |
| 200-400 px/s | Moderate fall | Multiplier = 1.1 |
| < 200 px/s | Slow collapse | Multiplier = 1.3 |

---

### 1.6 BBox Size Change (Thay đổi kích thước)

```python
# Line 543-545
bbox_size1 = w1 * h1
bbox_size2 = w2 * h2
size_change_ratio = abs(bbox_size2 - bbox_size1) / (bbox_size1 + 1)
```

**Mục đích:** Phát hiện di chuyển theo chiều sâu (đến gần/xa camera)

---

### 1.7 Sideways Fall Confidence (Độ tin cậy té ngang)

```python
# Line 354-358
sideways_conf = 0.55  # Base confidence

# Bonuses
sideways_conf += min((aspect_change - 1.2) * 0.25, 0.15)  # Aspect change bonus (max +0.15)
sideways_conf += min(horizontal_movement / 150, 0.15)     # Horizontal bonus (max +0.15)
sideways_conf += min(vertical_movement / 80, 0.10)        # Vertical bonus (max +0.10)

# Cap
sideways_conf = min(0.90, sideways_conf)  # Max 0.90
```

**Công thức toán học:**
$$\text{Conf}_{sideways} = \min\left(0.90, 0.55 + \min\left(\frac{\Delta_{aspect} - 1.2}{4}, 0.15\right) + \min\left(\frac{h_{move}}{150}, 0.15\right) + \min\left(\frac{v_{move}}{80}, 0.10\right)\right)$$

---

### 1.8 Rapid Downward Fall Confidence

```python
# Line 758-760
downward_confidence = min(0.9, 0.50 + (vertical_movement / 180))
downward_confidence *= severity_multiplier  # {1.3, 1.1, 1.0}
downward_confidence = min(0.95, downward_confidence)
```

**Công thức:**
$$\text{Conf}_{rapid} = \min\left(0.95, \min\left(0.9, 0.5 + \frac{v_{move}}{180}\right) \times M_{severity}\right)$$

---

### 1.9 Moderate Fall Confidence

```python
# Line 817-818
confidence = min(0.85, 0.55 + (vertical_movement / 100) * 0.2 + (aspect_change - 1.25) * 0.15)
```

**Công thức:**
$$\text{Conf}_{moderate} = \min\left(0.85, 0.55 + \frac{v_{move}}{500} + 0.15 \times (\Delta_{aspect} - 1.25)\right)$$

---

### 1.10 Dynamic Fall Confidence

```python
# Line 843-844
confidence = min(0.9, 0.60 + (aspect_change - 1.7) * 0.35 + min(vertical_movement / 140, 0.28))
```

**Công thức:**
$$\text{Conf}_{dynamic} = \min\left(0.9, 0.60 + 0.35 \times (\Delta_{aspect} - 1.7) + \min\left(\frac{v_{move}}{140}, 0.28\right)\right)$$

---

## 2️⃣ SEIZURE DETECTION - Phát hiện co giật

**File:** `src/seizure_detection/vsvig_detector.py`

### 2.1 Velocity Calculation (Tính vận tốc keypoints)

```python
# Line 515-518
velocities = np.diff(coords, axis=0)  # Shape: (T-1, 15, 2)
vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=2))  # Shape: (T-1, 15)
```

**Công thức:**
$$v_i = \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$

---

### 2.2 Rhythm Regularity (Độ đều nhịp)

```python
# Line 527-532
velocity_over_time = np.mean(vel_magnitudes, axis=1)  # Mean velocity per frame
rhythm_variance = np.var(velocity_over_time)
rhythm_regularity = 1.0 / (1.0 + rhythm_variance / 100.0)
```

**Công thức:**
$$R_{rhythm} = \frac{1}{1 + \frac{\sigma^2_{velocity}}{100}}$$

**Mục đích:** Phân biệt:

- Exercise (high regularity, R > 0.7)
- Seizure (low regularity, R < 0.5)

---

### 2.3 Jerkiness (Độ giật)

```python
# Line 544-545
vel_diff = np.diff(vel_magnitudes, axis=0)  # Acceleration
jerkiness = np.mean(np.abs(vel_diff))
```

**Công thức:**
$$J = \frac{1}{N} \sum_{i=1}^{N} |v_{i+1} - v_i|$$

**Ngưỡng:** `jerkiness > 12` → Có thể là co giật

---

### 2.4 Velocity Score

```python
# Line 553-554
velocity_variance = np.var(vel_magnitudes, axis=0).mean()
velocity_score = np.tanh(velocity_variance / 100.0) if velocity_variance > 40 else 0.0
```

**Công thức:**
$$S_{velocity} = \begin{cases} \tanh\left(\frac{\sigma^2_v}{100}\right) & \text{if } \sigma^2_v > 40 \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.5 Acceleration Score

```python
# Line 557-560
accelerations = np.diff(velocities, axis=0)  # Second derivative
acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=2))
acceleration_peaks = np.max(acc_magnitudes, axis=0).mean()
acceleration_score = np.tanh(acceleration_peaks / 120.0) if acceleration_peaks > 50 else 0.0
```

**Công thức:**
$$S_{acc} = \begin{cases} \tanh\left(\frac{\max(|a|)}{120}\right) & \text{if } \max(|a|) > 50 \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.6 Frequency Score (Điểm tần số)

```python
# Line 563-569
direction_changes = 0
for joint in range(min(8, vel_magnitudes.shape[1])):
    joint_vel = vel_magnitudes[:, joint]
    changes = np.sum(np.diff(np.sign(joint_vel)) != 0)
    direction_changes += changes
frequency_score = np.tanh(direction_changes / 60.0) if direction_changes > 15 else 0.0
```

**Công thức:**
$$S_{freq} = \begin{cases} \tanh\left(\frac{N_{changes}}{60}\right) & \text{if } N_{changes} > 15 \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.7 Intensity Score (Điểm cường độ)

```python
# Line 572-574
total_movement = np.mean(vel_magnitudes)
intensity_score = np.tanh(total_movement / 30.0) if total_movement > 8 else 0.0
```

**Công thức:**
$$S_{intensity} = \begin{cases} \tanh\left(\frac{\bar{v}}{30}\right) & \text{if } \bar{v} > 8 \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.8 Spike Score (Điểm đột biến)

```python
# Line 577-578
movement_spikes = np.max(vel_magnitudes, axis=0).mean()
spike_score = np.tanh(movement_spikes / 60.0) if movement_spikes > 20 else 0.0
```

**Công thức:**
$$S_{spike} = \begin{cases} \tanh\left(\frac{\max(v)}{60}\right) & \text{if } \max(v) > 20 \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.9 ⭐ Final Seizure Confidence (Công thức cuối cùng)

```python
# Line 594-600
seizure_confidence = (
    0.25 * velocity_score +      # 25%
    0.25 * acceleration_score +  # 25%
    0.20 * frequency_score +     # 20%
    0.15 * intensity_score +     # 15%
    0.15 * spike_score           # 15%
)
```

**Công thức chính:**
$$\boxed{C_{seizure} = 0.25 \cdot S_{vel} + 0.25 \cdot S_{acc} + 0.20 \cdot S_{freq} + 0.15 \cdot S_{int} + 0.15 \cdot S_{spike}}$$

---

### 2.10 Aspect Ratio from Keypoints

```python
# Line 742-759
valid_kpts = keypoints[keypoints[:, 2] > 0.3]  # Filter low confidence
x_coords, y_coords = valid_kpts[:, 0], valid_kpts[:, 1]
width = np.max(x_coords) - np.min(x_coords)
height = np.max(y_coords) - np.min(y_coords)
aspect_ratio = width / height
```

**Mục đích:** Xác định tư thế từ pose estimation

- `aspect_ratio > 1.5` → Nằm (lying)
- `aspect_ratio < 0.8` → Đứng (standing)

---

## 3️⃣ SEIZURE PREDICTOR - Dự đoán co giật

**File:** `src/seizure_detection/seizure_predictor.py`

### 3.1 Exponential Smoothing (Làm mượt theo cấp số nhân)

```python
# Line 77-82
alpha = 0.8  # smoothing_factor
smoothed = alpha * current_value + (1 - alpha) * previous_smoothed
```

**Công thức:**
$$S_t = \alpha \cdot x_t + (1 - \alpha) \cdot S_{t-1}$$

Trong đó:

- $\alpha = 0.8$ (trọng số giá trị mới)
- $x_t$ = giá trị hiện tại
- $S_{t-1}$ = giá trị đã làm mượt trước đó

---

### 3.2 Trend Calculation (Tính xu hướng)

```python
# Line 110-116
x = np.arange(len(history))
trend_slope = np.polyfit(x, history, 1)[0]  # Linear regression

if trend_slope > 0.01:
    trend = 'increasing'
elif trend_slope < -0.01:
    trend = 'decreasing'
else:
    trend = 'stable'
```

**Công thức Linear Regression:**
$$slope = \frac{N \sum xy - \sum x \sum y}{N \sum x^2 - (\sum x)^2}$$

---

### 3.3 Volatility (Độ biến động)

```python
# Line 119
volatility = float(np.std(history))
```

**Công thức:**
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$

---

### 3.4 Sustained High Detection

```python
# Line 124-126
recent_window = history[-10:] if len(history) >= 10 else history
sustained_high = bool(np.mean(recent_window) > 0.80)  # warning_threshold
```

**Mục đích:** Phát hiện confidence cao liên tục trong 10 frames gần nhất

---

## 4️⃣ RELIABILITY SCORE - Điểm độ tin cậy

**File:** `src/service/postgresql_healthcare_service.py` (Line 525-585)

### ⭐ Công thức tổng hợp:

$$\boxed{\text{Reliability} = \min\left(\max\left(S_{base} + S_{severity} + S_{quality} + S_{context}, 0\right), 1\right)}$$

---

### 4.1 Base Score (40%)

```python
base_score = confidence * 0.4
```

$$S_{base} = C \times 0.4$$

---

### 4.2 Event Severity Score (30%)

```python
event_severity = {
    'fall': 0.30,
    'abnormal_behavior': 0.28,
    'seizure': 0.28,
    'manual_emergency': 0.30,
    'sleep': 0.05,
    'normal_activity': 0.02
}
severity_score = event_severity.get(event_type, 0.15)
```

| Event Type        | Severity Score |
| ----------------- | -------------- |
| fall              | 0.30           |
| abnormal_behavior | 0.28           |
| seizure           | 0.28           |
| manual_emergency  | 0.30           |
| sleep             | 0.05           |
| normal_activity   | 0.02           |
| default           | 0.15           |

---

### 4.3 Detection Quality Score (15%)

```python
quality_score = 0.10  # Base: has bounding box
quality_score += 0.03  # Bonus: >= 2 detections
quality_score += 0.02  # Bonus: has keypoints
```

| Condition        | Score    |
| ---------------- | -------- |
| Has bounding box | +0.10    |
| ≥2 detections    | +0.03    |
| Has keypoints    | +0.02    |
| **Max**          | **0.15** |

---

### 4.4 Context Score (15%)

```python
context_scores = {
    'critical': 0.15,
    'high': 0.12,
    'warning': 0.08
}
context_score = context_scores.get(alert_level, 0.05)
context_score += 0.03  # Bonus: consecutive_detections >= 3
```

| Alert Level    | Score |
| -------------- | ----- |
| critical       | 0.15  |
| high           | 0.12  |
| warning        | 0.08  |
| default        | 0.05  |
| consecutive ≥3 | +0.03 |

---

### 📊 Ví dụ tính Reliability Score:

**Input:**

- `confidence = 0.75`
- `event_type = 'fall'`
- `has_bbox = True`
- `detection_count = 2`
- `has_keypoints = True`
- `alert_level = 'critical'`
- `consecutive_detections = 4`

**Calculation:**

```
Base Score:     0.75 × 0.4 = 0.30
Severity Score: 0.30 (fall)
Quality Score:  0.10 + 0.03 + 0.02 = 0.15
Context Score:  0.15 + 0.03 = 0.18

Total = 0.30 + 0.30 + 0.15 + 0.18 = 0.93
Final = min(max(0.93, 0), 1) = 0.93
```

---

## 5️⃣ MOTION DETECTION - Phát hiện chuyển động

**File:** `src/video_processing/simple_processing.py`

### 5.1 MOG2 Background Subtraction

```python
# Line 35-38
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200,        # Số frames để học background
    varThreshold=32,    # Ngưỡng variance
    detectShadows=True  # Phát hiện bóng
)
```

**Thuật toán:** Gaussian Mixture Model (GMM)

- Mỗi pixel được mô hình hóa bằng K Gaussian distributions
- Foreground = pixel không khớp với bất kỳ Gaussian nào

---

### 5.2 Motion Pixels Count

```python
# Line 58-61
fg_mask = bg_subtractor.apply(frame)
motion_pixels = cv2.countNonZero(fg_mask)
motion_detected = motion_pixels > 150  # threshold
```

**Logic:**

- Đếm số pixel foreground (motion)
- Nếu > 150 pixels → có chuyển động đáng kể

---

## 6️⃣ KEYFRAME DETECTION - Phát hiện khung hình chính

**File:** `src/video_processing/simple_processing.py`

### 6.1 Frame Difference

```python
# Line 134-139
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
diff = cv2.subtract(blur_gray, last_frame)
diff_magnitude = cv2.countNonZero(diff)
normalized_diff = diff_magnitude / (frame.shape[0] * frame.shape[1])
```

**Công thức:**
$$D_{norm} = \frac{\text{countNonZero}(|F_t - F_{t-1}|)}{W \times H}$$

---

### 6.2 Adaptive Keyframe Selection

```python
# Line 147-151
is_keyframe = normalized_diff > 0.01  # min_diff_threshold

if len(diff_history) >= 5:
    recent_avg = np.mean(diff_history[-5:])
    is_keyframe = is_keyframe and (normalized_diff > recent_avg * 1.5)
```

**Logic:**

- Frame là keyframe nếu:
  1. `normalized_diff > 0.01` (ngưỡng tối thiểu)
  2. `normalized_diff > 1.5 × average(5 frames gần nhất)` (adaptive)

---

## 7️⃣ EXERCISE DETECTION - Phát hiện bài tập

**File:** `src/fall_detection/simple_fall_detector.py` (Line 1001-1033)

### 7.1 Amplitude & Direction Changes

```python
# Line 1001-1005
y_array = np.array(shoulder_y_history)
diffs = np.diff(y_array)
sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
amplitude = np.max(y_array) - np.min(y_array)
```

**Mục đích:** Đếm số lần đổi hướng chuyển động

---

### 7.2 Jerkiness (Độ giật của exercise)

```python
# Line 1008-1012
acceleration = np.diff(diffs)  # Second derivative
jerkiness = np.mean(np.abs(acceleration))
```

**Công thức:**
$$J_{exercise} = \frac{1}{N} \sum |a_i|$$

---

### 7.3 Rhythm Regularity (CV method)

```python
# Line 1015-1019
abs_diffs = np.abs(diffs)
rhythm_cv = np.std(abs_diffs) / np.mean(abs_diffs)  # Coefficient of Variation
rhythm_regularity = 1.0 / (1.0 + rhythm_cv)
```

**Công thức:**
$$R = \frac{1}{1 + CV} = \frac{1}{1 + \frac{\sigma}{\mu}}$$

- `R > 0.6` → Chuyển động đều (exercise)
- `R < 0.4` → Chuyển động không đều (có thể seizure/fall)

---

### 7.4 Autocorrelation (Tự tương quan)

```python
# Line 1023-1033
centered = y_array - np.mean(y_array)
autocorr = np.correlate(centered, centered, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr = autocorr / (autocorr[0] + 1e-6)  # Normalize
has_repetitive_pattern = np.max(autocorr[2:]) > 0.5
```

**Mục đích:** Phát hiện pattern lặp lại (push-up, squat, etc.)

**Công thức Autocorrelation:**
$$R(\tau) = \frac{\sum_{t=0}^{N-\tau-1} (x_t - \bar{x})(x_{t+\tau} - \bar{x})}{\sum_{t=0}^{N-1} (x_t - \bar{x})^2}$$

---

## 8️⃣ STATUS THRESHOLDS - Ngưỡng trạng thái

**File:** `src/service/postgresql_healthcare_service.py` (Line 445-480)

### 8.1 Fall Status Thresholds

```python
if event_type == 'fall':
    if confidence >= 0.50: return 'danger'
    elif confidence >= 0.40: return 'warning'
    elif confidence >= 0.20: return 'suspect'
    else: return 'normal'
```

| Confidence | Status     | Action        |
| ---------- | ---------- | ------------- |
| ≥ 0.50     | 🔴 DANGER  | Báo động ngay |
| ≥ 0.40     | 🟠 WARNING | Cảnh báo      |
| ≥ 0.20     | 🟡 SUSPECT | Theo dõi      |
| < 0.20     | 🟢 NORMAL  | Bình thường   |

---

### 8.2 Seizure Status Thresholds

```python
if event_type in ['abnormal_behavior', 'seizure']:
    if confidence >= 0.50: return 'danger'
    elif confidence >= 0.30: return 'warning'
    elif confidence >= 0.15: return 'suspect'
    else: return 'normal'
```

| Confidence | Status     | Action        |
| ---------- | ---------- | ------------- |
| ≥ 0.50     | 🔴 DANGER  | Báo động ngay |
| ≥ 0.30     | 🟠 WARNING | Cảnh báo      |
| ≥ 0.15     | 🟡 SUSPECT | Theo dõi      |
| < 0.15     | 🟢 NORMAL  | Bình thường   |

---

## 📋 BẢNG TỔNG HỢP TẤT CẢ THRESHOLDS

| Category               | Parameter                    | Value     | File:Line                    |
| ---------------------- | ---------------------------- | --------- | ---------------------------- |
| **Fall Detection**     |                              |           |                              |
|                        | confidence_threshold         | 0.40      | simple_fall_detector.py:18   |
|                        | min_time_interval            | 0.15s     | simple_fall_detector.py:27   |
|                        | vertical_movement (rapid)    | >70px     | simple_fall_detector.py:511  |
|                        | vertical_movement (moderate) | >50px     | simple_fall_detector.py:270  |
|                        | aspect_change (moderate)     | >1.15     | simple_fall_detector.py:270  |
|                        | aspect_change (dynamic)      | >1.6      | simple_fall_detector.py:832  |
|                        | MIN_FALL_VELOCITY            | 150 px/s  | simple_fall_detector.py:700  |
|                        | danger_cooldown              | 15s       | simple_fall_detector.py:37   |
|                        | standing_up_cooldown         | 3s        | simple_fall_detector.py:41   |
| **Seizure Detection**  |                              |           |                              |
|                        | confidence_threshold         | 0.50      | vsvig_detector.py:42         |
|                        | temporal_window              | 10 frames | vsvig_detector.py:80         |
|                        | seizure_cooldown             | 45s       | vsvig_detector.py:88         |
|                        | lying_ratio                  | ≥70%      | vsvig_detector.py:287        |
|                        | jerkiness_threshold          | >12       | vsvig_detector.py:548        |
|                        | velocity_variance_min        | >40       | vsvig_detector.py:553        |
|                        | acceleration_peaks_min       | >50       | vsvig_detector.py:560        |
|                        | direction_changes_min        | >15       | vsvig_detector.py:569        |
| **Seizure Predictor**  |                              |           |                              |
|                        | smoothing_factor (α)         | 0.8       | seizure_predictor.py:22      |
|                        | alert_threshold              | 0.90      | seizure_predictor.py:23      |
|                        | warning_threshold            | 0.80      | seizure_predictor.py:24      |
|                        | trend_slope (increasing)     | >0.01     | seizure_predictor.py:112     |
| **Motion Detection**   |                              |           |                              |
|                        | motion_pixel_threshold       | 150       | simple_processing.py:29      |
|                        | MOG2 history                 | 200       | simple_processing.py:35      |
|                        | MOG2 varThreshold            | 32        | simple_processing.py:35      |
| **Keyframe Detection** |                              |           |                              |
|                        | min_diff_threshold           | 0.01      | simple_processing.py:92      |
|                        | adaptive_multiplier          | 1.5x      | simple_processing.py:150     |
| **Push-up Detection**  |                              |           |                              |
|                        | direction_changes            | ≥20       | simple_fall_detector.py:1041 |
|                        | aspect_ratio (horizontal)    | >1.3      | simple_fall_detector.py:1045 |
|                        | time_window                  | 8s        | simple_fall_detector.py:1038 |
|                        | large_movement_override      | >150px    | simple_fall_detector.py:343  |

---

## 📊 TỔNG KẾT

| Nhóm               | Số công thức |
| ------------------ | ------------ |
| Fall Detection     | 10           |
| Seizure Detection  | 10           |
| Seizure Predictor  | 4            |
| Reliability Score  | 4            |
| Motion Detection   | 2            |
| Keyframe Detection | 2            |
| Exercise Detection | 4            |
| Status Thresholds  | 2            |
| **TỔNG CỘNG**      | **~38-40**   |

---

## 📚 TÀI LIỆU THAM KHẢO

- [reliability_score_calculation.md](reliability_score_calculation.md) - Chi tiết tính điểm độ tin cậy
- [FALL_THRESHOLDS_SUMMARY.md](FALL_THRESHOLDS_SUMMARY.md) - Ngưỡng phát hiện té ngã
- [SEIZURE_THRESHOLDS_SUMMARY.md](SEIZURE_THRESHOLDS_SUMMARY.md) - Ngưỡng phát hiện co giật
- [PROJECT_LOGIC_AND_ALGORITHMS.md](PROJECT_LOGIC_AND_ALGORITHMS.md) - Logic tổng quan

---

> **Ghi chú:** Tài liệu này được tạo tự động và có thể cần cập nhật khi code thay đổi.
