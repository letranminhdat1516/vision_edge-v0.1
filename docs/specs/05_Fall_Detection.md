# Fall Detection – Heuristics + Velocity Model

- **Module**: [src/fall_detection/simple_fall_detector.py](src/fall_detection/simple_fall_detector.py) → class `SimpleFallDetector`
- **Mục đích**: Phát hiện té ngã real-time dựa trên phân tích bbox geometry và velocity, không cần AI model nặng.

---

## 1. Tham số khởi tạo (Constructor Parameters)

| Parameter              | Type  | Default | Mô tả                             |
| ---------------------- | ----- | ------- | --------------------------------- |
| `confidence_threshold` | float | 0.40    | Ngưỡng tối thiểu để xác nhận fall |

### Internal State Variables

```python
# Frame tracking
self.min_time_interval = 0.15        # Khoảng thời gian tối thiểu giữa 2 frames
self.frame_buffer = []               # Buffer lưu frames gần đây
self.max_buffer_size = 5             # Tối đa 5 frames trong buffer

# Velocity tracking
self.fall_start_time = None          # Thời điểm bắt đầu phát hiện té
self.fall_start_position = None      # Vị trí Y ban đầu
self.fall_velocity_history = []      # Lịch sử vận tốc

# Cooldowns
self.last_danger_fall_time = 0       # Timestamp fall DANGER gần nhất
self.danger_cooldown = 15            # 15s cooldown giữa các DANGER events
self.last_standing_up_time = 0       # Timestamp đứng dậy gần nhất
self.standing_up_cooldown = 3        # 3s cooldown sau khi đứng dậy

# Repeated sitting pattern
self.sitting_events = []             # [(timestamp, position_y), ...]
self.sitting_pattern_window = 10     # 10s window
self.sitting_pattern_threshold = 3   # 3 lần ngồi = exercise
```

---

## 2. Preprocessing: Safe Bbox Conversion

```python
def _safe_bbox_conversion(bbox) -> Optional[List[float]]:
    """Convert bbox to [x1, y1, x2, y2] format safely"""
    # Validation:
    # 1. Must have 4 elements
    # 2. All values must be finite floats
    # 3. x2 > x1 and y2 > y1 (positive size)
    # 4. All coordinates >= 0
```

### Bbox Features Extracted

```python
# Dimensions
w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]  # Frame 1
w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]  # Frame 2

# Aspect ratios
aspect_ratio1 = w1 / h1  # <1 = standing, >1 = lying
aspect_ratio2 = w2 / h2
aspect_change = aspect_ratio2 / aspect_ratio1

# Center positions
center1_x, center1_y = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
center2_x, center2_y = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2

# Movement
vertical_movement = abs(center2_y - center1_y)
horizontal_movement = abs(center2_x - center1_x)
```

---

## 3. Detection Strategy Pipeline

### Strategy Priority Order

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIORITY 1: Standing Up Detection (Filter)                      │
│  ↓ Pass                                                          │
│  PRIORITY 2: Small Posture Adjustment (Filter)                   │
│  ↓ Pass (with Sideways Fall Bypass)                              │
│  PRIORITY 3: Sideways Fall Detection                             │
│  ↓ Not Sideways                                                  │
│  PRIORITY 4: Slow Lying Down (Filter)                            │
│  ↓ Pass                                                          │
│  STRATEGY 0: Rapid Downward Movement (Main Detection)            │
│  ↓ Not Detected                                                  │
│  STRATEGY 0.5: Moderate Fall                                     │
│  ↓ Not Detected                                                  │
│  STRATEGY 1: Dynamic Fall (Aspect Change)                        │
│  ↓ Not Detected                                                  │
│  STRATEGY 2: Static Lying (Already on floor)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Filter Details

### Filter 1: Standing Up Detection

```python
# Điều kiện: Di chuyển LÊN (vertical > 300px và center2_y < center1_y)
is_moving_upward = center2_y < center1_y
has_large_upward_movement = vertical_movement > 300 and is_moving_upward

if has_large_upward_movement:
    # → Reject as "standing-up"
    # → Start standing_up_cooldown (3s)
```

**Logic**: Khi người đứng dậy từ sàn, center_y GIẢM (đi lên trên màn hình). Reject để tránh false positive.

### Filter 2: Small Posture Adjustment

```python
# Điều kiện: Vertical nhỏ (<60px) và di chuyển xuống
has_small_downward_movement = vertical_movement < 60 and center2_y > center1_y

# BYPASS nếu là sideways fall pattern
is_sideways_fall_pattern = (
    horizontal_movement > 40 and
    aspect_change > 1.2 and
    aspect_ratio2 > 1.4
)

if has_small_downward_movement and not is_sideways_fall_pattern:
    # → Reject as "posture-adjustment"
```

### Filter 3: Slow Lying Down (Controlled Descent)

```python
# Điều kiện cho controlled descent:
is_lying_down_pattern = (
    final_position_ratio > 0.90 and    # Gần sàn
    final_aspect_ratio > 1.2 and       # Nằm ngang
    0.85 <= aspect_change <= 1.15 and  # Aspect ổn định
    vertical_movement < 600 and        # Không quá nhanh
    vertical_velocity < 1500           # px/s - chậm
)
```

### Filter 4: Walking Across

```python
# Reject nếu horizontal > 80% vertical (đi ngang)
if horizontal_movement > vertical_movement * 0.8 and vertical_movement <= 150:
    # → Reject as "walking-across"
```

### Filter 5: Depth Movement

```python
# Phát hiện di chuyển ra xa/gần camera (bbox size thay đổi lớn)
size_change_ratio = abs(bbox_size2 - bbox_size1) / (bbox_size1 + 1)

if size_change_ratio > 1.50 and 150 < vertical_movement < 400:
    # → Reject as "depth-movement"
```

### Filter 6: Already Lying

```python
# Người đã nằm sẵn (aspect_ratio1 > 1.5)
if aspect_ratio1 > 1.5 and vertical_movement <= 250:
    # → Reject as "already-lying"
```

### Filter 7: Sitting/Squatting

```python
# Ngồi xuống (không nằm hẳn xuống sàn)
is_definitely_falling = (
    final_position_ratio >= 0.90 and  # Gần sàn
    final_aspect_ratio >= 1.4          # Nằm ngang thực sự
)

if not is_definitely_falling:
    # → Reject as "sitting-down"
```

### Filter 8: Repeated Sitting (Exercise)

```python
# 3+ lần ngồi xuống trong 10s = tập squat
if len(sitting_events) >= 3:
    # → Reject as "exercise-squat"
```

---

## 5. Velocity Model (Stroke Detection)

### Fall Duration & Velocity Calculation

```python
fall_duration = current_time - fall_start_time
total_fall_distance = center2_y - fall_start_position
fall_velocity = total_fall_distance / fall_duration  # px/s
```

### Fall Type Classification

| Fall Type       | Velocity     | Duration | Severity | Description          |
| --------------- | ------------ | -------- | -------- | -------------------- |
| `fast_fall`     | > 400 px/s   | Any      | 1.0×     | Té nhanh bình thường |
| `moderate_fall` | 150-400 px/s | Any      | 1.1×     | Té vừa               |
| `slow_collapse` | < 150 px/s   | ≥ 1.5s   | 1.3×     | Đột quỵ/yếu sức ⚠️   |

### Controlled Descent Rejection

```python
MIN_FALL_VELOCITY = 150  # px/s

if fall_velocity < MIN_FALL_VELOCITY and fall_duration > 0.5:
    # → Reject as "controlled-descent"
    # Người đang TỰ NẰM XUỐNG, không phải té
```

---

## 6. Confidence Calculation

### Rapid Fall Confidence

```python
# Base confidence từ vertical movement
conf = min(0.90, 0.50 + (vertical_movement / 180))

# Adjust theo fall type
conf *= severity_multiplier  # 1.0, 1.1, hoặc 1.3

# Cap
conf = min(0.95, conf)
```

### Sideways Fall Confidence

```python
sideways_conf = 0.55  # Base
sideways_conf += min((aspect_change - 1.2) * 0.25, 0.15)  # Aspect bonus
sideways_conf += min(horizontal_movement / 150, 0.15)     # Horizontal bonus
sideways_conf += min(vertical_movement / 80, 0.10)        # Vertical bonus
sideways_conf = min(0.90, sideways_conf)
```

### Final Acceptance Criteria

```python
# RAPID FALL: Chấp nhận khi:
is_definitely_rapid_fall = (
    fall_velocity >= 600 or  # Quá nhanh để là ngồi
    (final_position_ratio >= 0.90 and final_aspect_ratio >= 1.4)  # Nằm sàn
)
```

---

## 7. Cooldown System

| Cooldown               | Duration | Purpose                            |
| ---------------------- | -------- | ---------------------------------- |
| `danger_cooldown`      | 15s      | Tránh spam DANGER events liên tiếp |
| `standing_up_cooldown` | 3s       | Block detection sau khi đứng dậy   |

### Cooldown Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  FALL DETECTED (DANGER)                                          │
│       │                                                          │
│       ▼                                                          │
│  Set last_danger_fall_time = now()                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────┐                │
│  │  Next 15 seconds:                           │                │
│  │  - All new fall detections REJECTED         │                │
│  │  - category: "danger-cooldown"              │                │
│  └─────────────────────────────────────────────┘                │
│       │                                                          │
│       ▼                                                          │
│  After 15s: Ready to detect new falls                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Output Format

```python
{
    'fall_detected': True,
    'confidence': 0.75,
    'angle': 60.0,              # Ước tính góc té (degrees)
    'category': 'fall',         # 'fall' | 'no-fall' | filter type
    'method': 'rapid_downward', # Detection method used
    'fall_type': 'fast_fall',   # 'fast_fall' | 'moderate_fall' | 'slow_collapse' | 'sideways_fall'
    'fall_duration': 0.35,      # Thời gian té (seconds)
    'fall_velocity': 450.0,     # Vận tốc té (px/s)
    'alert_level': 'DANGER',    # For sideways falls
    'processing_time': 0.012    # Processing time (seconds)
}
```

---

## 9. Flowchart Tổng Quan

```
┌─────────────────────────────────────────────────────────────────┐
│                 INPUT: frame, timestamp, person_bbox             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Safe Bbox Conversion                                            │
│  Add to frame_buffer (max 5 frames)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Check: buffer >= 2 frames AND time_interval >= 0.15s ?          │
│              │ No                                                │
│              ▼                                                   │
│         Return no-fall                                           │
└──────────────┼──────────────────────────────────────────────────┘
               │ Yes
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Calculate Movement Features                                     │
│  - aspect_ratio1, aspect_ratio2, aspect_change                   │
│  - center1, center2, vertical_movement, horizontal_movement      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FILTER CHAIN (see Strategy Pipeline)                            │
│  Each filter can REJECT with specific category                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DETECTION STRATEGIES (0, 0.5, 1, 2)                             │
│  Calculate confidence, check thresholds                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         FALL DETECTED   NO DETECTION    FILTERED
         (update cooldown)               (return category)
```

---

## 10. Rejection Categories

| Category               | Reason                     | Method                          |
| ---------------------- | -------------------------- | ------------------------------- |
| `standing-up`          | Người đứng dậy             | `standing_up_filtered`          |
| `posture-adjustment`   | Điều chỉnh tư thế nhỏ      | `small_movement_filtered`       |
| `lying-down`           | Nằm xuống từ từ            | `controlled_descent_filtered`   |
| `walking-across`       | Đi ngang qua camera        | `horizontal_movement_filtered`  |
| `depth-movement`       | Di chuyển ra xa/gần camera | `depth_movement_filtered`       |
| `already-lying`        | Đã nằm sẵn                 | `already_lying_filtered`        |
| `sitting-down`         | Ngồi xuống                 | `sitting_filtered`              |
| `exercise-squat`       | Tập squat                  | `repeated_sitting_filtered`     |
| `bending-normal`       | Cúi người sâu              | `bending_filtered`              |
| `danger-cooldown`      | Trong cooldown             | `danger_cooldown_filtered`      |
| `standing-up-cooldown` | Sau khi đứng dậy           | `standing_up_cooldown_filtered` |
| `controlled-descent`   | Velocity quá chậm          | `controlled_descent_filtered`   |

---

## Notes (VI)

Kết hợp phân tích hình học bbox (aspect ratio, center movement) với velocity model để phát hiện té ngã. Hệ thống filter đa tầng loại bỏ: đứng dậy, cúi người, đi ngang, di chuyển chiều sâu, ngồi xuống. Cooldown 15s giữa các DANGER events, 3s sau khi đứng dậy. Slow collapse (velocity thấp, duration dài) được đánh dấu severity cao hơn vì có thể là đột quỵ.
