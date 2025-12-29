# Combined Video Pipeline – Orchestration & Cooldowns

- **Module**: [src/service/advanced_healthcare_pipeline.py](src/service/advanced_healthcare_pipeline.py) → class `AdvancedHealthcarePipeline`
- **Mục đích**: Điều phối toàn bộ pipeline xử lý video, quản lý cooldowns, và tích hợp các components detection.

---

## 1. Constructor Dependencies

```python
def __init__(self,
    camera,              # Camera input source
    video_processor,     # Keyframe detection + object detection
    fall_detector,       # SimpleFallDetector instance
    seizure_detector,    # VSViG or similar model
    seizure_predictor,   # SeizurePredictor instance
    alerts_folder,       # Path to save alert images
    camera_id=None,      # UUID for database
    user_id=None         # UUID for database
):
```

### Initialized Services

```python
self.event_publisher = HealthcareEventPublisher(user_id, camera_id)
self.snapshot_service = get_snapshot_service()  # MinIO integration
```

---

## 2. Frame Buffer System

### Pre-Event Frame Buffer

```python
from collections import deque
self._frame_buffer = deque(maxlen=5)      # 5 frames gần nhất
self._frame_timestamps = deque(maxlen=5)   # Timestamps tương ứng
```

**Mục đích**: Khi detect event, có thể capture các frames TRƯỚC sự kiện để xem diễn biến.

### Usage Flow

```
Frame t-4  →  Frame t-3  →  Frame t-2  →  Frame t-1  →  Frame t (Event!)
   ↓            ↓            ↓            ↓            ↓
[buffer[0]] [buffer[1]] [buffer[2]] [buffer[3]] [buffer[4]]
                              ↓
                    Capture all 5 + current for snapshots
```

---

## 3. Cooldown System

### 3.1 Global Event Cooldown

```python
self._last_any_event_time = 0           # Timestamp event cuối cùng
self._GLOBAL_EVENT_COOLDOWN = 45.0      # 45 giây giữa BẤT KỲ event nào
self._active_event_id = None            # Event đang active
self._active_event_type = None          # Loại event
```

**Logic**: Chặn TẤT CẢ events mới khi đã có event đang xử lý → Tránh spam (fall + seizure + seizure trong vài giây).

### 3.2 Type-Specific Cooldowns (trong FallDetector/SeizurePredictor)

| Type    | Cooldown | Purpose                     |
| ------- | -------- | --------------------------- |
| Fall    | 10s      | Giữa các fall detections    |
| Seizure | 30s      | Giữa các seizure detections |

### 3.3 DANGER Block Cooldown

```python
self._last_danger_time = 0
self._DANGER_BLOCK_DURATION = 60.0  # 60s chặn NORMAL sau DANGER
```

**Logic**: Sau khi detect DANGER, block NORMAL events 60s (vì BLIP caption có thể sai khi người vẫn nằm).

### 3.4 Normal Log Throttle

```python
self._last_normal_log_time = 0
# Only log NORMAL every ~10 seconds
```

---

## 4. Cooldown Check Flow

```python
def _check_and_clear_global_cooldown():
    """Check if cooldown expired and clear active event"""

    if self._active_event_id is None:
        return False  # No cooldown, can create new event

    time_since_last = current_time - self._last_any_event_time

    if time_since_last >= self._GLOBAL_EVENT_COOLDOWN:  # >= 45s
        # Clear active event
        self._active_event_id = None
        self._active_event_type = None
        return False  # Can create new event
    else:
        return True  # Still in cooldown, block new events
```

### Cooldown Timeline

```
Event A (Fall)                    Event B (Seizure)
    │                                 │
    ▼                                 │
┌───────────────────────────────────┐ │
│     45s GLOBAL COOLDOWN           │ │
│  (All new events BLOCKED)         │ │
└───────────────────────────────────┘ │
                                      ▼
                              Can create Event B
```

---

## 5. Keyframe Gating

### Frame Processing Logic

```python
def process_frame(frame):
    # Step 1: Check if keyframe
    processing_result = video_processor.process_frame(frame)

    if not processing_result['processed']:  # Not a keyframe
        # Return simple result, skip heavy AI
        return {
            "detection_result": {
                'fall_detected': False,
                'seizure_detected': False,
                'alert_level': 'normal'
            }
        }

    # Step 2: KEYFRAME - Run full AI detection
    stats['keyframes_detected'] += 1
    persons = processing_result['person_detections']

    # Run fall detection, seizure detection, etc.
    detection_result = process_dual_detection(frame, persons)
```

**Performance Impact:**
| Metric | Without Keyframe Gate | With Keyframe Gate |
|--------|----------------------|-------------------|
| Frames analyzed | 100% | ~10-20% (only keyframes) |
| CPU usage | High | Low |
| Latency | Higher | Lower |

---

## 6. Event Creation Strategy

### Flow: Event First → Snapshots → Link

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Create Event in Database                                │
│  ┌─────────────────────────────────────────────┐                │
│  │ event_id = create_event_detection(...)      │                │
│  │ → Returns UUID for linking                  │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Capture Snapshots (5 images)                            │
│  ┌─────────────────────────────────────────────┐                │
│  │ Frames: [buffer[-3], buffer[-2], buffer[-1],│                │
│  │          current, current+overlay]          │                │
│  │ Upload to MinIO with event_id in metadata   │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Update Event with Snapshot ID                           │
│  ┌─────────────────────────────────────────────┐                │
│  │ update_event_snapshot(event_id, snapshot_id)│                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Why Event First?

1. **Guaranteed `event_id`**: Snapshots need event_id for linking
2. **Audit Trail**: Event exists even if snapshot upload fails
3. **Realtime Updates**: Event published to listeners immediately

---

## 7. Status Mapping

### Internal Status Levels

| Level     | Meaning    | Trigger                                  |
| --------- | ---------- | ---------------------------------------- |
| `danger`  | Emergency  | Fall/Seizure confirmed (high confidence) |
| `warning` | Caution    | Potential issue detected                 |
| `suspect` | Monitoring | Unusual pattern                          |
| `normal`  | Safe       | No issues                                |

### Mobile Status Mapping

```python
status_map = {
    'danger': 'danger',
    'warning': 'abnormal_behavior',
    'suspect': 'abnormal_behavior',
    'normal': 'normal'
}
```

---

## 8. Statistics Tracking

```python
self.stats = {
    # Basic
    'start_time': time.time(),
    'total_frames': 0,
    'frames_processed': 0,
    'keyframes_detected': 0,
    'persons_detected': 0,
    'fps': 0.0,

    # Fall detection
    'fall_detections': 0,
    'last_fall_time': None,

    # Seizure detection
    'seizure_detections': 0,
    'last_seizure_time': None,
    'seizure_warnings': 0,
    'pose_extraction_failures': 0,

    # Alerts
    'critical_alerts': 0,
    'total_alerts': 0,
    'last_alert_time': None,
    'alert_type': 'normal'
}
```

---

## 9. Detection History

```python
self.detection_history = {
    'fall_confidences': [],           # Recent fall confidences
    'seizure_confidences': [],        # Recent seizure confidences
    'motion_levels': [],              # Recent motion levels
    'max_history': 10,                # Keep last 10
    'fall_confirmation_frames': 0,    # Frames confirming fall
    'seizure_confirmation_frames': 0, # Frames confirming seizure
    'last_significant_motion': time.time()
}
```

---

## 10. Overall Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT: Camera Frame                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Check Global Cooldown                                           │
│  ┌──────────────────────────────────┐                           │
│  │ Active event + < 45s elapsed?    │──Yes──▶ Block new events  │
│  └──────────────────────────────────┘                           │
└──────────────┼──────────────────────────────────────────────────┘
               │ No/Cleared
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Keyframe Detection                                              │
│  ┌──────────────────────────────────┐                           │
│  │ Is keyframe?                     │──No──▶ Return simple      │
│  └──────────────────────────────────┘                           │
└──────────────┼──────────────────────────────────────────────────┘
               │ Yes
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Person Detection (YOLOv8)                                       │
│  Get bounding boxes for all persons                              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Fall Detection         │     │  Seizure Detection      │
│  (SimpleFallDetector)   │     │  (VSViG + Predictor)    │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Event Decision                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Fall detected (conf >= threshold) AND cooldown passed?   │   │
│  │ OR Seizure CRITICAL AND cooldown passed?                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│              │ Yes                    │ No                       │
└──────────────┼────────────────────────┼─────────────────────────┘
               ▼                        ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Create Event           │     │  Return Detection       │
│  Capture Snapshots      │     │  Result (no event)      │
│  Update Cooldowns       │     └─────────────────────────┘
│  Trigger Alarm          │
└─────────────────────────┘
```

---

## 11. Multi-Camera Support

```python
def process_frame(frame, other_cameras=None):
    """
    Args:
        frame: Current frame
        other_cameras: List of other camera dicts in same room
                      [{'camera_id': ..., 'frame': ...}, ...]
    """
    # When event detected, can capture from multiple angles
```

---

## Notes (VI)

Pipeline điều phối: keyframe gating để giảm tải → fall/seizure detection song song → cooldown checks → event creation → snapshot capture. Cooldown system: global 45s, fall 10s, seizure 30s, DANGER blocks NORMAL 60s. Event tạo trước rồi mới capture snapshots để đảm bảo có event_id cho linking.
