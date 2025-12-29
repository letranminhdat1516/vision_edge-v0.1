# Event Publishing – Severity Mapping, Priority, Captions

- **Module**: [src/service/emergency_notification_dispatcher.py](src/service/emergency_notification_dispatcher.py) → class `HealthcareEventPublisher`
- **Mục đích**: Publish healthcare events với priority-based alert system, tích hợp AI caption generation.

---

## 1. Constructor

```python
def __init__(self,
    default_user_id: Optional[str] = None,
    default_camera_id: Optional[str] = None,
    default_room_id: Optional[str] = None
):
    # Load configuration
    self.config = config_loader.load_system_config()
    self.detection_settings = config_loader.load_detection_settings()

    # PostgreSQL service
    self.postgresql_service = realtime_service
```

---

## 2. Severity Mapping

### Confidence → Severity

```python
def _map_confidence_to_severity(confidence: float, event_type: str) -> str:
    # Load thresholds from config, fallback to defaults
    thresholds = detection_config.get('severity_mapping', {})
```

### Default Thresholds

| Event Type  | High (≥) | Medium (≥) | Low (<) |
| ----------- | -------- | ---------- | ------- |
| **Fall**    | 0.60     | 0.40       | 0.40    |
| **Seizure** | 0.50     | 0.30       | 0.30    |

### Mapping Logic

```python
if confidence >= thresholds['high']:     # Fall: ≥0.60, Seizure: ≥0.50
    return 'high'
elif confidence >= thresholds['medium']: # Fall: ≥0.40, Seizure: ≥0.30
    return 'medium'
else:
    return 'low'
```

---

## 3. Mobile Status Mapping

```python
def _map_status_for_mobile(severity: str) -> str:
    severity_to_mobile = {
        'high': 'danger',
        'medium': 'abnormal_behavior',
        'low': 'normal'
    }
    return severity_to_mobile.get(severity, 'normal')
```

### Mapping Table

| DB Severity | Mobile Status       | UI Color  | Action               |
| ----------- | ------------------- | --------- | -------------------- |
| `high`      | `danger`            | 🔴 Red    | Immediate alarm      |
| `medium`    | `abnormal_behavior` | 🟡 Yellow | Warning notification |
| `low`       | `normal`            | 🟢 Green  | Log only             |

---

## 4. Priority Level System

### Priority Calculation

```python
def _calculate_priority_level(severity: str, alert_status: str) -> int:
    base_priority = {
        'high': 4,
        'medium': 3,
        'low': 2
    }.get(severity, 1)

    # Reduce for acknowledged/resolved
    if alert_status == 'acknowledged':
        return max(1, base_priority - 2)  # 4→2, 3→1, 2→1
    elif alert_status == 'resolved':
        return 0

    return base_priority
```

### Priority Levels

| Level | Severity | Status       | Meaning                                 |
| ----- | -------- | ------------ | --------------------------------------- |
| 4     | high     | active       | 🔴 CRITICAL - Needs immediate attention |
| 3     | medium   | active       | 🟡 WARNING - Monitor closely            |
| 2     | low      | active       | 🟢 INFO - Logged for review             |
| 2     | high     | acknowledged | 👁️ Being handled                        |
| 1     | medium   | acknowledged | 👁️ Being handled                        |
| 0     | any      | resolved     | ✅ Closed                               |

---

## 5. Alert Creation Gating

### Decision Logic

```python
def _should_create_alert(confidence: float, event_type: str, user_id: str) -> Tuple[bool, str]:
    # Calculate new event priority
    severity = _map_confidence_to_severity(confidence, event_type)
    new_priority = _calculate_priority_level(severity, 'active')

    # Get current highest priority
    highest_alert = _get_highest_priority_alert(user_id)

    if highest_alert:
        current_max = highest_alert.get('priority_level', 0)
        # Create only if priority >= current max
        should_create = new_priority >= current_max
    else:
        # No existing alerts → create if not low priority
        should_create = new_priority > 2

    return should_create, severity
```

### Gating Flow

```
New Event (Fall, conf=0.65)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Calculate: severity='high', priority=4                          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Check existing alerts for user                                  │
│  ┌──────────────────────────────────┐                           │
│  │ Existing max priority = 3        │                           │
│  └──────────────────────────────────┘                           │
│              │                                                   │
│              ▼                                                   │
│  new_priority (4) >= current_max (3) ? → YES → Create alert     │
└─────────────────────────────────────────────────────────────────┘
```

### Why Priority Gating?

- **Prevent Alert Fatigue**: Don't create low priority alerts if high priority exists
- **Focus Attention**: Users see most important alerts first
- **Reduce Noise**: Skip redundant lower-priority alerts

---

## 6. Intelligent Caption Generation

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Alert Image Path                                         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  BLIP Model (Image → English Caption)                            │
│  "A person lying on the floor in a room"                         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Translation (English → Vietnamese)                              │
│  "Một người nằm trên sàn trong phòng"                            │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Add Emergency Context                                           │
│  "KHẨN CẤP - TÉ NGÃ: Một người nằm trên sàn trong phòng          │
│   - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: 75%)"                 │
└─────────────────────────────────────────────────────────────────┘
```

### Caption Enhancement

```python
def _enhance_caption_with_emergency_context(base_caption, status, event_type, confidence):
    if status == "danger":
        if event_type == "fall":
            prefix = "KHẨN CẤP - TÉ NGÃ:"
            suffix = f" - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: {confidence:.0%})"
        elif event_type in ["seizure", "abnormal_behavior"]:
            prefix = "KHẨN CẤP - CO GIẬT:"
            suffix = f" - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"

    elif status == "warning":
        if event_type == "fall":
            prefix = "CẢNH BÁO TÉ NGÃ:"
            suffix = f" - Cần theo dõi và kiểm tra (Tin cậy: {confidence:.0%})"

    return f"{prefix} {base_caption}{suffix}"
```

---

## 7. Static Fallback Messages

Khi AI caption không khả dụng:

```python
def _generate_static_action_message(status, event_type, confidence):
    if status == "danger":
        if event_type == "fall":
            return "⚠️ BÁO ĐỘNG NGUY HIỂM: Phát hiện té - Yêu cầu hỗ trợ gấp!"
        elif event_type in ["seizure", "abnormal_behavior"]:
            return "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!"

    elif status == "warning":
        return f"Phát hiện {event_type} ({confidence:.0%} confidence) - Cần theo dõi"

    else:  # normal
        return "Không có gì bất thường"
```

---

## 8. Event Response Format

```python
def _create_event_response(event_id, status, event_type, confidence,
                          camera_id, snapshot_timestamp, image_path=None):
    return {
        "imageUrl": f"{base_url}/snapshots/{event_id}.jpg",
        "status": status,        # "normal" | "warning" | "danger"
        "action": action_message, # AI or static message
        "time": snapshot_timestamp.isoformat()
    }
```

### Example Response

```json
{
  "imageUrl": "https://api.example.com/snapshots/abc-123.jpg",
  "status": "danger",
  "action": "KHẨN CẤP - TÉ NGÃ: Người cao tuổi nằm trên sàn phòng khách - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: 82%)",
  "time": "2024-01-15T10:30:45.123456"
}
```

---

## 9. Realtime Event Listeners

```python
def _setup_event_listeners():
    # Subscribe to database changes
    realtime_service.subscribe_to_events(
        'event_detections',
        'INSERT',
        self._handle_event_detection
    )
    realtime_service.subscribe_to_events(
        'event_detections',
        'INSERT',
        self._handle_alert
    )
```

### Handler Functions

```python
def _handle_event_detection(event_data):
    """Handle new detection from realtime"""
    detection = event_data.get('new_data', {})
    # Log, notify, update UI, etc.

def _handle_alert(event_data):
    """Handle new alert from realtime"""
    alert = event_data.get('new_data', {})
    # Custom alert handling
```

---

## 10. Publishing Methods

### `publish_fall_detection()`

```python
def publish_fall_detection(
    confidence: float,
    bounding_boxes: List[Dict],
    context: Optional[Dict] = None,
    camera_id: Optional[str] = None,
    room_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    # 1. Check priority gating
    # 2. Create event if allowed
    # 3. Generate response
```

### `publish_seizure_detection()`

```python
def publish_seizure_detection(
    confidence: float,
    severity: str,
    context: Optional[Dict] = None,
    ...
) -> Dict[str, Any]:
    # Similar flow to fall detection
```

---

## 11. Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│     INPUT: Detection Result (type, confidence, frame, ...)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Map Confidence → Severity                                       │
│  Fall: ≥0.60=high, ≥0.40=medium, <0.40=low                       │
│  Seizure: ≥0.50=high, ≥0.30=medium, <0.30=low                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Calculate Priority Level (1-4)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Priority Gating Check                                           │
│  ┌──────────────────────────────────┐                           │
│  │ new_priority >= current_max ?    │──No──▶ Skip (return)      │
│  └──────────────────────────────────┘                           │
└──────────────┼──────────────────────────────────────────────────┘
               │ Yes
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Generate Action Message                                         │
│  ┌────────────────────┐     ┌────────────────────┐              │
│  │ BLIP + Translation │ OR  │ Static Fallback    │              │
│  └────────────────────┘     └────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Create Event Response                                           │
│  {imageUrl, status, action, time}                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Publish to Database + Realtime                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notes (VI)

Ánh xạ confidence → severity (high/medium/low) dựa trên ngưỡng cấu hình. Priority system (1-4) quyết định có tạo alert không - chỉ tạo nếu priority mới ≥ priority cao nhất hiện tại. AI caption (BLIP → Vietnamese) cung cấp mô tả ngữ cảnh, fallback sang message tĩnh nếu không khả dụng.
