# Event Lifecycle & Snapshots (Operational Flow)

- **Mục đích**: Mô tả chi tiết luồng vận hành từ khi phát hiện sự kiện đến khi resolved, bao gồm lifecycle states và snapshot management.

---

## 1. Event Lifecycle States

### State Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DETECTED                                 │
│                    (Initial detection)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Create event in DB
                              │ Capture snapshots
                              │ Trigger alarm
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ALARM_ACTIVATED                              │
│                  (Alarm playing, waiting)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   ACKNOWLEDGED  │ │   AUTOCALLED    │ │   EXPIRED       │
│ (User confirmed)│ │ (30s timeout,   │ │ (No action,     │
│                 │ │  auto-call EMS) │ │  auto-expire)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RESOLVED                                 │
│                    (Case closed)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### State Definitions

| State             | Description                            | Next States                       |
| ----------------- | -------------------------------------- | --------------------------------- |
| `DETECTED`        | Sự kiện vừa phát hiện                  | ALARM_ACTIVATED                   |
| `ALARM_ACTIVATED` | Còi đang kêu, chờ response             | ACKNOWLEDGED, AUTOCALLED, EXPIRED |
| `ACKNOWLEDGED`    | User đã xác nhận trong app             | RESOLVED                          |
| `AUTOCALLED`      | Hết 30s không response → gọi emergency | RESOLVED                          |
| `EXPIRED`         | Event hết hạn không action             | RESOLVED                          |
| `RESOLVED`        | Đã xử lý xong                          | (Final state)                     |

---

## 2. Complete Operational Flow

### Phase 1: Detection

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Video Frame Input                                            │
│       │                                                          │
│       ▼                                                          │
│  2. Keyframe Check                                               │
│       │ Is keyframe? ──No──▶ Skip heavy processing              │
│       │ Yes                                                      │
│       ▼                                                          │
│  3. Person Detection (YOLOv8)                                    │
│       │                                                          │
│       ▼                                                          │
│  4. Dual Detection (Parallel)                                    │
│       ├── Fall Detection (SimpleFallDetector)                   │
│       └── Seizure Detection (VSViG + SeizurePredictor)          │
│       │                                                          │
│       ▼                                                          │
│  5. Check Cooldowns                                              │
│       │ In cooldown? ──Yes──▶ Skip event creation               │
│       │ No                                                       │
│       ▼                                                          │
│  6. Threshold Check                                              │
│       │ Passes thresholds? ──No──▶ Return detection result only │
│       │ Yes                                                      │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
```

### Phase 2: Event Creation

```
┌─────────────────────────────────────────────────────────────────┐
│  7. Create Event in Database                                     │
│       │                                                          │
│       │  INSERT INTO event_detections (                         │
│       │      event_id, event_type, confidence_score,            │
│       │      camera_id, user_id, lifecycle_state='DETECTED'     │
│       │  )                                                       │
│       │                                                          │
│       │  Returns: event_id (UUID)                               │
│       ▼                                                          │
│  8. Update Pipeline State                                        │
│       │                                                          │
│       │  _active_event_id = event_id                            │
│       │  _last_any_event_time = now()                           │
│       │  Start 45s global cooldown                              │
│       ▼                                                          │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
```

### Phase 3: Snapshot Capture

```
┌─────────────────────────────────────────────────────────────────┐
│  9. Gather Frames from Buffer                                    │
│       │                                                          │
│       │  frames_to_capture = [                                  │
│       │      buffer[-3],  # 3 frames ago                        │
│       │      buffer[-2],  # 2 frames ago                        │
│       │      buffer[-1],  # 1 frame ago                         │
│       │      current,     # Current frame                       │
│       │      current_with_overlay  # With detection boxes       │
│       │  ]                                                       │
│       ▼                                                          │
│  10. Upload to MinIO                                             │
│       │                                                          │
│       │  for frame in frames_to_capture:                        │
│       │      minio_service.upload_frame_image(                  │
│       │          frame, camera_id, event_type,                  │
│       │          confidence, metadata={event_id: ...}           │
│       │      )                                                   │
│       ▼                                                          │
│  11. Create Database Records                                     │
│       │                                                          │
│       │  INSERT INTO snapshots (snapshot_id, camera_id, ...)    │
│       │  INSERT INTO snapshot_images (image_id, cloud_url, ...) │
│       ▼                                                          │
│  12. Link Snapshot to Event                                      │
│       │                                                          │
│       │  UPDATE event_detections                                │
│       │  SET snapshot_id = ?                                    │
│       │  WHERE event_id = ?                                     │
│       ▼                                                          │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
```

### Phase 4: Alarm & Notification

```
┌─────────────────────────────────────────────────────────────────┐
│  13. Trigger Alarm                                               │
│       │                                                          │
│       │  NOTIFY system_alarm_trigger_channel                    │
│       │  Payload: {event_id, action: 'TRIGGER_ALARM'}           │
│       │                                                          │
│       │  → EmergencyAlarmHandler receives                       │
│       │  → audio_alert_service.play_emergency_alarm()           │
│       ▼                                                          │
│  14. Update Lifecycle State                                      │
│       │                                                          │
│       │  UPDATE event_detections                                │
│       │  SET lifecycle_state = 'ALARM_ACTIVATED'                │
│       ▼                                                          │
│  15. Send Push Notification                                      │
│       │                                                          │
│       │  FCM/APNS notification to mobile app                    │
│       │  {title, body, event_id, image_url}                     │
│       ▼                                                          │
│  16. Publish Realtime Update                                     │
│       │                                                          │
│       │  Supabase Realtime broadcast                            │
│       │  Mobile app receives instant update                     │
│       ▼                                                          │
└───────┼─────────────────────────────────────────────────────────┘
        ▼
```

### Phase 5: Response Handling

```
┌─────────────────────────────────────────────────────────────────┐
│  17. Wait for Response (30s timeout)                             │
│       │                                                          │
│       ├──▶ User taps "Acknowledge" in app                       │
│       │       │                                                  │
│       │       ▼                                                  │
│       │    lifecycle_state → ACKNOWLEDGED                        │
│       │    Stop alarm                                            │
│       │                                                          │
│       ├──▶ 30s elapsed, no response                             │
│       │       │                                                  │
│       │       ▼                                                  │
│       │    lifecycle_state → AUTOCALLED                         │
│       │    Call emergency services (Twilio/etc)                 │
│       │    Alarm continues!                                      │
│       │                                                          │
│       └──▶ User manually stops alarm                            │
│               │                                                  │
│               ▼                                                  │
│            NOTIFY system_alarm_stop_channel                      │
│            Stop alarm                                            │
│            lifecycle_state → RESOLVED                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Snapshot Structure

### 5 Images Captured Per Event

| Index | Content           | Purpose              |
| ----- | ----------------- | -------------------- |
| 0     | Frame t-3         | Context before event |
| 1     | Frame t-2         | Context before event |
| 2     | Frame t-1         | Just before event    |
| 3     | Frame t (current) | Event moment         |
| 4     | Frame t + overlay | With detection boxes |

### MinIO Object Structure

```
healthcare-snapshots/
└── {user_id}/
    └── {camera_id}/
        └── {event_type}/
            ├── {timestamp}_frame0_conf0.85.jpg
            ├── {timestamp}_frame1_conf0.85.jpg
            ├── {timestamp}_frame2_conf0.85.jpg
            ├── {timestamp}_frame3_conf0.85.jpg
            └── {timestamp}_overlay_conf0.85.jpg
```

### Database Relationships

```sql
event_detections
    └── snapshot_id ──────┐
                          │
snapshots ◀───────────────┘
    └── snapshot_id ──────┐
                          │
snapshot_images ◀─────────┘
    └── cloud_url (MinIO URL)
```

---

## 4. Timing & Timeouts

| Phase                         | Duration   | Notes                |
| ----------------------------- | ---------- | -------------------- |
| Detection → Event created     | < 100ms    | Fast DB insert       |
| Event → Snapshots uploaded    | 500ms - 2s | Depends on network   |
| Event → Alarm starts          | < 50ms     | LISTEN/NOTIFY        |
| Alarm → User response timeout | 30s        | Configurable         |
| Global cooldown               | 45s        | Between any events   |
| DANGER → block NORMAL         | 60s        | Prevent false NORMAL |

---

## 5. Error Recovery

### Snapshot Upload Failed

```python
try:
    snapshot_id = snapshot_service.create_detection_snapshot(...)
except Exception as e:
    logger.error(f"Snapshot failed: {e}")
    # Event already created, just missing snapshot
    # Can retry later or mark event as "no_snapshot"
```

### Alarm Trigger Failed

```python
try:
    alarm_result = audio_alert_service.play_emergency_alarm(...)
except Exception as e:
    logger.error(f"Alarm failed: {e}")
    # Event and snapshots exist
    # Fallback: send push notification only
    send_critical_push_notification(event_id)
```

### Database Connection Lost

```python
try:
    event_id = postgresql_service.publish_event_detection(...)
except psycopg2.OperationalError:
    # Queue event locally
    local_event_queue.append(event_data)
    # Retry when connection restored
```

---

## 6. Mobile App Integration

### Notification Payload

```json
{
  "notification": {
    "title": "⚠️ TÉ NGÃ PHÁT HIỆN",
    "body": "Phòng khách - Cần kiểm tra ngay!"
  },
  "data": {
    "event_id": "abc-123-xyz",
    "event_type": "fall",
    "confidence": "0.85",
    "image_url": "https://minio.../snapshot.jpg",
    "action": "OPEN_EVENT_DETAIL"
  }
}
```

### App Actions

| Action         | API Call                        | Result                         |
| -------------- | ------------------------------- | ------------------------------ |
| Acknowledge    | `POST /events/{id}/acknowledge` | ALARM_ACTIVATED → ACKNOWLEDGED |
| Resolve        | `POST /events/{id}/resolve`     | Any state → RESOLVED           |
| View snapshots | `GET /events/{id}/snapshots`    | Return all snapshot URLs       |
| Stop alarm     | `POST /alarms/stop`             | Stop audio, update state       |

---

## 7. Audit Trail

### Event Timeline (notes column)

```sql
UPDATE event_detections
SET notes = notes || E'\n' ||
    '[' || NOW() || '] Action: ' || action_description
WHERE event_id = ?;
```

### Example Notes

```
[2024-01-15 10:30:45] Created: Fall detected (conf: 0.85)
[2024-01-15 10:30:46] Snapshots: 5 images captured
[2024-01-15 10:30:46] Alarm: Triggered via LISTEN/NOTIFY
[2024-01-15 10:31:15] Timeout: No response in 30s
[2024-01-15 10:31:15] AUTOCALLED: Emergency services contacted
[2024-01-15 10:35:00] Resolved: Stopped by family_member (False alarm)
```

---

## Notes (VI)

Luồng hoàn chỉnh: Keyframe → Detection → Event DB → 5 Snapshots (MinIO) → Link → Alarm (LISTEN/NOTIFY) → Push Notification → Wait 30s → ACKNOWLEDGED hoặc AUTOCALLED → RESOLVED. Mỗi phase có error handling riêng để đảm bảo không mất data. Audit trail trong notes column để trace toàn bộ lifecycle.
