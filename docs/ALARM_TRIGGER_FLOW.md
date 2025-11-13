# 🔊 ALARM TRIGGER FLOW - LUỒNG KÍCH HOẠT CÒI BÁO ĐỘNG

## 📋 **FULL FLOW: DETECTION → ALARM**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. DETECTION PHASE (src/main.py)                                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 355-385: Main loop phát hiện fall/seizure
    │   - Advanced Healthcare Pipeline detect event
    │   - Confidence >= threshold (Fall: 0.45, Seizure: 0.30)
    │
    ├─ if event_type == 'fall':
    │   └─> publish_fall_detection(confidence, bounding_boxes, context)
    │
    └─ else: (seizure)
        └─> publish_seizure_detection(confidence, bounding_boxes, context)

┌─────────────────────────────────────────────────────────────────────┐
│ 2. EVENT PUBLISHER (src/service/emergency_notification_dispatcher.py)│
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 360-410: handle_fall_detection()
    │   ├─ Tạo event_data với frame (cho MinIO upload)
    │   ├─ Xác định severity: 'critical' hoặc 'warning'
    │   └─> Call postgresql_service.publish_event_detection(event_data)
    │
    └─ Line 480-560: handle_seizure_detection() (tương tự)

┌─────────────────────────────────────────────────────────────────────┐
│ 3. DATABASE INSERT (src/service/postgresql_healthcare_service.py)   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 740-1000: publish_event_detection()
    │   │
    │   ├─ Line 775-810: Upload image to MinIO (nếu có frame)
    │   │   └─> snapshot_service.create_detection_snapshot()
    │   │
    │   ├─ Line 908: ⚠️ KEY POINT: Set lifecycle_state
    │   │   'lifecycle_state': 'NOTIFIED'  ← Chỉ thông báo, KHÔNG alarm!
    │   │
    │   └─ Line 926-950: INSERT INTO event_detections
    │       - event_id, user_id, camera_id
    │       - event_type, confidence_score
    │       - lifecycle_state = 'NOTIFIED'  ← Không trigger alarm!
    │       - status = 'danger' hoặc 'warning'
    │
    └─> Event saved to database với state = 'NOTIFIED'

┌─────────────────────────────────────────────────────────────────────┐
│ 4. SUPABASE REALTIME (Automatic)                                    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Database → Supabase Realtime Channel
    │   - Broadcast INSERT event
    │   - Include: event_id, event_type, confidence, status
    │
    └─> Mobile App nhận notification (Realtime)

┌─────────────────────────────────────────────────────────────────────┐
│ 5. MOBILE APP (Flutter/React Native)                                │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Nhận event từ Supabase Realtime
    │   - Show notification popup
    │   - Display snapshot image (MinIO URL)
    │   - Show event details (type, confidence, camera)
    │
    ├─ User xem và quyết định:
    │   ├─ Option 1: Bấm "Dismiss" → Bỏ qua
    │   └─ Option 2: Bấm "Activate Alarm" → Trigger alarm ⬇️
    │
    └─ Nếu user bấm "Activate Alarm":
        └─> Call API: UPDATE event_detections
            SET lifecycle_state = 'ALARM_ACTIVATED'
            WHERE event_id = '...'

┌─────────────────────────────────────────────────────────────────────┐
│ 6. DATABASE TRIGGER (Supabase Function - Server Side)               │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Trigger: alarm_activation_trigger
    │   Function: notify_alarm_trigger()
    │
    ├─ Điều kiện kích hoạt:
    │   IF NEW.lifecycle_state = 'ALARM_ACTIVATED'
    │      AND OLD.lifecycle_state != 'ALARM_ACTIVATED'
    │
    └─ Action: NOTIFY PostgreSQL channel
        └─> pg_notify('system_alarm_channel', json_payload)
            - event_id
            - user_id
            - camera_id
            - state: 'ALARM_ACTIVATED'
            - message

┌─────────────────────────────────────────────────────────────────────┐
│ 7. ALARM HANDLER (src/infrastructure/services/                      │
│                    emergency_alarm_handler_psycopg.py)              │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 93-105: LISTEN to PostgreSQL channel
    │   - Channel: 'system_alarm_channel'
    │   - Connection: Direct port 5432 (not pooler)
    │   - Wait for notifications (realtime, < 50ms latency)
    │
    ├─ Line 145-172: _handle_notification()
    │   - Nhận notification từ database trigger
    │   - Parse JSON payload
    │   - Extract: event_id, user_id, state
    │
    ├─ Line 167-168: Check state
    │   if state == 'ALARM_ACTIVATED':
    │       └─> _process_alarm_activated_sync(data)
    │
    └─ Line 230-268: _process_alarm_activated_sync()
        └─> Trigger còi! ⬇️

┌─────────────────────────────────────────────────────────────────────┐
│ 8. AUDIO ALERT SERVICE (src/infrastructure/services/                │
│                         audio_alert_service.py)                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 244-247: Call play_emergency_alarm()
    │   alarm_result = asyncio.run(
    │       audio_alert_service.play_emergency_alarm(
    │           user_id=user_id,
    │           triggered_by='alarm_activation',
    │           duration=10  # 10 giây
    │       )
    │   )
    │
    ├─ Line 160-230: play_emergency_alarm()
    │   ├─ Load sound file: emergency_siren.mp3
    │   ├─ Set volume: 100%
    │   ├─ Play với pygame.mixer
    │   ├─ Loop: -1 (infinite loop)
    │   └─ Schedule auto-stop after 10s
    │
    └─> 🔊 ALARM ĐANG PHÁ CÒI!

┌─────────────────────────────────────────────────────────────────────┐
│ 9. AUTO STOP (After 10 seconds)                                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├─ Line 252-257: _auto_stop_after_duration()
    │   await asyncio.sleep(10)  # Wait 10 seconds
    │   await stop_alarm()
    │
    ├─ Line 263-291: stop_alarm()
    │   pygame.mixer.stop()  # Stop playback
    │   is_playing = False
    │
    └─> 🔇 Alarm tự động tắt sau 10s

┌─────────────────────────────────────────────────────────────────────┐
│ 10. UPDATE DATABASE (Alarm Handler)                                 │
└─────────────────────────────────────────────────────────────────────┘
    │
    └─ Line 254-260: _update_event_status()
        UPDATE event_detections
        SET lifecycle_state = 'ACKED',  # Acknowledged
            notes = 'ALARM ACTIVATED FROM MOBILE at ...'
        WHERE event_id = '...'
```

---

## 🎯 **KEY POINTS**

### ❌ **KHÔNG TỰ ĐỘNG TRIGGER:**

```python
# Line 908: postgresql_healthcare_service.py
'lifecycle_state': 'NOTIFIED'  # ← Chỉ thông báo, KHÔNG alarm!
```

**Lý do:** Event luôn được tạo với `lifecycle_state = 'NOTIFIED'`, không có logic tự động chuyển sang `'ALARM_ACTIVATED'`

### ✅ **MANUAL TRIGGER (Qua Mobile):**

```
Detection → Database (NOTIFIED) → Mobile Notification →
User bấm button → UPDATE (ALARM_ACTIVATED) → Trigger fires →
Handler nhận NOTIFY → Play alarm
```

---

## 📂 **FILES LIÊN QUAN:**

1. **src/main.py** (Line 355-385)

   - Detect fall/seizure events
   - Call event*publisher.publish*\*\_detection()

2. **src/service/emergency_notification_dispatcher.py** (Line 360-560)

   - handle_fall_detection()
   - handle_seizure_detection()
   - Call postgresql_service.publish_event_detection()

3. **src/service/postgresql_healthcare_service.py** (Line 740-1000)

   - publish_event_detection()
   - INSERT với lifecycle_state='NOTIFIED' ← KEY!
   - Upload MinIO image

4. **Database Trigger** (Supabase SQL Editor)

   - notify_alarm_trigger() function
   - Trigger on UPDATE lifecycle_state → 'ALARM_ACTIVATED'
   - NOTIFY 'system_alarm_channel'

5. **src/infrastructure/services/emergency_alarm_handler_psycopg.py**

   - LISTEN 'system_alarm_channel'
   - \_handle_notification() → \_process_alarm_activated_sync()
   - Call audio_alert_service.play_emergency_alarm()

6. **src/infrastructure/services/audio_alert_service.py**
   - play_emergency_alarm() - Phát còi
   - stop_alarm() - Tắt còi
   - \_auto_stop_after_duration() - Auto-stop 10s

---

## 🔧 **ĐỂ TỰ ĐỘNG ALARM KHI DETECT:**

### **Thay đổi Line 908 trong postgresql_healthcare_service.py:**

```python
# BEFORE:
'lifecycle_state': 'NOTIFIED',  # Chỉ thông báo

# AFTER (Option 1 - Auto alarm cho high confidence):
'lifecycle_state': 'ALARM_ACTIVATED' if event_data.get('confidence', 0.0) >= 0.60 else 'NOTIFIED',

# AFTER (Option 2 - Luôn auto alarm):
'lifecycle_state': 'ALARM_ACTIVATED',  # Tự động alarm mọi detection
```

**Kết quả:**

```
Detection (confidence >= 0.60) →
Database INSERT với lifecycle_state='ALARM_ACTIVATED' →
Trigger fires ngay lập tức →
Alarm handler nhận NOTIFY →
🔊 Còi kêu tự động!
```

---

## 🧪 **TESTING FLOW:**

### **Test Manual Trigger:**

```bash
# Terminal 1: Run main.py (start alarm handler)
cd src
python main.py

# Terminal 2: Trigger alarm manually
cd examples
python trigger_alarm_test.py
# → Select event → UPDATE lifecycle_state → Alarm plays!
```

### **Test Detection Flow:**

```bash
# Run main.py
python src/main.py

# Press 'e' key → Create test event
# Check database:
# - lifecycle_state = 'NOTIFIED' (not ALARM_ACTIVATED)
# - Alarm KHÔNG tự động phát
```

---

## 📊 **DATABASE STATES:**

```sql
-- Events flow through these states:
NOTIFIED          ← Detection mới (default)
    ↓ (Manual user action)
ALARM_ACTIVATED   ← User bấm "Activate Alarm" button
    ↓ (Handler processed)
ACKED             ← Alarm đã phát, đã xác nhận
    ↓ (Optional)
DISMISSED         ← User bấm "Dismiss"
RESOLVED          ← Sự kiện đã xử lý xong
```

---

## 🎯 **SUMMARY:**

**Hiện tại:**

- Detection → Database (NOTIFIED) → Mobile → **User phải bấm button** → Alarm
- Alarm **KHÔNG tự động** khi detect

**Để tự động:**

- Thay đổi 1 dòng code (Line 908)
- Detection → Database (ALARM_ACTIVATED) → Trigger → **Alarm tự động**

**Anh muốn em thay đổi thành tự động alarm không? 🤔**
