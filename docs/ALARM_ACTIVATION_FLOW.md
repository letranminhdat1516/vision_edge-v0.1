# 🔊 ALARM ACTIVATION FLOW - ĐIỀU KIỆN KÍCH HOẠT CÒI BÁO ĐỘNG

## 🎯 **TÓM TẮT NGẮN GỌN**

**Còi báo động CHƯA TỰ ĐỘNG kích hoạt khi phát hiện sự kiện!**

Hiện tại còi chỉ kích hoạt khi:

1. ✅ **User bấm nút "Activate Alarm" trên mobile app**
2. ✅ **Manual trigger qua script `trigger_alarm_test.py`**

**KHÔNG** tự động kích hoạt khi:

- ❌ Phát hiện Fall Detection (té ngã)
- ❌ Phát hiện Seizure Detection (co giật)
- ❌ Confidence cao
- ❌ Status = 'danger'

---

## 📋 **CHI TIẾT FLOW HIỆN TẠI**

### **1️⃣ DETECTION → DATABASE (Tự động)**

```python
# src/service/postgresql_healthcare_service.py - Line 908
'lifecycle_state': 'NOTIFIED',  # ✅ Event được tạo với state NOTIFIED
'confirmation_state': 'DETECTED',
'verification_status': 'PENDING',
'status': 'danger' hoặc 'warning'  # Dựa vào confidence
```

**Điều kiện status:**

- Fall: confidence >= 0.60 → `danger`, >= 0.40 → `warning`
- Seizure: confidence >= 0.50 → `danger`, >= 0.30 → `warning`

**Kết quả:** Event được lưu vào database với `lifecycle_state = 'NOTIFIED'`

---

### **2️⃣ MOBILE APP NHẬN NOTIFICATION (Tự động)**

```
Database → Supabase Realtime → Mobile App
```

**Mobile nhận:**

- Event details (event_type, confidence, camera_id)
- Status (danger/warning)
- Snapshot image URL (MinIO)

**Mobile hiển thị:**

- 🚨 Alert popup
- 📸 Snapshot image
- ⏰ Timestamp
- 🔘 Button "Activate Alarm" ← **KEY POINT!**

---

### **3️⃣ ALARM ACTIVATION (Manual - Qua Mobile)**

**User bấm "Activate Alarm" button trên mobile:**

```sql
-- Mobile app gửi API request
UPDATE event_detections
SET lifecycle_state = 'ALARM_ACTIVATED',
    last_action_at = NOW()
WHERE event_id = '...'
```

**Database Trigger fires:**

```sql
-- Trigger function: notify_alarm_trigger()
-- Location: Supabase SQL Editor (server-side)

CREATE OR REPLACE FUNCTION notify_alarm_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Chỉ trigger khi lifecycle_state thay đổi THÀNH 'ALARM_ACTIVATED'
    IF NEW.lifecycle_state = 'ALARM_ACTIVATED'
       AND OLD.lifecycle_state != 'ALARM_ACTIVATED' THEN

        -- Gửi notification qua PostgreSQL NOTIFY
        PERFORM pg_notify('system_alarm_channel', json_build_object(
            'event_id', NEW.event_id,
            'user_id', NEW.user_id,
            'camera_id', NEW.camera_id,
            'state', 'ALARM_ACTIVATED',
            'message', 'Alarm activated by user',
            'old_lifecycle_state', OLD.lifecycle_state,
            'new_lifecycle_state', NEW.lifecycle_state
        )::text);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER alarm_activation_trigger
AFTER UPDATE ON event_detections
FOR EACH ROW
EXECUTE FUNCTION notify_alarm_trigger();
```

---

### **4️⃣ ALARM HANDLER NHẬN NOTIFICATION (Tự động)**

**Handler lắng nghe PostgreSQL channel:**

```python
# src/infrastructure/services/emergency_alarm_handler_psycopg.py

# Line 42: Lắng nghe channel
self.channel_name = 'system_alarm_channel'

# Line 93: Kết nối và LISTEN
cur.execute(f"LISTEN {self.channel_name};")

# Line 98: Nhận notification
for notify in gen:
    self._handle_notification(notify)  # Line 102

# Line 167: Check state
if state == 'ALARM_ACTIVATED':
    self._process_alarm_activated_sync(data)  # Line 168
```

---

### **5️⃣ ALARM PLAYBACK (Tự động)**

```python
# Line 231-268: _process_alarm_activated_sync()

# Line 242: Trigger alarm
alarm_result = asyncio.run(audio_alert_service.play_emergency_alarm(
    user_id=user_id,
    triggered_by='alarm_activation',
    duration=10  # 10 giây
))

# Line 245-248: Nếu thành công
if alarm_result['success']:
    logger.info("✅ ✅ ✅ ALARM ACTIVATED SUCCESSFULLY! ✅ ✅ ✅")
    logger.info(f"   Volume: {alarm_result.get('volume', 1.0) * 100:.0f}%")

    # Update event state
    self._update_event_status(
        event_id=event_id,
        lifecycle_state='ACKED',  # Đã xác nhận
        notes=f"ALARM ACTIVATED FROM MOBILE at {datetime.now()}"
    )
```

**Audio Service:**

```python
# src/infrastructure/services/audio_alert_service.py
# Plays emergency_alarm.wav (10s, volume 100%)
```

---

## ⚠️ **VẤN ĐỀ HIỆN TẠI**

### ❌ **KHÔNG TỰ ĐỘNG ALARM KHI DETECT**

**Flow thiếu:**

```
Detection (confidence >= 0.60)
    ↓ ❌ KHÔNG TỰ ĐỘNG
    ↓
ALARM_ACTIVATED
```

**Lý do:**

- `publish_event_detection()` tạo event với `lifecycle_state='NOTIFIED'`
- Không có logic tự động chuyển sang `'ALARM_ACTIVATED'`
- Chỉ đợi user bấm button trên mobile

---

## 💡 **GIẢI PHÁP ĐỀ XUẤT**

### **Option 1: AUTO ALARM cho Critical Events (RECOMMEND)**

**Tự động kích hoạt alarm khi:**

- Event type = 'fall' hoặc 'seizure'
- Confidence >= 0.60 (danger threshold)
- Status = 'danger'

**Implementation:**

```python
# src/service/postgresql_healthcare_service.py
# Line 908: Thay đổi logic

if event_data.get('confidence', 0.0) >= 0.60:
    initial_lifecycle_state = 'ALARM_ACTIVATED'  # ✅ Auto trigger alarm
else:
    initial_lifecycle_state = 'NOTIFIED'  # Chỉ thông báo

db_event = {
    # ...
    'lifecycle_state': initial_lifecycle_state,
    # ...
}
```

**Kết quả:**

- High confidence events → Alarm tự động
- Low confidence events → Chỉ notification, user quyết định

---

### **Option 2: ALWAYS NOTIFY, NEVER AUTO-ALARM (Hiện tại)**

**Keep current behavior:**

- Mọi detection → `lifecycle_state='NOTIFIED'`
- User xem notification → Quyết định activate alarm
- Tránh false positive gây phiền

**Phù hợp khi:**

- False positive rate còn cao
- Người dùng muốn control hoàn toàn
- Demo cần manual control

---

### **Option 3: CONFIGURABLE THRESHOLD**

**Thêm config auto-alarm threshold:**

```python
# .env hoặc database config
AUTO_ALARM_THRESHOLD=0.70  # Chỉ auto alarm khi >= 70%

# Code
if event_data.get('confidence', 0.0) >= auto_alarm_threshold:
    lifecycle_state = 'ALARM_ACTIVATED'
else:
    lifecycle_state = 'NOTIFIED'
```

---

## 📊 **LIFECYCLE STATES**

```
NOTIFIED          ← Phát hiện event, gửi notification
    ↓ (User action hoặc auto)
ALARM_ACTIVATED   ← Trigger alarm playback
    ↓ (Handler xử lý)
ACKED             ← Đã xác nhận, alarm đã play
    ↓ (Optional)
RESOLVED          ← Đã giải quyết
```

---

## 🔍 **KIỂM TRA HIỆN TẠI**

### **Check Database Trigger:**

```sql
-- Supabase SQL Editor
SELECT
    trigger_name,
    event_manipulation,
    event_object_table,
    action_statement
FROM information_schema.triggers
WHERE trigger_name LIKE '%alarm%';
```

### **Check Recent Events:**

```sql
SELECT
    event_id,
    event_type,
    confidence_score,
    status,
    lifecycle_state,
    detected_at
FROM event_detections
WHERE user_id = '37cbad15-483d-42ff-b07d-fbf3cd1cc863'
ORDER BY detected_at DESC
LIMIT 10;
```

**Expected:**

- `lifecycle_state = 'NOTIFIED'` ← Mới tạo
- `lifecycle_state = 'ALARM_ACTIVATED'` ← User bấm button
- `lifecycle_state = 'ACKED'` ← Alarm đã play

---

## 🎯 **TESTING**

### **Test Manual Alarm:**

```bash
# Terminal 1: Run main.py
cd src
python main.py

# Terminal 2: Trigger alarm
cd examples
python trigger_alarm_test.py
```

### **Test Auto Detection:**

```bash
# Run main.py
python src/main.py

# Press 'e' key to create test events
# Check: lifecycle_state = 'NOTIFIED' (not ALARM_ACTIVATED)
```

---

## 📝 **KẾT LUẬN**

**Hiện tại:**

- ✅ Detection → Database → Mobile Notification: **HOẠT ĐỘNG**
- ✅ Mobile Button → Alarm Trigger: **HOẠT ĐỘNG**
- ❌ Detection → Auto Alarm: **KHÔNG HOẠT ĐỘNG**

**Để tự động alarm khi detect:**

- Cần thay đổi logic tạo event (Option 1 hoặc 3)
- Hoặc giữ nguyên để user control (Option 2)

**Anh muốn em implement Option nào? 🤔**
