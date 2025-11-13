# 🔇 ALARM STOP CONDITIONS - ĐIỀU KIỆN DỪNG ALARM

## 🎯 **TÓM TẮT THAY ĐỔI**

Đã thêm **4 cơ chế dừng alarm**:

1. ✅ **Stop khi detect >= 2 người** (Safety check - Ngăn false alarm)
2. ✅ **Stop khi lifecycle_state thay đổi** (User dismiss/resolve event)
3. ✅ **Auto-stop sau 10 giây** (Timeout fallback)
4. ✅ **Manual stop** (Direct call hoặc keyboard shortcut)

---

## 📋 **CHI TIẾT THAY ĐỔI**

### **1️⃣ STOP KHI DETECT >= 2 NGƯỜI**

**Logic:** Nếu phát hiện >= 2 người, alarm tự động dừng (giả định có người đến giúp đỡ)

#### **File: src/main.py**

**Multi-camera mode (Line 328-338):**

```python
result = cam_data['pipeline'].process_frame(frame)
detection_result = result["detection_result"]
person_detections = result["person_detections"]

# ✅ STOP ALARM if detect >= 2 people (safety check)
num_people = len(person_detections) if person_detections else 0
if num_people >= 2 and audio_alert_service.is_playing:
    print(f"👥 Detected {num_people} people - STOPPING ALARM (safety check)")
    import asyncio
    asyncio.run(audio_alert_service.stop_alarm())
```

**Single-camera mode (Line 610-620):**

```python
result = pipeline.process_frame(frame)
detection_result = result["detection_result"]
person_detections = result["person_detections"]

# ✅ STOP ALARM if detect >= 2 people (safety check)
num_people = len(person_detections) if person_detections else 0
if num_people >= 2 and audio_alert_service.is_playing:
    print(f"👥 Detected {num_people} people - STOPPING ALARM (safety check)")
    import asyncio
    asyncio.run(audio_alert_service.stop_alarm())
```

**Kết quả:**

```
Frame 1: Detect 1 person falling → Alarm starts 🔊
Frame 2: Detect 1 person on ground → Alarm continues 🔊
Frame 3: Detect 2 people → Alarm STOPS 🔇 (Someone came to help!)
```

---

### **2️⃣ STOP KHI LIFECYCLE_STATE THAY ĐỔI**

**Logic:** Khi user/system thay đổi lifecycle_state từ `ALARM_ACTIVATED` sang state khác, alarm dừng

#### **File: database_triggers_alarm_stop.sql**

**Database Trigger:**

```sql
CREATE OR REPLACE FUNCTION notify_alarm_stop_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Trigger khi lifecycle_state thay đổi TỪ ALARM_ACTIVATED sang state khác
    IF OLD.lifecycle_state = 'ALARM_ACTIVATED'
       AND NEW.lifecycle_state != 'ALARM_ACTIVATED' THEN

        -- Gửi notification qua PostgreSQL NOTIFY
        PERFORM pg_notify('system_alarm_stop_channel', json_build_object(
            'event_id', NEW.event_id,
            'user_id', NEW.user_id,
            'camera_id', NEW.camera_id,
            'action', 'STOP_ALARM',
            'old_lifecycle_state', OLD.lifecycle_state,
            'new_lifecycle_state', NEW.lifecycle_state,
            'message', 'Lifecycle state changed from ALARM_ACTIVATED to ' || NEW.lifecycle_state
        )::text);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER alarm_stop_trigger
AFTER UPDATE ON event_detections
FOR EACH ROW
EXECUTE FUNCTION notify_alarm_stop_trigger();
```

#### **File: src/infrastructure/services/emergency_alarm_handler_psycopg.py**

**Line 42-43: Thêm stop channel**

```python
self.channel_name = 'system_alarm_channel'  # Alarm activation
self.stop_channel_name = 'system_alarm_stop_channel'  # Alarm stop
```

**Line 94-95: Listen cả 2 channels**

```python
cur.execute(f"LISTEN {self.channel_name};")
cur.execute(f"LISTEN {self.stop_channel_name};")
```

**Line 155-166: Check channel trong \_handle_notification()**

```python
# Check which channel sent the notification
if notify.channel == self.stop_channel_name:
    # Stop alarm request
    logger.info(f"🔇 STOP ALARM REQUEST received")
    logger.info(f"   Event ID: {event_id}")
    logger.info(f"   Reason: {message}")
    self._process_alarm_stop_sync(data)
    return
```

**Line 280-310: Thêm hàm \_process_alarm_stop_sync()**

```python
def _process_alarm_stop_sync(self, event_data: Dict[str, Any]):
    """Xử lý alarm stop request (synchronous)"""
    try:
        event_id = str(event_data.get('event_id', ''))
        reason = event_data.get('message', 'Unknown reason')

        logger.info(f"🔇 Processing ALARM STOP: {event_id}")
        logger.info(f"   Reason: {reason}")

        # Stop alarm
        import asyncio
        stop_result = asyncio.run(audio_alert_service.stop_alarm())

        if stop_result['success']:
            logger.info("✅ ✅ ✅ ALARM STOPPED SUCCESSFULLY! ✅ ✅ ✅")
        else:
            logger.warning(f"⚠️ No alarm was playing")
    except Exception as e:
        logger.error(f"❌ Error stopping alarm: {e}")
```

**Kết quả:**

```
User bấm "Dismiss" button →
UPDATE lifecycle_state = 'DISMISSED' →
Database trigger fires →
NOTIFY 'system_alarm_stop_channel' →
Handler receives notification →
🔇 Alarm STOPS!
```

---

### **3️⃣ AUTO-STOP SAU 10 GIÂY**

**Logic:** Alarm tự động tắt sau 10 giây (safety fallback - tránh alarm kêu vô hạn)

#### **File: src/infrastructure/services/audio_alert_service.py**

**Line 213-215: Enable auto-stop (pygame backend)**

```python
# Schedule auto-stop after 10 seconds
import asyncio
asyncio.create_task(self._auto_stop_after_duration())
```

**Line 226-228: Enable auto-stop (pydub backend)**

```python
# Schedule auto-stop after 10 seconds
import asyncio
asyncio.create_task(self._auto_stop_after_duration())
```

**Line 250-260: Auto-stop function**

```python
async def _auto_stop_after_duration(self):
    """Tự động dừng sau duration"""
    import asyncio
    try:
        await asyncio.sleep(self.alert_duration)  # Default: 10 seconds
        if self.is_playing:
            await self.stop_alarm()
            logger.info(f"⏰ Auto-stopped alarm after {self.alert_duration}s")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in auto-stop: {e}")
```

**Configuration (.env):**

```bash
ALERT_DURATION_SECONDS=10  # Auto-stop after 10 seconds
```

**Kết quả:**

```
Alarm starts → Plays for 10s → Auto-stop 🔇
Priority: Auto-stop có priority thấp nhất
         (Nếu có điều kiện khác trigger trước 10s thì stop ngay)
```

---

### **4️⃣ MANUAL STOP**

**Logic:** Stop alarm trực tiếp qua code hoặc keyboard shortcut

#### **Direct Call:**

```python
import asyncio
from infrastructure.services.audio_alert_service import audio_alert_service

# Stop alarm
result = asyncio.run(audio_alert_service.stop_alarm())
```

#### **Keyboard Shortcut (có thể thêm vào main.py):**

```python
# Press 's' key to stop alarm
if key == ord('s') and audio_alert_service.is_playing:
    asyncio.run(audio_alert_service.stop_alarm())
    print("🔇 Alarm stopped manually")
```

---

## 🔄 **FULL FLOW**

### **Scenario 1: Detect >= 2 People**

```
1. Fall detected (1 person) → Alarm starts 🔊
2. Person still on ground → Alarm continues 🔊
3. Detect 2 people (someone came) → Alarm STOPS 🔇
```

### **Scenario 2: User Dismiss**

```
1. Fall detected → Database (ALARM_ACTIVATED) → Alarm starts 🔊
2. User bấm "Dismiss" button → UPDATE lifecycle_state='DISMISSED'
3. Database trigger fires → NOTIFY stop channel
4. Handler receives → Alarm STOPS 🔇
```

### **Scenario 3: System Resolve**

```
1. Seizure detected → Alarm starts 🔊
2. Admin verify & resolve → UPDATE lifecycle_state='RESOLVED'
3. Database trigger fires → NOTIFY stop channel
4. Handler receives → Alarm STOPS 🔇
```

### **Scenario 4: Auto-Stop Timeout**

```
1. Fall detected → Alarm starts 🔊
2. No one dismisses → Alarm continues 🔊
3. After 10 seconds → Auto-stop 🔇
4. Event remains ALARM_ACTIVATED (for manual review)
```

---

## 📊 **LIFECYCLE STATES FLOW**

```
NOTIFIED           ← Detection mới
    ↓
ALARM_ACTIVATED    ← Alarm đang kêu 🔊
    ↓ (Trigger stop conditions)
    ├─ DISMISSED   ← User bấm dismiss → Stop alarm 🔇
    ├─ RESOLVED    ← Admin resolve → Stop alarm 🔇
    ├─ CANCELED    ← System cancel → Stop alarm 🔇
    └─ ACKED       ← Acknowledged → Stop alarm 🔇
```

**Bất kỳ transition nào TỪ `ALARM_ACTIVATED` → Stop alarm!**

---

## 🧪 **TESTING**

### **Test 1: Stop khi detect >= 2 người**

```bash
# Run main.py
python src/main.py

# Step 1: Trigger fall detection (1 person)
# Press 'e' → Create fall event → Alarm plays 🔊

# Step 2: Đứng thêm 1 người vào camera
# → Detect 2 people → Alarm STOPS 🔇

# Expected log:
# 👥 Detected 2 people - STOPPING ALARM (safety check)
# ✅ Emergency alarm stopped
```

### **Test 2: Stop khi lifecycle_state thay đổi**

```bash
# Terminal 1: Run main.py
python src/main.py

# Terminal 2: Trigger alarm
cd examples
python trigger_alarm_test.py
# → Select event → Alarm plays 🔊

# Terminal 3: Stop alarm qua database
# Supabase SQL Editor hoặc psql:
UPDATE event_detections
SET
    lifecycle_state = 'DISMISSED',
    dismissed_at = NOW()
WHERE lifecycle_state = 'ALARM_ACTIVATED'
LIMIT 1;

# Expected log in Terminal 1:
# 🔔 NOTIFICATION RECEIVED!
# 🔇 STOP ALARM REQUEST received
# ✅ ✅ ✅ ALARM STOPPED SUCCESSFULLY! ✅ ✅ ✅
```

### **Test 3: Verify no auto-stop**

```bash
# Run main.py
python src/main.py

# Trigger alarm
cd examples
python trigger_alarm_test.py

# Wait 10 seconds → Alarm KHÔNG tự động tắt
# Wait 30 seconds → Alarm VẪN kêu
# → Must stop manually or via conditions
```

---

## 🗄️ **DATABASE SETUP**

### **Run SQL Script:**

```bash
# Option 1: Supabase SQL Editor
# 1. Open Supabase Dashboard
# 2. SQL Editor
# 3. Copy paste database_triggers_alarm_stop.sql
# 4. Run

# Option 2: psql command line
psql -h <supabase_host> -U postgres -d postgres -f database_triggers_alarm_stop.sql
```

### **Verify Triggers:**

```sql
SELECT
    trigger_name,
    event_manipulation,
    action_statement
FROM information_schema.triggers
WHERE event_object_table = 'event_detections'
  AND trigger_name LIKE '%alarm%';

-- Expected:
-- 1. alarm_activation_trigger - Start alarm
-- 2. alarm_stop_trigger - Stop alarm
```

---

## 📝 **FILES MODIFIED**

1. ✅ `src/main.py`

   - Line 328-338: Multi-camera stop logic (>= 2 people)
   - Line 610-620: Single-camera stop logic (>= 2 people)

2. ✅ `src/infrastructure/services/audio_alert_service.py`

   - Line 213-215: Enabled auto-stop after 10s (pygame)
   - Line 226-228: Enabled auto-stop after 10s (pydub)
   - Line 250-260: Auto-stop implementation

3. ✅ `src/infrastructure/services/emergency_alarm_handler_psycopg.py`

   - Line 42-43: Added stop_channel_name
   - Line 94-95: Listen both channels
   - Line 155-166: Check stop channel
   - Line 280-310: Added \_process_alarm_stop_sync()

4. ✅ `database_triggers_alarm_stop.sql` (NEW)
   - Complete SQL script for alarm stop trigger

---

## ⚙️ **CONFIGURATION**

### **.env**

```bash
ALERT_DURATION_SECONDS=10   # Auto-stop after 10 seconds (default: 30)
EMERGENCY_ALERT_VOLUME=1.0  # Volume 100%
AUDIO_ALERT_ENABLED=true    # Enable alarm
```

### **Database Channels:**

```
system_alarm_channel       → Start alarm (ALARM_ACTIVATED)
system_alarm_stop_channel  → Stop alarm (lifecycle_state change)
```

---

## 🎯 **SUMMARY**

**Stop Conditions (4 ways):**

1. ✅ **>= 2 people detected** → Auto-stop (safety - có người giúp đỡ)
2. ✅ **lifecycle_state changes** → Trigger stop (user dismiss/admin resolve)
3. ✅ **10s timeout** → Auto-stop (safety fallback)
4. ✅ **Manual stop** → Direct call hoặc keyboard shortcut

**Alarm Duration:**

- **Default:** Max 10 seconds (auto-stop)
- **With conditions:** Có thể stop sớm hơn (khi detect >= 2 người hoặc user dismiss)

**Priority (Thứ tự ưu tiên):**

1. **Highest:** >= 2 people detected (immediate stop)
2. **High:** lifecycle_state change (user action)
3. **Medium:** Manual stop
4. **Lowest:** 10s timeout (fallback)

**Benefits:**

- ✅ Prevent false alarms (2+ people = help arrived)
- ✅ User can dismiss (mobile button)
- ✅ Admin can resolve (manual intervention)
- ✅ Safety timeout (tránh alarm kêu vô hạn nếu có lỗi)
- ✅ Flexible control (nhiều cách stop)

**Recommendation:**

- ✅ Keep all 4 mechanisms (tốt nhất)
- ✅ Test thoroughly in production
- ✅ Monitor alarm duration in logs
- ✅ Adjust timeout in .env if needed (ALERT_DURATION_SECONDS)
