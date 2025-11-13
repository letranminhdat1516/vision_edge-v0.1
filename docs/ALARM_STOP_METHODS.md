# 🔇 CÁCH TẮT ALARM - STOP ALARM METHODS

## 🎯 **TÓM TẮT**

Có **3 cách** để tắt alarm đang phát:

1. ✅ **Auto Stop** - Tự động tắt sau thời gian (mặc định 10s)
2. ✅ **Manual Stop qua API** - User bấm nút "Dismiss" trên mobile
3. ✅ **Direct Stop** - Gọi trực tiếp `audio_alert_service.stop_alarm()`

---

## 🔊 **1. AUTO STOP (Đang hoạt động)**

### **Mechanism:**

```python
# src/infrastructure/services/audio_alert_service.py

# Line 242: Play alarm với duration
alarm_result = asyncio.run(audio_alert_service.play_emergency_alarm(
    user_id=user_id,
    triggered_by='alarm_activation',
    duration=10  # ← 10 giây
))

# Line 247: Auto-schedule stop
asyncio.create_task(self._auto_stop_after_duration())

# Line 252-257: Auto stop function
async def _auto_stop_after_duration(self):
    await asyncio.sleep(self.alert_duration)  # Wait 10s
    if self.is_playing:
        await self.stop_alarm()
        logger.info(f"⏰ Auto-stopped alarm after {self.alert_duration}s")
```

### **Config:**

```python
# .env
ALERT_DURATION_SECONDS=10  # Mặc định 10 giây

# Code
self.alert_duration = int(os.getenv('ALERT_DURATION_SECONDS', '30'))
```

**Kết quả:** Alarm tự động tắt sau 10 giây

---

## 📱 **2. MANUAL STOP QUA MOBILE APP (Recommend)**

### **Flow:**

```
User bấm "Dismiss Alarm" button
    ↓
Mobile app gửi API request
    ↓
UPDATE event_detections
SET lifecycle_state = 'DISMISSED',
    dismissed_at = NOW(),
    is_canceled = TRUE
    ↓
Database Trigger fires (cần tạo mới)
    ↓
NOTIFY 'system_alarm_stop_channel'
    ↓
Alarm Handler nhận notification
    ↓
Call audio_alert_service.stop_alarm()
    ↓
🔇 Alarm stopped!
```

### **Implementation (Cần thêm):**

#### **Step 1: Tạo Database Trigger**

```sql
-- Supabase SQL Editor
CREATE OR REPLACE FUNCTION notify_alarm_stop_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Trigger khi lifecycle_state chuyển sang DISMISSED
    IF NEW.lifecycle_state = 'DISMISSED'
       AND OLD.lifecycle_state != 'DISMISSED' THEN

        PERFORM pg_notify('system_alarm_stop_channel', json_build_object(
            'event_id', NEW.event_id,
            'user_id', NEW.user_id,
            'action', 'STOP_ALARM',
            'message', 'User dismissed alarm from mobile'
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

#### **Step 2: Update Alarm Handler**

```python
# src/infrastructure/services/emergency_alarm_handler_psycopg.py

# Line 42: Add stop channel
self.stop_channel_name = 'system_alarm_stop_channel'

# Line 93: Listen to both channels
cur.execute(f"LISTEN {self.channel_name};")
cur.execute(f"LISTEN {self.stop_channel_name};")  # ← ADD THIS

# Line 145-172: Modify _handle_notification()
def _handle_notification(self, notify):
    try:
        data = json.loads(notify.payload)

        # Check channel
        if notify.channel == 'system_alarm_stop_channel':
            # Handle STOP request
            self._process_alarm_stop_sync(data)

        elif notify.channel == 'system_alarm_channel':
            # Handle ALARM_ACTIVATED (existing)
            state = data.get('state')
            if state == 'ALARM_ACTIVATED':
                self._process_alarm_activated_sync(data)
    except Exception as e:
        logger.error(f"Error: {e}")

# ADD NEW METHOD:
def _process_alarm_stop_sync(self, event_data: Dict[str, Any]):
    """Stop alarm when user dismisses"""
    try:
        event_id = str(event_data.get('event_id', ''))

        logger.info(f"🔇 Processing ALARM STOP: {event_id}")

        # Stop alarm
        import asyncio
        stop_result = asyncio.run(audio_alert_service.stop_alarm())

        if stop_result['success']:
            logger.info("✅ ALARM STOPPED BY USER")
            logger.info(f"   Event: {event_id[:8]}...")
            logger.info(f"   Action: User dismissed from mobile")
        else:
            logger.warning(f"⚠️ No alarm was playing: {stop_result['message']}")

    except Exception as e:
        logger.error(f"Error stopping alarm: {e}")
```

#### **Step 3: Mobile App Button**

```dart
// Mobile app (Flutter/React Native)

// Dismiss button
ElevatedButton(
  child: Text('Dismiss Alarm 🔇'),
  onPressed: () async {
    // Call API
    await supabase
      .from('event_detections')
      .update({
        'lifecycle_state': 'DISMISSED',
        'dismissed_at': DateTime.now().toIso8601String(),
        'is_canceled': true,
      })
      .eq('event_id', eventId);

    // Show toast
    showToast('Alarm dismissed');
  },
)
```

---

## 💻 **3. DIRECT STOP (Debug/Testing)**

### **Method 1: Python Script**

```python
# examples/stop_alarm.py
import asyncio
from infrastructure.services.audio_alert_service import audio_alert_service

async def main():
    print("🔇 Stopping alarm...")
    result = await audio_alert_service.stop_alarm()

    if result['success']:
        print("✅ Alarm stopped!")
    else:
        print(f"❌ {result['message']}")

if __name__ == '__main__':
    asyncio.run(main())
```

### **Method 2: Terminal (khi main.py đang chạy)**

```bash
# Option A: Press key in main.py terminal
# Thêm key handler vào main.py:
# Press 's' key → stop alarm

# Option B: Kill process
pkill -f emergency_alarm
# Hoặc
Ctrl+C  # Stop main.py → stop alarm
```

### **Method 3: Database Manual Update**

```sql
-- Supabase SQL Editor
UPDATE event_detections
SET
    lifecycle_state = 'DISMISSED',
    dismissed_at = NOW(),
    is_canceled = TRUE
WHERE lifecycle_state = 'ALARM_ACTIVATED'
  AND dismissed_at IS NULL
ORDER BY detected_at DESC
LIMIT 1;
```

---

## 🔧 **STOP ALARM API**

### **Function:**

```python
# src/infrastructure/services/audio_alert_service.py
# Line 263-291

async def stop_alarm(self) -> Dict[str, Any]:
    """
    Dừng báo động

    Returns:
        Dict với status và message
    """
    if not self.is_playing:
        return {"success": False, "message": "No alarm is playing"}

    try:
        # Stop pygame mixer
        if self.audio_backend == 'pygame':
            import pygame
            pygame.mixer.stop()  # ← Stop tất cả sounds

        # Reset state
        self.is_playing = False
        self.current_sound = None

        logger.info("✅ Emergency alarm stopped")

        return {
            "success": True,
            "message": "Alarm stopped successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to stop alarm: {e}")
        return {"success": False, "message": str(e)}
```

### **Usage:**

```python
# Import
from infrastructure.services.audio_alert_service import audio_alert_service

# Async context
result = await audio_alert_service.stop_alarm()

# Sync context (use asyncio.run)
import asyncio
result = asyncio.run(audio_alert_service.stop_alarm())

# Check result
if result['success']:
    print("Alarm stopped!")
else:
    print(f"Failed: {result['message']}")
```

---

## 📊 **LIFECYCLE STATES**

```
NOTIFIED          ← Event mới tạo
    ↓
ALARM_ACTIVATED   ← Alarm đang phát
    ↓ (3 options)
    ├─ DISMISSED  ← User bấm Dismiss (stop alarm)
    ├─ ACKED      ← Auto-stop sau duration
    └─ CANCELED   ← Error hoặc manual cancel
```

**Fields liên quan:**

```sql
lifecycle_state VARCHAR(27)  -- State của event
dismissed_at TIMESTAMP        -- Thời điểm dismiss
is_canceled BOOLEAN           -- Có bị cancel không
acknowledged_at TIMESTAMP     -- Thời điểm acknowledge
```

---

## ⚙️ **CONFIG DURATION**

### **.env:**

```bash
ALERT_DURATION_SECONDS=10     # Auto-stop sau 10 giây (mặc định 30)
EMERGENCY_ALERT_VOLUME=1.0    # Volume 100%
AUDIO_ALERT_ENABLED=true      # Enable/disable alarm
```

### **Override trong code:**

```python
# Play với custom duration
alarm_result = await audio_alert_service.play_emergency_alarm(
    user_id=user_id,
    triggered_by='alarm_activation',
    duration=5  # ← Override: chỉ 5 giây thay vì 10s
)
```

---

## 🧪 **TESTING**

### **Test Auto-Stop:**

```bash
# Terminal 1: Run main.py
cd src
python main.py

# Terminal 2: Trigger alarm
cd examples
python trigger_alarm_test.py

# Kết quả:
# - Alarm phát 10 giây
# - Tự động tắt
# - Log: "⏰ Auto-stopped alarm after 10s"
```

### **Test Manual Stop:**

```bash
# Terminal 1: main.py đang chạy và alarm đang phát

# Terminal 2: Run stop script
python examples/stop_alarm.py

# Kết quả:
# - Alarm dừng ngay lập tức
# - Log: "✅ Emergency alarm stopped"
```

### **Test Mobile Dismiss:**

```bash
# 1. Trigger alarm
python examples/trigger_alarm_test.py

# 2. Mở mobile app
# 3. Bấm "Dismiss" button
# 4. Check main.py terminal:
#    → "🔇 Processing ALARM STOP"
#    → "✅ ALARM STOPPED BY USER"
```

---

## 📝 **PRIORITY IMPLEMENTATION**

### **Hiện tại:**

✅ Auto-stop sau 10 giây - **HOẠT ĐỘNG**
✅ Direct stop qua script - **HOẠT ĐỘNG**
❌ Mobile dismiss button - **CHƯA CÓ**

### **Cần thêm (Recommend):**

1. **Database Trigger cho DISMISSED** (5 phút)

   - Tạo `notify_alarm_stop_trigger()`
   - Listen channel `system_alarm_stop_channel`

2. **Update Alarm Handler** (10 phút)

   - Thêm listen stop channel
   - Thêm `_process_alarm_stop_sync()`

3. **Mobile App Button** (15 phút)

   - Add "Dismiss Alarm" button
   - Call UPDATE API
   - Show feedback toast

4. **Test Integration** (10 phút)
   - Test full flow
   - Verify alarm stops
   - Check logs

**Total: ~40 phút để implement full mobile dismiss**

---

## 🎯 **KẾT LUẬN**

**Hiện tại có thể tắt alarm bằng:**

1. ✅ **Đợi 10 giây** - Auto-stop (mặc định)
2. ✅ **Run stop script** - Debug/testing
3. ❌ **Mobile button** - Chưa implement

**Recommendation:**

- Giữ auto-stop (an toàn, tránh alarm vô tận)
- Thêm mobile dismiss button (UX tốt hơn)
- Duration 10s là hợp lý cho emergency alarm

**Anh muốn em implement mobile dismiss button không? 🤔**
