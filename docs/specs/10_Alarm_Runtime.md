# Alarm Runtime – PostgreSQL LISTEN/NOTIFY + Audio

- **Module**: [src/infrastructure/services/emergency_alarm_handler_psycopg.py](src/infrastructure/services/emergency_alarm_handler_psycopg.py) → class `EmergencyAlarmHandlerPsycopg`
- **Mục đích**: Real-time alarm handling sử dụng PostgreSQL native LISTEN/NOTIFY, độ trễ < 50ms.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PostgreSQL Database                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  NOTIFY system_alarm_trigger_channel, '{...}'            │    │
│  │  NOTIFY system_alarm_stop_channel, '{...}'               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ PostgreSQL Wire Protocol
                              │ (Port 5432 - DIRECT, not pooler!)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               EmergencyAlarmHandlerPsycopg                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LISTEN system_alarm_trigger_channel                    │    │
│  │  LISTEN system_alarm_stop_channel                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Alert Service                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  play_emergency_alarm() / stop_alarm()                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Constructor & Connection

```python
def __init__(self, postgresql_service=None):
    # Deduplication
    self.processed_events = set()
    self.last_cleanup_time = datetime.now()

    # CRITICAL: Direct connection (port 5432)
    # Pooler (port 6543) does NOT support LISTEN/NOTIFY!
    self.database_url = f"postgresql://{user}:{password}@{host}:5432/{db_name}"

    # Channel names
    self.trigger_channel_name = 'system_alarm_trigger_channel'
    self.stop_channel_name = 'system_alarm_stop_channel'
```

### Why Direct Connection?

| Connection Type    | Port | LISTEN/NOTIFY    | Use Case         |
| ------------------ | ---- | ---------------- | ---------------- |
| **Direct**         | 5432 | ✅ Supported     | Real-time events |
| Pooler (PgBouncer) | 6543 | ❌ Not supported | Web APIs         |

---

## 3. Channel System

### Channels

| Channel                        | Purpose     | Payload                                                |
| ------------------------------ | ----------- | ------------------------------------------------------ |
| `system_alarm_trigger_channel` | Start alarm | `{event_id, user_id, camera_id, action, triggered_by}` |
| `system_alarm_stop_channel`    | Stop alarm  | `{event_id, reason, stopped_by}`                       |

### Trigger Payload Example

```json
{
  "event_id": "abc-123-xyz",
  "user_id": "user-456",
  "camera_id": "cam-789",
  "action": "TRIGGER_ALARM",
  "triggered_by": "api"
}
```

### Stop Payload Example

```json
{
  "event_id": "abc-123-xyz",
  "reason": "User acknowledged alert",
  "stopped_by": "mobile_app"
}
```

---

## 4. Listener Implementation

### Main Loop (Separate Thread)

```python
def _listen_loop(self):
    """Runs in dedicated thread"""

    while True:
        try:
            # Connect
            self.listen_conn = psycopg.connect(self.database_url, autocommit=True)

            with self.listen_conn.cursor() as cur:
                # Subscribe to channels
                cur.execute(f"LISTEN {self.trigger_channel_name};")
                cur.execute(f"LISTEN {self.stop_channel_name};")

                # Poll loop
                while self.is_running:
                    gen = self.listen_conn.notifies(timeout=1.0)

                    for notify in gen:
                        self._handle_notification(notify)

                    time.sleep(0.01)  # Prevent tight loop

        except psycopg.OperationalError:
            # Connection lost, retry
            time.sleep(5)

        except psycopg.InterfaceError:
            # Interface error, retry
            time.sleep(5)
```

### Notification Structure

```python
notify.channel   # "system_alarm_trigger_channel"
notify.payload   # '{"event_id": "abc-123", ...}'  (JSON string)
```

---

## 5. Notification Processing

### Handler Flow

```python
def _handle_notification(self, notify):
    # Parse JSON
    data = json.loads(notify.payload)
    event_id = data.get('event_id')

    # Check channel
    if notify.channel == self.stop_channel_name:
        self._process_alarm_stop_sync(data)
        return

    if notify.channel == self.trigger_channel_name:
        # Deduplication
        if event_id in self.processed_events:
            return  # Already processed

        if data.get('action') == 'TRIGGER_ALARM':
            self._process_alarm_trigger_sync(data)
```

---

## 6. Trigger Processing

```python
def _process_alarm_trigger_sync(self, event_data):
    """Process TRIGGER_ALARM request"""

    event_id = event_data.get('event_id')

    # Track for deduplication
    self.processed_events.add(event_id)

    # Play alarm (INDEFINITE - no auto-stop!)
    alarm_result = audio_alert_service.play_emergency_alarm(
        user_id=event_id,      # Use event_id for tracking
        triggered_by=event_data.get('triggered_by'),
        duration=0             # 0 = INFINITE, until stop command
    )

    if alarm_result['success']:
        logger.info("✅ ALARM PLAYING!")
    else:
        logger.error(f"❌ ALARM FAILED: {alarm_result['message']}")
```

### Key Point: `duration=0`

- Alarm plays **indefinitely** until explicit stop
- No auto-timeout
- Ensures alarm continues until situation resolved

---

## 7. Stop Processing

```python
def _process_alarm_stop_sync(self, event_data):
    """Process STOP_ALARM request"""

    event_id = event_data.get('event_id')
    reason = event_data.get('reason')
    stopped_by = event_data.get('stopped_by')

    # ⚠️ IMPORTANT: Ignore auto-stop for AUTOCALLED events
    old_lifecycle = event_data.get('old_lifecycle_state')
    new_lifecycle = event_data.get('new_lifecycle_state')

    if old_lifecycle == 'ALARM_ACTIVATED' and new_lifecycle == 'AUTOCALLED':
        logger.warning("Ignoring stop for AUTOCALLED - keep alarm running!")
        return

    # Step 1: Stop audio
    stop_result = audio_alert_service.stop_alarm(event_id=event_id)

    # Step 2: Update event → RESOLVED
    if event_id:
        self._update_event_to_resolved(event_id, reason, stopped_by)
```

### AUTOCALLED Protection

```
ALARM_ACTIVATED → AUTOCALLED transition:
- Emergency services have been contacted
- Alarm MUST continue until manual stop
- Database trigger fires NOTIFY but we IGNORE it
```

---

## 8. Event Resolution

```python
def _update_event_to_resolved(self, event_id, reason, stopped_by):
    """Update event lifecycle to RESOLVED"""

    # Check current state
    cursor.execute("""
        SELECT lifecycle_state, status
        FROM event_detections
        WHERE event_id = %s
    """, (event_id,))

    result = cursor.fetchone()
    current_state = result['lifecycle_state']

    # Only update if in alarm state
    if current_state in ['ALARM_ACTIVATED', 'AUTOCALLED']:
        cursor.execute("""
            UPDATE event_detections
            SET
                lifecycle_state = 'RESOLVED',
                last_action_at = NOW(),
                notes = notes || 'Resolved: Stopped by ' || %s || ' (' || %s || ')'
            WHERE event_id = %s
              AND lifecycle_state IN ('ALARM_ACTIVATED', 'AUTOCALLED')
        """, (stopped_by, reason, event_id))

        conn.commit()
```

---

## 9. Deduplication & Cache Cleanup

### Processed Events Set

```python
self.processed_events = set()  # Track processed event_ids
```

### Cleanup (Prevent Memory Leak)

```python
def _cleanup_processed_cache(self):
    """Called every 5 minutes"""
    if len(self.processed_events) > 1000:
        self.processed_events.clear()

    self.last_cleanup_time = datetime.now()
```

---

## 10. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  API Call: POST /alarms/trigger                                  │
│  Body: {event_id: "abc", user_id: "xyz"}                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PostgreSQL: NOTIFY system_alarm_trigger_channel                 │
│  Payload: '{"event_id":"abc","action":"TRIGGER_ALARM",...}'      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ < 50ms latency
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EmergencyAlarmHandler._listen_loop()                            │
│  Receives notification via psycopg                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  _handle_notification()                                          │
│  - Parse JSON                                                    │
│  - Check deduplication                                           │
│  - Route to trigger/stop handler                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  TRIGGER_ALARM          │     │  STOP_ALARM             │
│  - Add to processed_set │     │  - Check AUTOCALLED     │
│  - Play alarm (∞)       │     │  - Stop audio           │
│                         │     │  - Update → RESOLVED    │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 11. Lifecycle State Transitions

```
┌─────────────────────────────────────────────────────────────────┐
│  DETECTED → ALARM_ACTIVATED                                      │
│       │                                                          │
│       │ (30s timeout, no acknowledgment)                         │
│       ▼                                                          │
│  ALARM_ACTIVATED → AUTOCALLED                                    │
│       │         (Emergency services contacted)                   │
│       │                                                          │
│       │ (Manual stop via API/App)                                │
│       ▼                                                          │
│  AUTOCALLED → RESOLVED                                           │
│       │                                                          │
│  OR   │ (User acknowledges in app)                               │
│       ▼                                                          │
│  ALARM_ACTIVATED → ACKNOWLEDGED → RESOLVED                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Singleton Pattern

```python
# Module-level singleton
emergency_alarm_handler = EmergencyAlarmHandlerPsycopg()

# Usage:
from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
await emergency_alarm_handler.start_listening()
```

---

## 13. Error Handling

### Connection Errors

```python
except psycopg.OperationalError as e:
    logger.error(f"Connection lost: {e}")
    time.sleep(5)
    # Loop continues, will reconnect

except psycopg.InterfaceError as e:
    logger.error(f"Interface error: {e}")
    time.sleep(5)
```

### Graceful Shutdown

```python
def stop(self):
    self.is_running = False

    if self.listen_conn:
        self.listen_conn.close()

    logger.info("Emergency Alarm Handler stopped")
```

---

## Notes (VI)

Sử dụng PostgreSQL LISTEN/NOTIFY để nhận events real-time (độ trễ < 50ms). **BẮT BUỘC** dùng direct connection (port 5432), pooler không hỗ trợ LISTEN. Channels: `trigger_channel` để bật còi, `stop_channel` để tắt. Alarm chạy vô hạn (duration=0) cho đến khi có stop command. AUTOCALLED events được bảo vệ - không tự động tắt khi emergency services đã được gọi.
