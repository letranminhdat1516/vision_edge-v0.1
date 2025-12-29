# Alarm Runtime – PostgreSQL LISTEN/NOTIFY + Audio

- Module: [src/infrastructure/services/emergency_alarm_handler_psycopg.py](src/infrastructure/services/emergency_alarm_handler_psycopg.py) → class `EmergencyAlarmHandlerPsycopg`

## Channels
- `system_alarm_trigger_channel`, `system_alarm_stop_channel` on direct port 5432.

## Deduplication
- `processed_events` set; cleanup periodically.

## Trigger Processing
- `TRIGGER_ALARM` plays alarm indefinitely (duration=0) until stopped.

## Stop Processing
- `stop_alarm`, then update event `lifecycle_state → RESOLVED` unless it’s an AUTOCALLED transition from DB trigger (ignored for safety).

Notes (VI): Lắng nghe NOTIFY thời gian thực, bật/tắt còi, cập nhật trạng thái sự kiện an toàn.
