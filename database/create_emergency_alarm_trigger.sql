-- =====================================================
-- EMERGENCY ALARM TRIGGER - PostgreSQL LISTEN/NOTIFY
-- Tự động gửi notification khi:
-- 1. INSERT event với event_type = 'manual_emergency'
-- 2. UPDATE event với lifecycle_state = 'ALARM_ACTIVATED'
-- =====================================================

-- Tạo function để gửi notification
CREATE OR REPLACE FUNCTION notify_emergency_alarm()
RETURNS TRIGGER AS $$
DECLARE
    notification JSON;
BEGIN
    -- Xác định loại event
    IF (TG_OP = 'INSERT' AND NEW.event_type = 'manual_emergency') THEN
        -- Event mới: manual_emergency
        notification = json_build_object(
            'event_type', 'manual_emergency',
            'event_id', NEW.event_id,
            'user_id', NEW.user_id,
            'camera_id', NEW.camera_id,
            'snapshot_id', NEW.snapshot_id,
            'event_description', NEW.event_description,
            'detected_at', NEW.detected_at,
            'message', 'New manual emergency request',
            'operation', 'INSERT'
        );
        
        -- Gửi notification
        PERFORM pg_notify('emergency_alarm_channel', notification::text);
        
    ELSIF (TG_OP = 'UPDATE' AND NEW.lifecycle_state = 'ALARM_ACTIVATED' AND OLD.lifecycle_state != 'ALARM_ACTIVATED') THEN
        -- Event cũ được activate alarm
        notification = json_build_object(
            'event_type', 'alarm_activated',
            'event_id', NEW.event_id,
            'user_id', NEW.user_id,
            'camera_id', NEW.camera_id,
            'snapshot_id', NEW.snapshot_id,
            'event_description', NEW.event_description,
            'old_lifecycle_state', OLD.lifecycle_state,
            'new_lifecycle_state', NEW.lifecycle_state,
            'detected_at', NEW.detected_at,
            'message', 'Alarm activated from mobile app',
            'operation', 'UPDATE'
        );
        
        -- Gửi notification
        PERFORM pg_notify('emergency_alarm_channel', notification::text);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Xóa trigger cũ nếu có
DROP TRIGGER IF EXISTS emergency_alarm_trigger ON event_detections;

-- Tạo trigger mới
CREATE TRIGGER emergency_alarm_trigger
    AFTER INSERT OR UPDATE ON event_detections
    FOR EACH ROW
    EXECUTE FUNCTION notify_emergency_alarm();

-- Test trigger
COMMENT ON TRIGGER emergency_alarm_trigger ON event_detections IS 
'Trigger to send PostgreSQL notifications for emergency alarms via LISTEN/NOTIFY';

-- Kiểm tra trigger đã được tạo
SELECT 
    trigger_name, 
    event_manipulation, 
    event_object_table,
    action_statement
FROM information_schema.triggers
WHERE trigger_name = 'emergency_alarm_trigger';

-- Hướng dẫn test
COMMENT ON FUNCTION notify_emergency_alarm() IS 
'To test: 
1. In one terminal: LISTEN emergency_alarm_channel;
2. In another terminal: 
   INSERT INTO event_detections (...) VALUES (...) -- with event_type=manual_emergency
   OR
   UPDATE event_detections SET lifecycle_state=ALARM_ACTIVATED WHERE ...
3. Check terminal 1 for notifications';
