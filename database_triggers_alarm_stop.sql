-- ============================================================================
-- DATABASE TRIGGERS FOR ALARM STOP
-- Run this in Supabase SQL Editor
-- ============================================================================

-- ============================================================================
-- 1. TRIGGER: Stop Alarm khi lifecycle_state thay đổi từ ALARM_ACTIVATED
-- ============================================================================

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
        
        -- Log để debug
        RAISE NOTICE 'Alarm stop notification sent for event %', NEW.event_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Tạo trigger
DROP TRIGGER IF EXISTS alarm_stop_trigger ON event_detections;

CREATE TRIGGER alarm_stop_trigger
AFTER UPDATE ON event_detections
FOR EACH ROW
EXECUTE FUNCTION notify_alarm_stop_trigger();

-- ============================================================================
-- 2. TEST TRIGGER
-- ============================================================================

-- Test bằng cách update lifecycle_state từ ALARM_ACTIVATED
-- Giả sử có event với lifecycle_state = 'ALARM_ACTIVATED'

-- Step 1: Tìm event đang ALARM_ACTIVATED
SELECT 
    event_id,
    event_type,
    lifecycle_state,
    detected_at
FROM event_detections
WHERE lifecycle_state = 'ALARM_ACTIVATED'
ORDER BY detected_at DESC
LIMIT 5;

-- Step 2: Update sang state khác (ví dụ: DISMISSED)
-- UNCOMMENT để test:
/*
UPDATE event_detections
SET 
    lifecycle_state = 'DISMISSED',
    dismissed_at = NOW(),
    is_canceled = TRUE,
    notes = 'User dismissed alarm from mobile'
WHERE lifecycle_state = 'ALARM_ACTIVATED'
  AND event_id = '<PASTE_EVENT_ID_HERE>'
LIMIT 1;
*/

-- Step 3: Check notification (xem log trong main.py terminal)
-- Handler sẽ nhận notification và stop alarm

-- ============================================================================
-- 3. VERIFY TRIGGERS
-- ============================================================================

-- Kiểm tra cả 2 triggers đã được tạo
SELECT 
    trigger_name,
    event_manipulation,
    event_object_table,
    action_timing,
    action_statement
FROM information_schema.triggers
WHERE event_object_table = 'event_detections'
  AND trigger_name LIKE '%alarm%'
ORDER BY trigger_name;

-- Expected results:
-- 1. alarm_activation_trigger - Kích hoạt alarm khi chuyển sang ALARM_ACTIVATED
-- 2. alarm_stop_trigger - Dừng alarm khi chuyển TỪ ALARM_ACTIVATED

-- ============================================================================
-- 4. CLEANUP (Nếu muốn xóa trigger)
-- ============================================================================

-- UNCOMMENT để xóa:
/*
DROP TRIGGER IF EXISTS alarm_stop_trigger ON event_detections;
DROP FUNCTION IF EXISTS notify_alarm_stop_trigger();
*/

-- ============================================================================
-- 5. USAGE EXAMPLES
-- ============================================================================

-- Example 1: User bấm "Dismiss" trên mobile
UPDATE event_detections
SET 
    lifecycle_state = 'DISMISSED',
    dismissed_at = NOW(),
    is_canceled = TRUE
WHERE event_id = '<event_id>';
-- → Trigger fires → Stop alarm

-- Example 2: Admin resolve event
UPDATE event_detections
SET 
    lifecycle_state = 'RESOLVED',
    verified_at = NOW()
WHERE event_id = '<event_id>';
-- → Trigger fires → Stop alarm

-- Example 3: System cancel event
UPDATE event_detections
SET 
    lifecycle_state = 'CANCELED',
    is_canceled = TRUE
WHERE event_id = '<event_id>';
-- → Trigger fires → Stop alarm

-- ============================================================================
-- 6. MONITORING
-- ============================================================================

-- Check recent lifecycle_state changes
SELECT 
    event_id,
    event_type,
    lifecycle_state,
    dismissed_at,
    is_canceled,
    detected_at,
    last_action_at
FROM event_detections
WHERE lifecycle_state IN ('DISMISSED', 'RESOLVED', 'CANCELED', 'ALARM_ACTIVATED')
ORDER BY last_action_at DESC
LIMIT 20;

-- Count by lifecycle_state
SELECT 
    lifecycle_state,
    COUNT(*) as count
FROM event_detections
GROUP BY lifecycle_state
ORDER BY count DESC;
