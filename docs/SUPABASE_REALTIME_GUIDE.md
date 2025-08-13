# Supabase Realtime Integration Guide

## Tổng quan

Hệ thống Healthcare Monitoring đã được tích hợp với Supabase Realtime để cung cấp khả năng theo dõi sự kiện té ngã và co giật theo thời gian thực. Khi hệ thống phát hiện té ngã hoặc co giật, sự kiện sẽ được publish lên Supabase database và tự động notify tới tất cả clients đang lắng nghe.

## Tính năng chính

### 1. **Real-time Event Detection**
- Phát hiện té ngã (`fall_detection`) với confidence score và bounding boxes
- Phát hiện co giật (`seizure_detection`) với temporal analysis
- Tự động lưu snapshot và metadata

### 2. **Alert System**
- Tự động tạo alerts dựa trên độ nghiêm trọng của sự kiện
- Phân loại alert: `low`, `medium`, `high`, `critical`
- Tracking trạng thái alert: `active`, `acknowledged`, `resolved`

### 3. **Multi-client Support**
- Hỗ trợ nhiều client cùng lắng nghe realtime events
- JavaScript/Web client integration
- Python client integration  
- Webhook support cho external systems

## Cài đặt và Cấu hình

### 1. **Cài đặt Dependencies**
```bash
pip install supabase python-dotenv
```

### 2. **Cấu hình Environment Variables**
Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```

Cập nhật các giá trị trong `.env`:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here  
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

### 3. **Khởi tạo Database Schema**
Chạy SQL script trong `database/postgresql_schema.sql` trên Supabase dashboard để tạo các bảng cần thiết.

## Sử dụng

### 1. **Chạy Healthcare Pipeline với Supabase**
```bash
cd src
python main.py
```

Pipeline sẽ tự động:
- Kết nối tới Supabase
- Publish events khi phát hiện té ngã/co giật
- Lưu snapshots và metadata
- Tạo alerts tự động

### 2. **Chạy Realtime Client**
Mở terminal khác và chạy:
```bash
cd examples
python healthcare_realtime_client.py
```

Client sẽ:
- Lắng nghe realtime events từ Supabase
- Hiển thị thông tin chi tiết về các sự kiện
- Xử lý alerts theo mức độ nghiêm trọng

## JSON Format Specification

### Event Detection Format
```json
{
  "event_id": "uuid",
  "user_id": "uuid",
  "camera_id": "uuid",
  "room_id": "uuid", 
  "event_type": "fall_detection | seizure_detection",
  "confidence_score": 0.85,
  "bounding_boxes": [
    {
      "x": 100, "y": 150,
      "width": 200, "height": 300,
      "class": "person", "confidence": 0.95
    }
  ],
  "context_data": {
    "motion_level": 0.75,
    "detection_type": "direct | confirmation", 
    "confirmation_frames": 2,
    "frame_number": 1250
  },
  "detected_at": "2025-08-14T10:30:45.123Z"
}
```

### Alert Format
```json
{
  "alert_id": "uuid",
  "event_id": "uuid",
  "alert_type": "fall_detected | seizure_detected",
  "severity": "high | critical",
  "alert_message": "Fall detected with 85% confidence. Immediate attention required.",
  "alert_data": {
    "confidence": 0.85,
    "image_path": "/alerts/fall_detected_20250814_103045.jpg",
    "timestamp": "2025-08-14T10:30:45.123Z"
  },
  "status": "active"
}
```

Chi tiết đầy đủ xem trong: `docs/supabase_realtime_format.json`

## Integration Examples

### JavaScript/Web Client
```javascript
const { createClient } = require('@supabase/supabase-js');
const supabase = createClient('YOUR_SUPABASE_URL', 'YOUR_SUPABASE_ANON_KEY');

// Listen for new events
supabase
  .channel('healthcare_events')
  .on('postgres_changes', {
    event: 'INSERT',
    schema: 'public',
    table: 'event_detections'
  }, (payload) => {
    console.log('New event detected:', payload.new);
    handleHealthcareEvent(payload.new);
  })
  .subscribe();
```

### Python Client
```python
from service.supabase_realtime_service import realtime_service

def handle_event(event_data):
    event = event_data['new_data']
    print(f"New {event['event_type']}: {event['confidence_score']:.2%}")

realtime_service.subscribe_to_events(
    'event_detections', 
    'INSERT', 
    handle_event
)
```

### REST API Access
```bash
# Get recent events
curl -X GET "https://your-project.supabase.co/rest/v1/event_detections?order=created_at.desc&limit=10" \
  -H "Authorization: Bearer YOUR_SUPABASE_ANON_KEY"

# Get active alerts  
curl -X GET "https://your-project.supabase.co/rest/v1/alerts?status=eq.active" \
  -H "Authorization: Bearer YOUR_SUPABASE_ANON_KEY"
```

## Logs và Monitoring

### Event Logs
Khi pipeline phát hiện sự kiện, sẽ log:
```
🚨 FALL DETECTED! Confidence: 0.85 | Motion: 0.75 | Direct Detection
📊 Alert Level: HIGH | Emergency Type: Fall
📡 Fall event published to Supabase: 123e4567-e89b-12d3-a456-426614174000
```

### Realtime Client Logs
Client sẽ hiển thị:
```
============================================================
🔔 NEW HEALTHCARE EVENT #1
============================================================
Event ID: 123e4567-e89b-12d3-a456-426614174000
Type: fall_detection
Confidence: 85.00%
Status: detected
🆘 HIGH CONFIDENCE FALL - EMERGENCY RESPONSE REQUIRED!
```

## Troubleshooting

### 1. **Connection Issues**
- Kiểm tra SUPABASE_URL và keys trong `.env`
- Verify network connectivity
- Check Supabase project status

### 2. **No Events Received**
- Verify database triggers are enabled
- Check RLS (Row Level Security) policies
- Ensure proper table permissions

### 3. **Performance Issues** 
- Monitor Supabase usage/quotas
- Consider implementing event batching
- Optimize query patterns

## Advanced Features

### 1. **Custom Event Handlers**
```python
def custom_fall_handler(event_data):
    # Custom logic for fall events
    send_emergency_sms(event_data)
    call_911_if_critical(event_data)
    update_patient_record(event_data)

realtime_service.subscribe_to_events(
    'event_detections',
    'INSERT', 
    custom_fall_handler
)
```

### 2. **Webhook Integration**
Sử dụng Supabase Edge Functions để gửi webhooks:
```sql
-- Create webhook trigger
CREATE OR REPLACE FUNCTION notify_webhook()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM net.http_post(
    'https://your-webhook-endpoint.com/healthcare',
    to_json(NEW)
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER healthcare_webhook_trigger
  AFTER INSERT ON event_detections
  FOR EACH ROW
  EXECUTE FUNCTION notify_webhook();
```

### 3. **Multiple Environment Support**
```python
# Development
SUPABASE_URL=https://dev-project.supabase.co

# Production  
SUPABASE_URL=https://prod-project.supabase.co
```

## Security

### 1. **API Key Management**
- Sử dụng Service Role Key cho server operations
- Sử dụng Anon Key cho client operations
- Không commit keys vào git

### 2. **Row Level Security (RLS)**
```sql
-- Enable RLS on tables
ALTER TABLE event_detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own events" ON event_detections
  FOR SELECT USING (auth.uid() = user_id);
```

### 3. **Rate Limiting**
Supabase có built-in rate limiting, monitor usage để tránh exceed limits.

## Support

Để được hỗ trợ:
1. Check logs trong terminal
2. Review Supabase dashboard cho errors
3. Verify database schema và permissions
4. Test connection với simple queries

## Changelog

- **v1.0**: Initial Supabase realtime integration
- **v1.1**: Added alert system and multi-client support
- **v1.2**: Added webhook and REST API support
