# MO JSON Format Specification

## ✅ Updated Status Format (3 values only)

### Status Values:
- `normal` - Tình trạng bình thường, không có cảnh báo
- `warning` - Cảnh báo nhẹ, cần theo dõi  
- `danger` - Nguy hiểm, cần can thiệp ngay lập tức

### Healthcare Monitor Mapping:
```
Healthcare Monitor → MO Format
"normal"     → "normal"
"warning"    → "warning"  
"high"       → "danger"
"critical"   → "danger"
```

## 📋 Complete JSON Structure:

```json
{
  "userId": "patient123",
  "sessionId": "session_patient123_1754331078",
  "imageUrl": "http://localhost:8001/alerts/alert_2025-08-04_13-19-13-145_fall_detected_conf_0.850.jpg",
  "status": "danger",
  "action": "fall_detected",
  "location": "Room_101_Bed_A",
  "time": 1754331078
}
```

## 🔗 API Endpoints:

### 1. GET Detection Events
```
GET http://localhost:8001/api/demo/detection-events
```

### 2. POST Add Event
```
POST http://localhost:8001/api/demo/add-event
Content-Type: application/json

{
  "action": "fall_detected",
  "status": "danger", 
  "userId": "patient123"
}
```

### 3. WebSocket Real-time
```
ws://localhost:8001/ws/demo/{user_id}
```

### 4. Get Alert Images
```
GET http://localhost:8001/api/demo/alert-images
```

### 5. Direct Image Access
```
GET http://localhost:8001/alerts/{filename}
```

## 📊 Sample Response:

```json
{
  "success": true,
  "data": [
    {
      "userId": "patient123",
      "sessionId": "session_demo_001", 
      "imageUrl": "http://localhost:8001/alerts/alert_2025-08-04_13-21-49-310_fall_detected_conf_0.850.jpg",
      "status": "danger",
      "action": "fall_detected",
      "location": "Room_101_Bed_A",
      "time": 1754330763
    },
    {
      "userId": "patient888",
      "sessionId": "session_1754331071",
      "imageUrl": "http://localhost:8001/alerts/alert_2025-08-04_13-22-21-240_fall_detected_conf_0.850.jpg", 
      "status": "warning",
      "action": "fall_detected",
      "location": "Room_101_Bed_A",
      "time": 1754331071
    },
    {
      "userId": "patient456",
      "sessionId": "session_demo_003",
      "imageUrl": "http://localhost:8001/alerts/alert_2025-08-04_13-21-56-797_fall_detected_conf_0.850.jpg",
      "status": "normal", 
      "action": "person_detected",
      "location": "Room_102_Bed_B",
      "time": 1754331003
    }
  ],
  "total": 5,
  "timestamp": "2025-08-05T01:11:24.232725"
}
```

## 🎯 Available Actions:
- `fall_detected` - Fall detection alert
- `seizure_detected` - Seizure detection alert  
- `person_detected` - Normal person detection
- `seizure_warning` - Seizure warning (pre-alert)

## 📁 Real Alert Images:
- Total: 36 fall detection images from 2025-08-04
- Format: `alert_YYYY-MM-DD_HH-MM-SS-mmm_fall_detected_conf_X.XXX.jpg`
- Confidence range: 0.704 - 0.850
- Direct access: `http://localhost:8001/alerts/{filename}`
