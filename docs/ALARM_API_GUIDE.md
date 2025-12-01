# 🚨 Alarm Control API - Hướng Dẫn Sử Dụng

## 🎯 Tổng Quan

API **CỰC KỲ ĐƠN GIẢN** - chỉ có **1 ENDPOINT DUY NHẤT** để BẬT/TẮT alarm.

### ✨ Đặc Điểm

- ✅ **1 API duy nhất**: POST `/api/alarm/control`
- ✅ **Tích hợp sẵn**: Chạy cùng `main.py`, không cần chạy riêng
- ✅ **Auto-start**: API, Worker, Handler tất cả tự động khởi động
- ✅ **Cực đơn giản**: Chỉ cần truyền `event_id`, `user_id`, `camera_id`, và `enabled` (true/false)

---

## 🚀 Cách Chạy

### Bước 1: Chỉ cần chạy main.py

```bash
python src/main.py
```

**Hệ thống tự động khởi động:**

1. ✅ Emergency Alarm Handler (LISTEN/NOTIFY)
2. ✅ Event Lifecycle Worker (auto-alarm sau 30s)
3. ✅ FastAPI Server (port 8000)

### Bước 2: API đã sẵn sàng!

```
🌐 API Server: http://localhost:8000
📖 Docs: http://localhost:8000/docs
```

---

## 📡 API Endpoint

### 🎛️ POST `/api/alarm/control` - BẬT/TẮT Alarm

**CHỈ 1 ENDPOINT DUY NHẤT!**

#### Request Body

```json
{
  "event_id": "uuid-of-event",
  "user_id": "uuid-of-user",
  "camera_id": "uuid-of-camera",
  "enabled": true // true = BẬT, false = TẮT
}
```

#### Example: BẬT Alarm

```bash
curl -X POST http://localhost:8000/api/alarm/control \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "abc-123",
    "user_id": "user-456",
    "camera_id": "cam-789",
    "enabled": true
  }'
```

#### Example: TẮT Alarm

```bash
curl -X POST http://localhost:8000/api/alarm/control \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "abc-123",
    "user_id": "user-456",
    "camera_id": "cam-789",
    "enabled": false
  }'
```

#### Response

```json
{
  "success": true,
  "action": "BẬT", // hoặc "TẮT"
  "message": "Đã BẬT alarm thành công",
  "data": {
    "success": true,
    "message": "...",
    "notification_sent": true
  }
}
```

---

## 🧪 Testing

### Sử dụng test script

```bash
python test_alarm_control_api.py
```

### Hoặc test thủ công với Python

```python
import requests

# BẬT alarm
response = requests.post("http://localhost:8000/api/alarm/control", json={
    "event_id": "test-001",
    "user_id": "b7757b17-4b5e-4f21-86db-5d6e5afe81c7",
    "camera_id": "cam-001",
    "enabled": True  # BẬT
})
print(response.json())

# TẮT alarm
response = requests.post("http://localhost:8000/api/alarm/control", json={
    "event_id": "test-001",
    "user_id": "b7757b17-4b5e-4f21-86db-5d6e5afe81c7",
    "camera_id": "cam-001",
    "enabled": False  # TẮT
})
print(response.json())
```

---

## 🔄 Lifecycle State Flow

### **API Control → Automatic State Update**

| API Call        | Alarm            | Lifecycle State     |
| --------------- | ---------------- | ------------------- |
| `enabled=true`  | ✅ **BẬT** alarm | → `ALARM_ACTIVATED` |
| `enabled=false` | ❌ **TẮT** alarm | → `RESOLVED`        |

### 1️⃣ Manual Control (API)

```
API enabled=true (BẬT)
→ Phát alarm ngay
→ lifecycle_state = 'ALARM_ACTIVATED'
→ escalated_at = NOW()

API enabled=false (TẮT)
→ Dừng alarm ngay
→ lifecycle_state = 'RESOLVED'
→ resolved_at = NOW()
```

### 2️⃣ Auto-Alarm Flow (Worker)

```
Event danger/warning created
→ lifecycle_state = 'NOTIFIED'
→ [Đợi 30s...]
→ Worker auto-trigger alarm
→ lifecycle_state = 'ALARM_ACTIVATED'
→ [Đợi 30s với status=normal...]
→ Worker auto-resolve
→ lifecycle_state = 'RESOLVED'
```

### 3️⃣ Mix Flow (API Override Worker)

```
Worker auto-alarm → ALARM_ACTIVATED
↓
User gọi API enabled=false → RESOLVED
✅ User có quyền override automation!
```

📖 **Chi tiết:** Xem [LIFECYCLE_STATE_FLOW.md](../LIFECYCLE_STATE_FLOW.md)

---

## 📊 Get Alarm Status

### GET `/api/alarm/status`

```bash
curl http://localhost:8000/api/alarm/status
```

Response:

```json
{
  "is_playing": true,
  "active_alarms": [
    {
      "event_id": "abc-123",
      "timestamp": "2025-11-30T10:30:00"
    }
  ],
  "audio_backend": "winsound",
  "timestamp": "2025-11-30T10:35:00"
}
```

---

## 🔍 Swagger UI

Mở browser: `http://localhost:8000/docs`

- ✅ Interactive API documentation
- ✅ Try it out trực tiếp
- ✅ See all request/response schemas

---

## 💡 Tips

### Lấy User ID và Camera ID từ database

```sql
-- Lấy user_id
SELECT id, email FROM users;

-- Lấy camera_id
SELECT id, camera_name, user_id FROM cameras;
```

### Xem logs

```bash
# Logs hiển thị trực tiếp trên console khi chạy main.py
# Bạn sẽ thấy:
# - 🔊 Alarm triggered/stopped
# - 🤖 Worker checking events
# - 📡 NOTIFY sent/received
```

---

## ❓ Troubleshooting

### API không chạy?

```bash
# Check port 8000 có bị chiếm không
netstat -ano | findstr :8000

# Hoặc thay đổi port trong main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Worker không auto-alarm?

```bash
# Check logs trong console
# Worker chạy mỗi 10s, sẽ có log:
# "🤖 Checking events for auto-alarm promotion..."
```

### Alarm không phát?

```bash
# Check audio device
python -c "from infrastructure.services.audio_alert_service import audio_alert_service; print(audio_alert_service.get_status())"
```

---

## 📝 Summary

**TLDR:**

1. Chạy `python src/main.py` → tất cả tự động start
2. POST `/api/alarm/control` với `enabled=true` → BẬT alarm
3. POST `/api/alarm/control` với `enabled=false` → TẮT alarm
4. Done! 🎉

**Không cần:**

- ❌ Chạy API server riêng
- ❌ Chạy worker riêng
- ❌ Nhiều endpoints phức tạp
- ❌ Quan tâm lifecycle_state

**Chỉ cần:**

- ✅ Biết `event_id`, `user_id`, `camera_id`
- ✅ Set `enabled` = `true`/`false`
- ✅ Gọi 1 API duy nhất!
