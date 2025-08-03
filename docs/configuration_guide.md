# Configuration Usage Guide
# How to configure and customize the Healthcare Monitoring System

## 📋 **Nơi Config Các Thành Phần Hệ Thống**

### **1. Video Source Configuration (Config Nguồn Video)**

#### **File Config Chính:**
```
📁 config/
├── healthcare_config.py      # Core configuration classes
├── healthcare_config.json    # JSON config file (customizable)
├── config_loader.py          # Configuration loader
├── setup_dev_env.bat        # Development environment
└── setup_prod_env.bat       # Production environment
```

#### **Config Video Sources:**
```python
# Trong healthcare_config.py
VIDEO_SOURCES = {
    "default_camera": {
        "type": "camera", 
        "source": 0,                    # USB camera index
        "description": "Default camera"
    },
    "imou_camera_main": {
        "type": "rtsp",
        "source": "rtsp://admin:password@192.168.1.100:554/stream1",
        "description": "IMOU Main Camera"
    },
    "test_video": {
        "type": "file",
        "source": "test_videos/demo.mp4",
        "description": "Test video file"
    }
}
```

#### **Environment Variables (Biến Môi Trường):**
```bash
# Video source selection
set VIDEO_SOURCE=imou_camera_main
set RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream1

# Camera credentials
set CAMERA_USER=admin
set CAMERA_PASSWORD=Abc123456
set CAMERA_IP=192.168.1.100
```

### **2. Alert System Configuration (Config Hệ Thống Cảnh Báo)**

#### **Alert Levels & Timing:**
```python
# Trong healthcare_config.py - AlertSystemConfig
ALERT_LEVELS = {
    "INFO": {
        "delay_ms": 5000,        # 5 giây delay
        "actions": ["log"],
        "channels": ["database"]
    },
    "WARNING": {
        "delay_ms": 1000,        # 1 giây delay
        "actions": ["nurse_notification"],
        "channels": ["mobile_push", "database", "dashboard"]
    },
    "CRITICAL": {
        "delay_ms": 500,         # 500ms delay
        "actions": ["immediate_response"],
        "channels": ["mobile_push", "email", "sms", "database"]
    },
    "EMERGENCY": {
        "delay_ms": 200,         # 200ms delay - CỰC NHANH
        "actions": ["emergency_protocol"],
        "channels": ["emergency_call", "all_notifications"]
    }
}
```

#### **Notification Channels:**
```python
NOTIFICATION_CHANNELS = {
    "mobile_push": {
        "enabled": True,
        "service": "firebase",
        "tokens": ["device_token_1", "device_token_2"]
    },
    "email": {
        "enabled": True,
        "smtp_server": "smtp.hospital.com",
        "recipients": ["nurse@hospital.com", "doctor@hospital.com"]
    },
    "sms": {
        "enabled": True,
        "service": "twilio",
        "phone_numbers": ["+1234567890", "+1234567891"]
    }
}
```

### **3. Motion Detection Configuration (Config Phát Hiện Chuyển Động)**

#### **Basic Settings:**
```python
# Trong MotionDetectionConfig
THRESHOLD = 150              # Healthcare: nhạy hơn security (200)
START_FRAMES = 2             # Trigger nhanh (vs 5 cho security)
PROCESSING_RESOLUTION = (256, 144)  # Resolution thấp cho tốc độ

# Zone-based sensitivity
ZONE_SENSITIVITY = {
    "bed_area": {
        "threshold": 100,     # Rất nhạy
        "coordinates": (0.2, 0.3, 0.8, 0.8),
        "priority": "high"
    },
    "bathroom": {
        "threshold": 80,      # Cực kỳ nhạy
        "coordinates": (0.0, 0.0, 0.3, 0.5),
        "priority": "critical"
    }
}
```

### **4. YOLO Detection Configuration (Config YOLO)**

#### **Model & Performance:**
```python
# Trong YOLOConfig
MODEL_NAME = "yolov8s"       # Balanced speed/accuracy
CONFIDENCE_THRESHOLD = 0.5   # Standard confidence
DEVICE = "auto"              # auto, cpu, cuda:0

# Performance optimization
MODEL_OPTIMIZATION = {
    "precision": "fp16",     # 50% memory reduction
    "optimize": True,        # TensorRT optimization
    "batch_size": 1          # Real-time processing
}

# Healthcare class filtering
HEALTHCARE_CLASSES = {
    "primary": [0],          # person
    "medical": [56, 59],     # chair, toilet
    "context": [60, 62, 67]  # tv, laptop, phone
}
```

### **5. Healthcare Analytics Configuration**

#### **Fall Detection:**
```python
FALL_DETECTION = {
    "enabled": True,
    "height_width_ratio": 1.2,    # Ngưỡng phát hiện nằm
    "position_velocity": 100,     # Tốc độ chuyển động (px/s)
    "ground_proximity": 0.8,      # 80% bottom frame
    "confidence_threshold": 0.7,   # Độ tin cậy cao
    "frame_history": 10           # Phân tích 10 frame
}
```

#### **Medical Zones:**
```python
MEDICAL_ZONES = {
    "bed_area": {
        "coordinates": (0.2, 0.3, 0.8, 0.8),  # Normalized coordinates
        "type": "rest_area",
        "alerts": ["fall_risk", "extended_absence"]
    },
    "bathroom": {
        "coordinates": (0.0, 0.0, 0.3, 0.5),
        "type": "high_risk", 
        "alerts": ["fall_risk", "emergency"]
    }
}
```

## 🔧 **Cách Config Thực Tế**

### **Method 1: Environment Variables (Khuyến nghị)**
```bash
# Chạy development environment
config/setup_dev_env.bat

# Chạy production environment  
config/setup_prod_env.bat

# Hoặc set manual
set HEALTHCARE_ENV=production
set VIDEO_SOURCE=imou_camera_main
set RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream1
```

### **Method 2: JSON Configuration File**
```bash
# Sử dụng custom config file
python examples/healthcare_monitoring.py --config config/healthcare_config.json

# Hoặc trong code:
from config import ConfigLoader
config = ConfigLoader("config/healthcare_config.json")
```

### **Method 3: Direct Code Configuration**
```python
# Trong examples/healthcare_monitoring.py
motion_config = {
    'threshold': 120,          # Tuỳ chỉnh theo môi trường
    'start_frames': 2,
    'resolution': (256, 144)
}

yolo_config = {
    'model_name': 'yolov8s',
    'confidence': 0.6,         # Tăng confidence cho ít false positive
    'healthcare_mode': True
}

# Alert configuration
alert_config = {
    'emergency_delay': 100,    # ms - Cực nhanh cho emergency
    'critical_delay': 300,     # ms - Nhanh cho critical  
    'warning_delay': 800,      # ms - Bình thường cho warning
    'info_delay': 3000         # ms - Chậm cho info
}
```

## 📊 **Config Theo Từng Scenario**

### **ICU Room (Phòng ICU):**
```python
icu_config = {
    "motion_threshold": 80,    # Cực kỳ nhạy
    "alert_delay": 100,        # Cực nhanh (100ms)
    "yolo_confidence": 0.7,    # Độ tin cậy cao
    "fall_detection": True,    # Bật fall detection
    "priority": "realtime"     # Xử lý real-time
}
```

### **General Ward (Phòng Bệnh Thường):**
```python
ward_config = {
    "motion_threshold": 150,   # Nhạy vừa phải
    "alert_delay": 500,        # Nhanh (500ms)
    "yolo_confidence": 0.5,    # Standard confidence
    "fall_detection": True,    # Bật fall detection
    "priority": "high"         # Xử lý ưu tiên cao
}
```

### **Corridor (Hành Lang):**
```python
corridor_config = {
    "motion_threshold": 200,   # Ít nhạy hơn
    "alert_delay": 2000,       # Chậm hơn (2s)
    "yolo_confidence": 0.4,    # Confidence thấp hơn
    "fall_detection": True,    # Vẫn bật fall detection
    "priority": "medium"       # Ưu tiên trung bình
}
```

## ⚙️ **Dynamic Configuration (Config Động)**

### **Runtime Configuration Changes:**
```python
# Thay đổi sensitivity trong runtime
processor.tune_motion_sensitivity(120)

# Thay đổi YOLO targets
processor.update_yolo_targets(['person', 'wheelchair'])

# Thay đổi alert thresholds
healthcare_analyzer.update_alert_thresholds({
    'fall_risk': 0.8,
    'emergency': 0.9
})
```

### **Automatic Optimization:**
```python
# Tự động điều chỉnh dựa trên performance
if cpu_usage > 80:
    motion_threshold += 10      # Giảm sensitivity
    frame_skip_rate = 2         # Skip frames
    
if false_positive_rate > 5:
    yolo_confidence += 0.05     # Tăng confidence
    
if alert_spam_rate > 10:
    alert_cooldown += 5         # Tăng cooldown
```

## 🔍 **Debugging Configuration**

### **Config Validation:**
```python
# Kiểm tra config hợp lệ
from config import validate_config, print_config

validate_config()  # Kiểm tra lỗi config
print_config()     # In ra config hiện tại
```

### **Performance Monitoring:**
```python
# Monitor config performance
stats = processor.get_stats()
print(f"Motion detection rate: {stats['motion_rate']:.2%}")
print(f"False positive rate: {stats['false_positive_rate']:.2%}")
print(f"Alert response time: {stats['avg_alert_time']:.3f}s")
```

## 🎯 **Config Summary**

| Component | Config File | Environment Variable | Runtime Change |
|-----------|-------------|---------------------|----------------|
| **Video Source** | ✅ healthcare_config.py | ✅ VIDEO_SOURCE | ❌ |
| **Motion Detection** | ✅ MotionDetectionConfig | ✅ MOTION_THRESHOLD | ✅ tune_sensitivity() |
| **YOLO Settings** | ✅ YOLOConfig | ✅ YOLO_CONFIDENCE | ✅ update_targets() |
| **Alert System** | ✅ AlertSystemConfig | ✅ ENABLE_*_ALERTS | ✅ update_thresholds() |
| **Healthcare Analytics** | ✅ HealthcareAnalyticsConfig | ❌ | ✅ configure_zones() |

**Tóm lại**: Tất cả config đều có thể thay đổi trong **config/healthcare_config.py**, một số có thể override bằng **environment variables**, và nhiều setting có thể điều chỉnh **runtime** để tối ưu performance!

Bạn muốn config phần nào cụ thể không?
