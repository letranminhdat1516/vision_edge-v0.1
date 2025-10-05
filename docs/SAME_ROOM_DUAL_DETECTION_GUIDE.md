# Same Room Dual Detection Setup Guide

## Tổng quan

Hệ thống **Same Room Dual Detection** sử dụng 2 camera trong cùng 1 phòng để:

- Loại bỏ điểm mù (blind spots)
- Phát hiện đồng thời (simultaneous detection)
- Kết hợp kết quả từ 2 camera để cải thiện độ chính xác

## Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐
│   Camera 01     │    │   Camera 02     │
│  (Left Corner)  │    │ (Right Corner)  │
└─────┬───────────┘    └─────┬───────────┘
      │                      │
      │   RTSP Streams       │
      │                      │
      └──────┬─────────┬─────┘
             │         │
    ┌────────▼─────────▼────────┐
    │  Same Room Dual Detection │
    │       Fusion Engine       │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────┐
    │ Healthcare Events │
    │  - Fall Detection │
    │  - Seizure Alert  │
    │  - Motion Tracking│
    └───────────────────┘
```

## Cấu hình Camera

### 1. Vị trí Camera

```json
{
  "cameras": {
    "camera_01": {
      "area": "Living Room",
      "position": "left",
      "description": "Left corner coverage",
      "rtsp_url": "rtsp://username:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
    },
    "camera_02": {
      "area": "Living Room",
      "position": "right",
      "description": "Right corner coverage",
      "rtsp_url": "rtsp://username:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0"
    }
  }
}
```

### 2. Detection Settings

```json
{
  "detection_settings": {
    "camera_01": {
      "fall_detection": {
        "enabled": true,
        "sensitivity": 0.6,
        "min_confidence": 0.4
      },
      "seizure_detection": {
        "enabled": true,
        "sensitivity": 0.7,
        "temporal_window": 25
      }
    },
    "camera_02": {
      "fall_detection": {
        "enabled": true,
        "sensitivity": 0.6,
        "min_confidence": 0.4
      },
      "seizure_detection": {
        "enabled": true,
        "sensitivity": 0.7,
        "temporal_window": 25
      }
    }
  }
}
```

## Fusion Algorithm

### 1. Detection Fusion

```python
def _fuse_detections(self, detections):
    \"\"\"Kết hợp kết quả từ 2 camera\"\"\"

    # Weighted scoring
    left_weight = 0.5
    right_weight = 0.5

    # Fall detection fusion
    fall_confidence = (
        left_detection.fall_confidence * left_weight +
        right_detection.fall_confidence * right_weight
    )

    # Seizure detection fusion
    seizure_confidence = (
        left_detection.seizure_confidence * left_weight +
        right_detection.seizure_confidence * right_weight
    )

    return fused_result
```

### 2. Coverage Areas

- **Camera 01 (Left)**: Góc trái phòng, bàn làm việc, cửa ra vào
- **Camera 02 (Right)**: Góc phải phòng, giường ngủ, cửa sổ
- **Overlap Zone**: Khu vực trung tâm được cả 2 camera quan sát

## Cài đặt và Chạy

### 1. Dependencies

```bash
pip install opencv-python
pip install numpy
pip install threading
```

### 2. Environment Variables

```bash
# Camera 01 (Left)
export IMOU_CAMERA_01_URL="rtsp://user:pass@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
export IMOU_CAMERA_01_USERNAME="your_username"
export IMOU_CAMERA_01_PASSWORD="your_password"

# Camera 02 (Right)
export IMOU_CAMERA_02_URL="rtsp://user:pass@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0"
export IMOU_CAMERA_02_USERNAME="your_username"
export IMOU_CAMERA_02_PASSWORD="your_password"
```

### 3. Test Camera Connection

```bash
cd examples
python test_dual_detection.py
```

### 4. Run Dual Detection System

```bash
cd examples
python same_room_dual_detection_main.py
```

### 5. Test Only Camera Connection

```bash
python same_room_dual_detection_main.py --test-cameras
```

## File Structure

```
src/
├── service/
│   └── same_room_dual_detection.py    # Main dual detection class
├── config/
│   ├── config.json                    # Camera configurations
│   └── detection_settings.json       # Detection parameters
└── camera/
    ├── config.py                      # Camera config class
    └── simple_camera.py               # RTSP camera handler

examples/
├── same_room_dual_detection_main.py   # Main application
└── test_dual_detection.py            # Test script
```

## API Usage

### 1. Initialize Dual Detection

```python
from service.same_room_dual_detection import SameRoomDualDetection

# Setup cameras
cameras = {
    "camera_01": {
        "camera": SimpleIMOUCamera(config1),
        "position": "left",
        "area": "Living Room"
    },
    "camera_02": {
        "camera": SimpleIMOUCamera(config2),
        "position": "right",
        "area": "Living Room"
    }
}

# Create dual detector
dual_detector = SameRoomDualDetection(cameras=cameras)
```

### 2. Start Detection

```python
# Start detection threads
dual_detector.start()

# Get statistics
stats = dual_detector.get_statistics()
print(f"Detections: {stats}")

# Check for recent alerts
if dual_detector.has_recent_detections():
    print("Alert detected!")
```

### 3. Stop Detection

```python
dual_detector.stop()
```

## Monitoring và Debug

### 1. Log Files

- `dual_detection.log`: Main application log
- `camera_01.log`: Camera 01 specific logs
- `camera_02.log`: Camera 02 specific logs

### 2. Statistics Monitoring

```python
{
    "left_camera_detections": 145,
    "right_camera_detections": 152,
    "fused_detections": 89,
    "alert_count": 3,
    "uptime_seconds": 3600,
    "fusion_rate": 0.58
}
```

### 3. Health Check

```python
# Check camera status
health = dual_detector.get_health_status()
print(f"System health: {health}")
```

## Troubleshooting

### 1. Camera Connection Issues

```bash
# Test individual camera
python -c "
from camera.simple_camera import SimpleIMOUCamera
from camera.config import IMOUCameraConfig

config = IMOUCameraConfig.from_env('CAMERA_01')
camera = SimpleIMOUCamera(config)
print('Connected:', camera.connect())
"
```

### 2. Detection Issues

- Kiểm tra lighting conditions
- Verify camera angles and coverage
- Adjust sensitivity settings
- Check detection thresholds

### 3. Performance Issues

- Monitor CPU/Memory usage
- Check network bandwidth
- Optimize frame processing rate
- Reduce video resolution if needed

## Best Practices

### 1. Camera Placement

- **Height**: 2.5-3m từ mặt đất
- **Angle**: 30-45 độ hướng xuống
- **Coverage**: Overlap 20-30% giữa 2 camera
- **Lighting**: Tránh backlight và shadows

### 2. Network Setup

- **Bandwidth**: Minimum 10Mbps per camera
- **Latency**: < 100ms for realtime processing
- **Stability**: Stable connection, avoid WiFi if possible

### 3. Detection Tuning

- **Fall Detection**: Sensitivity 0.6-0.8
- **Seizure Detection**: Temporal window 20-30 frames
- **Motion Threshold**: Adjust based on room activity

## Future Enhancements

1. **AI-based Fusion**: Machine learning để optimize fusion weights
2. **3D Reconstruction**: Sử dụng stereo vision để tạo 3D model
3. **Advanced Tracking**: Multi-object tracking across cameras
4. **Edge Computing**: Local processing để reduce latency
5. **Mobile Integration**: Real-time mobile notifications

## Support

- **Documentation**: [Healthcare Optimization Guide](healthcare_optimization_guide.md)
- **API Reference**: [Advanced Healthcare Guide](ADVANCED_HEALTHCARE_GUIDE.md)
- **Mobile Setup**: [Flutter Integration Guide](FLUTTER_INTEGRATION_GUIDE.md)
