# 🎥 Multi-Camera Healthcare System - Setup Guide

## 📋 Tổng quan

Hệ thống **Multi-Camera Fusion** sử dụng 2 camera IMOU và tự động chọn frame tốt nhất để thực hiện detection healthcare.

### 🎯 Mục tiêu:

- **2 Camera**: Camera 1 (Living Room) + Camera 2 (Bedroom)
- **Best Frame Selection**: Hệ thống tự động chọn camera có frame quality tốt nhất
- **Full Healthcare Features**: Fall detection + Seizure detection + Emergency notifications
- **Single Output**: Chỉ 1 kết quả detection từ frame tốt nhất

## 🚀 Cách hoạt động

### **Frame Selection Algorithm:**

```
Camera 1 Frame ──┐
                 ├── Quality Analyzer ──> Best Frame Selector ──> Healthcare Pipeline
Camera 2 Frame ──┘
```

### **Quality Metrics:**

1. **Brightness Score** (30%): Độ sáng tối ưu (80-180)
2. **Sharpness Score** (40%): Độ rõ nét (Laplacian variance)
3. **Motion Score** (30%): Chuyển động vừa phải (0.02-0.15 ratio)

### **Selection Priority:**

- **Quality Score**: Frame có chất lượng cao nhất
- **Camera Priority**: Living Room > Bedroom (nếu quality tương đương)
- **Recency**: Frame mới nhất (trong vòng 1 giây)

## 🔧 Setup Instructions

### **1. Camera Configuration**

#### **Camera 1 - Living Room:**

- **IP**: 192.168.8.122
- **RTSP**: `rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1`
- **Priority**: High (thường được chọn khi quality tương đương)

#### **Camera 2 - Bedroom:**

- **IP**: 192.168.8.123
- **RTSP**: `rtsp://admin:L2C37340@192.168.8.123:554/cam/realmonitor?channel=1&subtype=1`
- **Priority**: Medium

### **2. Configuration Files đã được cập nhật:**

#### **src/config/config.json:**

```json
"cameras": {
  "camera_01": {
    "camera_id": "22222222-2222-2222-2222-222222222201",
    "name": "Living Room Camera",
    "location": "Living Room",
    "rtsp_url": "rtsp://admin:L2C37340@192.168.8.122:554/..."
  },
  "camera_02": {
    "camera_id": "22222222-2222-2222-2222-222222222202",
    "name": "Bedroom Camera",
    "location": "Bedroom",
    "rtsp_url": "rtsp://admin:L2C37340@192.168.8.123:554/..."
  }
}
```

#### **src/config/detection_settings.json:**

```json
"camera_specific": {
  "camera_01": {
    "fall_sensitivity_multiplier": 0.90,    // Higher sensitivity
    "seizure_sensitivity_multiplier": 0.85,
    "description": "Living Room - High sensitivity"
  },
  "camera_02": {
    "fall_sensitivity_multiplier": 1.15,    // Lower sensitivity
    "seizure_sensitivity_multiplier": 1.10,
    "description": "Bedroom - Reduced sensitivity"
  }
}
```

### **3. New Components:**

#### **Multi-Camera Manager** (`src/service/multi_camera_manager.py`):

- Quản lý kết nối 2 camera đồng thời
- Phân tích chất lượng frame real-time
- Chọn frame tốt nhất cho detection

#### **Multi-Camera Pipeline** (`examples/multi_camera_healthcare_system.py`):

- Tích hợp multi-camera với healthcare pipeline
- Hiển thị statistics selection
- Full healthcare features (fall + seizure)

## 🏃‍♂️ How to Run

### **Option 1: Test Multi-Camera System**

```bash
cd d:\FPT\Capstone\vision_edge-v0.1
python examples/multi_camera_healthcare_system.py
```

### **Option 2: Test Single Camera (Original)**

```bash
python src/main.py
```

## 📊 Expected Results

### **Console Output:**

```
🎥 Multi-Camera Healthcare Monitoring System
📹 Initializing 2-camera system with best frame selection...
✅ Living Room Camera connected successfully
✅ Bedroom Camera connected successfully
✅ 2 cameras connected successfully!
🎯 System will automatically select the best camera frame for detection
📊 Frame selection based on: Quality + Motion + Brightness + Sharpness

📊 Processed: 150 frames | Cam1: 89 | Cam2: 61 | Connected: 2
```

### **GUI Windows:**

1. **Multi-Camera Healthcare Monitor**: Hiển thị frame được chọn
2. **AI Detection View**: Hiển thị person detection + keypoints

### **Keyboard Controls:**

- **'q'**: Quit system
- **'s'**: Show detailed statistics
- **'c'**: Show camera selection statistics

## 🔍 Monitoring & Debugging

### **Camera Selection Statistics:**

```bash
# Press 'c' to see:
🎥 CAMERA SELECTION STATISTICS:
Camera 1 selected: 89 times (59.3%)
Camera 2 selected: 61 times (40.7%)
Total selections: 150
Selection criteria working: ✅
```

### **Quality Indicators:**

- **Selection Balance**: Cả 2 camera đều được chọn (không 100% 1 camera)
- **Frame Rate**: ~15 FPS processing
- **Healthcare Detection**: Fall + Seizure hoạt động bình thường

## 🛠️ Troubleshooting

### **Camera Connection Issues:**

1. **Check IP addresses**: Đảm bảo 192.168.8.122 và 192.168.8.123 accessible
2. **Test RTSP URLs**: Dùng VLC player test từng URL
3. **Network**: Ping test camera IPs
4. **Credentials**: Username/password correct

### **Frame Selection Issues:**

1. **Single Camera Dominance**: Nếu 1 camera luôn được chọn (>90%)

   - Check lighting conditions
   - Check motion levels
   - Verify camera positioning

2. **No Frames**: Nếu không có frame nào
   - Check camera connections
   - Verify RTSP URLs
   - Check network connectivity

### **Performance Issues:**

1. **Low FPS**:

   - Reduce resolution to 320x240
   - Increase motion threshold
   - Disable frame saving

2. **High CPU**:
   - Reduce keyframe threshold
   - Limit simultaneous processing

## 📈 Expected Performance

### **With 2 Cameras:**

- **Frame Processing**: ~15 FPS
- **Selection Algorithm**: <5ms overhead
- **Memory Usage**: ~2x single camera
- **Detection Accuracy**: Same as single camera (best frame selected)

### **Quality Metrics:**

- **Brightness Score**: 0.8-1.0 (good lighting)
- **Sharpness Score**: 0.7-1.0 (sharp images)
- **Motion Score**: 0.5-1.0 (moderate movement)
- **Overall Quality**: 0.6-1.0 (acceptable range)

## 🎯 Next Steps

1. **Test with Real Cameras**: Verify both RTSP connections
2. **Tune Quality Thresholds**: Adjust based on your environment
3. **Add More Cameras**: Extend to 3+ cameras if needed
4. **Optimize Selection**: Fine-tune selection algorithm
5. **Add Failover**: Implement camera failure detection

---

**💡 Tips:**

- Living Room camera sẽ được ưu tiên khi quality tương đương
- Bedroom camera được chọn khi có chuyển động/lighting tốt hơn
- System tự động failover nếu 1 camera disconnect
- All healthcare features hoạt động như single-camera system
