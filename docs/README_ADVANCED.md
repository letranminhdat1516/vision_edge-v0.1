# Vision Edge v0.1 - Advanced Healthcare Monitoring System

## 🏥 Tổng quan hệ thống

**Vision Edge** là hệ thống giám sát y tế tiên tiến sử dụng AI để phát hiện đồng thời các tình huống khẩn cấp trong chăm sóc sức khỏe:

### ✨ Tính năng chính
- **🩹 Fall Detection**: Phát hiện té ngã với độ chính xác cao
- **🧠 Seizure Detection**: Phát hiện co giật bằng VSViG AI model
- **📹 IMOU Camera Integration**: Kết nối camera IMOU qua RTSP
- **📊 Real-time Statistics**: Thống kê và giám sát thời gian thực
- **🎯 Keypoint Visualization**: Hiển thị điểm khớp cơ thể (có thể bật/tắt)
- **🚨 Emergency Alert System**: Hệ thống cảnh báo đa cấp độ

## 🚀 Quick Start

### 1. Chạy hệ thống đầy đủ
```bash
python run_advanced_monitor.py
```

### 2. Chạy từ examples directory
```bash
cd examples
python advanced_healthcare_monitor.py
```

### 3. Demo nhanh
```bash
cd examples
python demo_advanced_monitor.py
```

## 🎮 Điều khiển trong thời gian chạy

- **'k'**: Bật/tắt hiển thị keypoints
- **'s'**: Bật/tắt hiển thị statistics
- **'q'**: Thoát chương trình

## 📊 Dual Detection System

### Fall Detection (Phát hiện té ngã)
- **Technology**: YOLO + Pose Analysis + Rule-based Detection
- **Confidence Threshold**: 0.7
- **Alert Cooldown**: 2 seconds
- **Features**: 
  - Real-time pose estimation
  - Body orientation analysis  
  - Motion pattern recognition
  - False positive filtering

### Seizure Detection (Phát hiện co giật)
- **Technology**: VSViG (ECCV 2024) + Temporal Analysis
- **Model**: Skeleton-based Spatiotemporal ViG
- **Temporal Window**: 30 frames (2 seconds @ 15 FPS)
- **Features**:
  - Medical-grade pose estimation
  - Spatiotemporal pattern analysis
  - Temporal consistency checking
  - Progressive alert system

## 🎯 Alert Levels

| Level | Color | Description | Condition |
|-------|-------|-------------|-----------|
| **Normal** | 🟢 Green | Hoạt động bình thường | Không có phát hiện |
| **Warning** | 🟡 Yellow | Nghi ngờ co giật | Seizure confidence 0.4-0.7 |
| **High** | 🟠 Orange | Phát hiện té ngã | Fall detected |
| **Critical** | 🔴 Red | Phát hiện co giật | Seizure detected |

## 📈 Real-time Statistics

### General Metrics
- Runtime, FPS, Processing efficiency
- Total frames, Processed frames
- Person detection count

### Fall Detection Stats  
- Falls detected, Average confidence
- Last fall time, False positives

### Seizure Detection Stats
- Seizures detected, Warnings issued
- Pose extraction failures
- Temporal buffer status

### Emergency Stats
- Critical alerts, Total alerts
- Current alert status
- Last alert time

## 🔧 Project Structure

```
vision_edge-v0.1/
├── src/                          # Core source code
│   ├── camera/                   # Camera integration
│   ├── video_processing/         # Video processing pipeline
│   ├── fall_detection/           # Fall detection system
│   ├── seizure_detection/        # NEW: Seizure detection system
│   │   ├── pose_estimator.py     # Medical-grade pose estimation
│   │   ├── vsvig_detector.py     # VSViG model wrapper
│   │   └── seizure_predictor.py  # Real-time prediction system
│   └── utils/                    # Utility functions
├── models/                       # AI Models
│   ├── VSViG/                    # NEW: VSViG seizure detection models
│   └── yolo/                     # YOLO detection models
├── examples/                     # Example applications
│   ├── advanced_healthcare_monitor.py  # NEW: Dual detection system
│   ├── demo_advanced_monitor.py        # NEW: Quick demo script
│   └── healthcare_monitor_simple.py    # Original fall detection
├── docs/                         # Documentation
│   └── ADVANCED_HEALTHCARE_GUIDE.md    # NEW: Complete usage guide
├── run_advanced_monitor.py       # NEW: Main run script
└── README.md                     # This file
```

## 📋 Phase Development Status

### ✅ Phase 1: Camera Integration (Complete)
- IMOU camera RTSP connection
- Stream optimization and buffering
- Auto-reconnection mechanism

### ✅ Phase 2: Object Detection (Complete)  
- YOLO person detection
- Motion detection optimization
- Keyframe selection algorithm

### ✅ Phase 3: Fall Detection (Complete)
- Pose-based fall detection
- Rule-based analysis
- Alert system implementation

### ✅ Phase 4: Seizure Detection (Complete)
- **Phase 4A**: VSViG model integration ✅
- **Phase 4B**: Temporal analysis system ✅
- **Phase 4C**: Dual detection coordination ✅
- **Phase 4D**: Advanced statistics & visualization ✅

## 🛠️ Technical Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU recommended for seizure detection
- **Camera**: IMOU camera with RTSP support
- **Network**: Stable network connection for camera stream

### Software Dependencies
```bash
pip install opencv-python
pip install ultralytics
pip install torch torchvision
pip install numpy
pip install pathlib
```

### Camera Configuration
- **Protocol**: RTSP
- **URL Format**: `rtsp://username:password@ip:port/path`
- **Resolution**: 640x480 (configurable)
- **FPS**: 15 (configurable)

## 🔍 Performance Optimization

### Adaptive Processing
- Motion detection để chỉ xử lý frame có activity
- Keyframe selection để tối ưu computation
- Buffer management cho temporal analysis
- Multi-threading cho camera capture

### Memory Management
- Automatic cleanup sau mỗi session
- Optimized frame buffers
- Efficient model loading
- Resource monitoring

## 📊 Statistics & Monitoring

### Real-time Dashboard
- Live processing statistics
- Detection confidence levels
- Alert frequency analysis
- Performance metrics

### Logging System
- Comprehensive event logging
- Error tracking and diagnostics
- Performance profiling
- Alert history

## 🚨 Emergency Response

### Alert Priority System
1. **Critical**: Seizure detection → Immediate medical response
2. **High**: Fall detection → Quick assistance needed  
3. **Warning**: Suspicious activity → Monitoring required
4. **Normal**: Regular monitoring → No action needed

### Integration Capabilities
- Hospital management systems
- Emergency response services
- Family notification systems
- Mobile app integration (planned)

## 🔬 AI Models

### Fall Detection
- **YOLO**: Person detection and tracking
- **Pose Estimation**: 17-keypoint body analysis
- **Rule Engine**: Orientation and movement analysis

### Seizure Detection  
- **VSViG Model**: ECCV 2024 research implementation
- **Temporal Analysis**: 30-frame sliding window
- **Medical Features**: Specialized for seizure patterns

## 📖 Usage Examples

### Basic Monitoring
```python
from advanced_healthcare_monitor import AdvancedHealthcareMonitor

# Initialize with default settings
monitor = AdvancedHealthcareMonitor(
    show_keypoints=True,
    show_statistics=True
)

# Start monitoring
monitor.run_monitoring()
```

### Custom Configuration
```python
# Initialize with custom settings
monitor = AdvancedHealthcareMonitor(
    show_keypoints=False,  # Hide keypoints
    show_statistics=True   # Show statistics
)
```

## 🔧 Troubleshooting

### Common Issues

#### Camera Connection Problems
- Verify camera IP and credentials
- Check RTSP URL format
- Ensure network connectivity
- Confirm camera RTSP support

#### Performance Issues
- Monitor GPU/CPU usage
- Adjust processing thresholds
- Check available memory
- Optimize camera resolution

#### Model Loading Issues
- Verify VSViG model files exist
- Check YOLO model download
- Ensure sufficient disk space
- Validate model file integrity

## 📚 Documentation

- **[Advanced Healthcare Guide](docs/ADVANCED_HEALTHCARE_GUIDE.md)**: Comprehensive usage guide
- **[API Documentation](docs/)**: Detailed API reference
- **[Troubleshooting Guide](docs/)**: Common issues and solutions

## 🎯 Future Roadmap

### Planned Features
- [ ] Multi-camera support
- [ ] Cloud analytics dashboard  
- [ ] Mobile app integration
- [ ] Advanced AI models
- [ ] Custom alert rules
- [ ] Integration APIs

### Research Integration
- [ ] Latest seizure detection research
- [ ] Advanced fall detection algorithms
- [ ] Multi-modal sensor fusion
- [ ] Predictive health analytics

## 👥 Team & Credits

**Developed by**: FPT Capstone Project Team  
**Institution**: FPT University  
**Project**: Vision Edge Healthcare Monitoring  
**Version**: 0.1  
**Year**: 2024

### Research Credits
- **VSViG Model**: ECCV 2024 Paper Implementation
- **YOLO**: Ultralytics YOLO implementation
- **Pose Estimation**: Medical-grade pose analysis

## 📄 License

This project is part of FPT University Capstone Project.  
For academic and research purposes.

---

## 🚀 Getting Started Now

1. **Clone và setup project**:
   ```bash
   git clone [repository]
   cd vision_edge-v0.1
   pip install -r requirements.txt
   ```

2. **Configure camera trong code**:
   ```python
   # Edit camera_config in advanced_healthcare_monitor.py
   camera_config = {
       'url': 'rtsp://your_username:your_password@your_camera_ip:554/path',
       # ... other settings
   }
   ```

3. **Chạy hệ thống**:
   ```bash
   python run_advanced_monitor.py
   ```

4. **Customize settings**:
   - Chọn hiển thị keypoints: y/n
   - Chọn hiển thị statistics: y/n
   - Sử dụng controls: k/s/q

**🏥 Hệ thống sẵn sàng giám sát sức khỏe với AI dual detection!**
