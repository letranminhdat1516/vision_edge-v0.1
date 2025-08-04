# Vision Edge - Healthcare Monitoring System

**🏥 Advanced Healthcare Monitoring với AI-Powered Video Analysis**

[![Status](https://img.shields.io/badge/Status-Phase%202%20Completed-success)]()
[![Performance](https://img.shields.io/badge/Performance-82--96%25%20Optimized-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange)]()

---

## 🎯 Project Objectives & Status

### ✅ **Phase 1: Core System Integration** (COMPLETED)
- [x] IMOU camera RTSP integration  
- [x] YOLO person detection
- [x] Healthcare monitoring pipeline
- [x] Dual display system

### ✅ **Phase 2: Performance Optimization** (COMPLETED) ⭐
- [x] Video keyframe detection integration
- [x] Smart frame filtering (82-96% efficiency improvement)
- [x] Automated storage system
- [x] Real-time statistics & logging

### ✅ **Phase 3: Fall Detection** (COMPLETED) ⭐
- [x] Specialized fall detection integration
- [x] SimpleFallDetector implementation
- [x] Real-time fall alert system
- [x] Healthcare monitoring enhancement

### 🔄 **Phase 4: Seizure Detection** (IN PROGRESS) 🧠
- [x] VSViG model integration analysis
- [x] Seizure detection architecture design
- [ ] Custom pose estimation for medical scenarios
- [ ] Real-time seizure probability prediction
- [ ] Dual detection system (Fall + Seizure)

---

## 🚀 System Architecture

### **Optimized Processing Pipeline**
```
📹 IMOU Camera (RTSP Stream)
    ↓
🔍 Motion Detection (Filter static frames)
    ↓
🎬 Keyframe Detection (Extract important frames)
    ↓
🤖 YOLO Person Detection (Healthcare optimized)
    ↓
🔄 Dual AI Analysis:
    ├── 🩹 Fall Detection (SimpleFallDetector)
    └── 🧠 Seizure Detection (VSViG Model) ← NEW!
    ↓
🏥 Healthcare Analysis (Emergency classification)
    ↓
💾 Smart Storage (Keyframes, detections, alerts)
    ↓
📊 Real-time Statistics & Emergency Alerts
```

### **Performance Achievement**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| YOLO Processing | 30 FPS | 3-8 FPS | **82-96% reduction** |
| Storage Usage | All frames | Important only | **80-95% reduction** |
| Processing Efficiency | 0% skipped | 84.8% skipped | **Major optimization** |

---

## 📁 Project Structure

```
vision_edge-v0.1/
├── src/
│   ├── camera/
│   │   └── simple_camera.py          # IMOU camera integration
│   ├── video_processing/
│   │   └── simple_processing.py      # Complete AI pipeline
│   ├── fall_detection/               # Fall detection system
│   │   ├── simple_fall_detector_v2.py
│   │   └── ai_models/
│   └── seizure_detection/            # NEW! Seizure detection
│       ├── vsvig_detector.py
│       ├── pose_estimator.py
│       └── seizure_predictor.py
├── examples/
│   └── healthcare_monitor_simple.py  # Main application
├── models/
│   ├── keyframe_detection/           # Video keyframe detector
│   ├── fall_detection/              # Fall detection models
│   └── VSViG/                       # NEW! Seizure detection models
│       └── VSViG/
│           ├── VSViG.py
│           ├── VSViG-base.pth
│           └── pose.pth
├── data/
│   └── saved_frames/                # Smart storage system
│       ├── keyframes/
│       ├── detections/
│       └── alerts/
├── logs/                           # Comprehensive logging
├── docs/                           # Project documentation
└── README.md                       # This file
```

---

## 🔧 Technical Components

### **Core Technologies**
- **Camera**: IMOU RTSP integration
- **AI/ML**: YOLOv8 person detection + Fall detection + VSViG seizure detection
- **Video Processing**: OpenCV với keyframe detection
- **Performance**: Video-keyframe-detector optimization
- **Healthcare AI**: Dual detection system (Fall + Seizure)
- **Pose Estimation**: Custom pose model for medical scenarios
- **Logging**: Comprehensive event tracking
- **Storage**: Smart frame saving system

### **Key Features**
- **Real-time Processing**: 30 FPS camera stream
- **Smart Filtering**: Motion + Keyframe detection
- **Healthcare Focus**: Person-only detection
- **Automated Saving**: Important frames only
- **Live Statistics**: Performance monitoring
- **Error Recovery**: Auto-reconnection system

---

## 📊 Current Performance

### **Latest Run Statistics**
```
📊 HEALTHCARE MONITOR STATISTICS
Uptime: 0.9 minutes
Total Frames: 797
Frames Processed: 121 (15.2%)
Keyframes Detected: 121
Persons Detected: 118 (97.5% accuracy)
Processing Efficiency: 84.8% frames skipped
```

### **System Capabilities**
- **Motion Detection**: 56-63% of frames have motion
- **Keyframe Rate**: 15-18% of total frames processed
- **Person Detection**: 97.5% accuracy on keyframes
- **System Stability**: Continuous operation with auto-recovery

---

## 🚀 Quick Start

### **Requirements**
```bash
# Python 3.10+
pip install ultralytics opencv-python numpy peakutils matplotlib pillow
```

### **Run Healthcare Monitor**
```bash
cd examples
python healthcare_monitor_simple.py
```

### **View Logs**
```bash
tail -f logs/healthcare_monitor_*.log
```

---

## 📚 Documentation

- **[📋 Project Overview](docs/README.md)** - Complete documentation index
- **[🔧 YOLO Integration](docs/01_YOLO_RTSP_INTEGRATION.md)** - Phase 1 implementation  
- **[⚡ Keyframe Detection](docs/02_KEYFRAME_DETECTION_INTEGRATION.md)** - Phase 2 optimization
- **[📊 Performance Analysis](docs/02_KEYFRAME_DETECTION_INTEGRATION.md#performance-results)** - Detailed metrics

---

## 🏥 Healthcare Applications

### **Target Use Cases**
- **Elder Care**: Monitoring elderly people for falls and emergencies
- **Hospital Monitoring**: Patient observation and safety
- **Home Healthcare**: Remote patient monitoring
- **Rehabilitation**: Progress tracking and safety monitoring

### **Detection Capabilities**
- **Person Detection**: Real-time human presence detection
- **Motion Analysis**: Movement pattern analysis  
- **Alert System**: Automated emergency notifications
- **Data Storage**: Important event archival

---

## 🔮 Future Roadmap

### **Phase 3: Advanced Features** 
- **Specialized Fall Detection**: Advanced AI model for fall detection
- **Database Integration**: Long-term data storage và analytics
- **API Development**: RESTful API cho mobile integration
- **WebSocket Notifications**: Real-time alerts system

### **Phase 4: Enterprise Features**
- **Multi-camera Support**: Scale to multiple camera streams  
- **Cloud Integration**: Cloud storage và processing
- **Mobile Application**: Remote monitoring app
- **Analytics Dashboard**: Advanced reporting system

---

## 🤝 Contributing

This project is part of FPT Capstone project. For technical details, see documentation trong `/docs/` folder.

### **Development Team**
- **Architecture**: AI-powered healthcare monitoring system
- **Technologies**: Python, YOLO, OpenCV, Video Analysis
- **Focus**: Performance optimization và real-time processing

---

## 📄 License

This project is developed as part of FPT University Capstone project.

---

## 📞 Support

For technical documentation và implementation details:
- **Documentation**: `/docs/` folder
- **Logs**: `/logs/` folder  
- **Configuration**: Check `healthcare_monitor_simple.py`

---

*Last Updated: August 4, 2025*  
*Project Status: Phase 2 Completed - Production Ready* ✅
- Lưu trữ dữ liệu vào Supabase
- Giao tiếp với frontend/mobile app

## Cấu trúc thư mục

```
vision_edge-v0.1/
├── src/
│   ├── camera/              # Xử lý Camera IMOU RTSP
│   ├── video_processing/    # Video stream + motion detection (OpenCV)
│   ├── keyframe_detector/   # Video keyframe detector
│   ├── ysvio/              # YSVIO - detect cỗ giật
│   ├── person_fall_detection/ # Person Fall Detection
│   ├── websocket/          # WebSocket handler
│   └── supabase/           # Supabase Realtime connection
├── config/                 # Configuration files
├── logs/                   # Log files
└── tests/                  # Test files
```

## Luồng dữ liệu

1. **Camera IMOU** → thu thập video RTSP
2. **Video Processing** → xử lý motion detection
3. **Keyframe Detector** → trích xuất frames quan trọng
4. **AI Analysis** → YSVIO & Person Fall Detection song song
5. **Alert System** → WebSocket + Supabase gửi cảnh báo

## Công nghệ sử dụng

- **Computer Vision**: OpenCV
- **AI/ML**: YOLO, Custom models
- **Database**: Supabase
- **Real-time Communication**: WebSocket
- **Video Processing**: RTSP protocol
- **Programming Language**: Python/JavaScript
