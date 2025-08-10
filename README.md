# Vision Edge - Healthcare Monitoring System (Function-based Architecture)

**🏥 Advanced Healthcare Monitoring với Function-based Programming**

[![Status](https://img.shields.io/badge/Status-Function--based%20Architecture-success)]()
[![Architecture](https://img.shields.io/badge/Architecture-Function--based-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange)]()

---

## 🎯 Overview

Vision Edge is a comprehensive healthcare monitoring system completely rewritten with **function-based programming** principles. The system provides real-time patient safety monitoring using advanced computer vision and healthcare-specific detection algorithms.

## 🏗️ Architecture Transformation

### From Class-based to Function-based
This system has been completely refactored from object-oriented programming to function-based programming, providing:

- **🧩 Modular Design**: Clear separation of concerns with focused modules
- **⚡ Pure Functions**: Predictable behavior with minimal side effects  
- **🔄 Composable Components**: Functions can be easily combined for different workflows
- **✅ Better Testability**: Individual functions can be tested in isolation
- **🔧 Enhanced Maintainability**: Easier to understand and modify individual components

### Folder Structure
```
healthcare_monitor_functional/
├── main.py                     # Application entry point
├── core/                       # Core utilities
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging system
│   └── utils.py               # Common utilities
├── camera/                     # Camera control
│   └── controls.py            # Camera functions
├── processing/                 # Video processing
│   └── video.py               # Video analysis functions
├── detection/                  # Detection algorithms
│   ├── motion.py              # Motion detection
│   ├── fall.py                # Fall detection  
│   └── seizure.py             # Seizure detection
├── visualization/              # Display functions
│   └── display.py             # Visualization functions
└── alerts/                     # Alert management
    └── management.py          # Alert functions
```

## ✨ Key Features

- **📹 Real-time Video Processing**: Live RTSP stream processing from IMOU cameras
- **🚨 Fall Detection**: Advanced fall detection using pose estimation and motion analysis
- **⚠️ Seizure Detection**: VSViG-based seizure detection with temporal analysis
- **👤 Person Detection**: YOLO-based person detection and tracking
- **📢 Alert System**: Multi-channel alerting (email, webhook, logging)
- **⚙️ Function-based Architecture**: Modern, maintainable, and testable codebase

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration
cd healthcare_monitor_functional
python main.py

# Run with custom configuration
python main.py --config ../config.json

# Run with debug logging
python main.py --debug
```

### Configuration
Edit `config.json` to customize:
```json
{
  "camera": {
    "type": "rtsp",
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
    "width": 640,
    "height": 480,
    "fps": 15
  },
  "detection": {
    "yolo_model": "yolov8s.pt",
    "confidence_threshold": 0.5,
    "fall_threshold": 0.7,
    "seizure_threshold": 0.7
  },
  "alerts": {
    "email_notifications": true,
    "webhook_notifications": false
  }
}
```

## 📋 Function-based Design Principles

### 1. Single Responsibility
Each function has a single, well-defined purpose:
```python
def calculate_motion_level(frame1, frame2, bbox):
    """Calculate motion level between two frames within bounding box"""

def enhance_confidence_with_motion(base_confidence, motion_level):
    """Enhance confidence using motion information"""
```

### 2. Pure Functions
Most functions are pure with no side effects:
```python
def smooth_confidence_values(confidence_history, current_confidence):
    """Pure function - same inputs always produce same outputs"""
```

### 3. Composability
Functions can be easily combined:
```python
motion_level = calculate_motion_level(prev_frame, curr_frame, bbox)
enhanced_confidence = enhance_confidence_with_motion(base_conf, motion_level)
smoothed_confidence = smooth_confidence_values(history, enhanced_confidence)
```

### 4. Dependency Injection
Functions receive their dependencies as parameters:
```python
def process_fall_detection(detector, predictor, frame, bbox, history, frames, motion, config):
    """All dependencies injected as parameters"""
```

## 🔧 Core Modules

### Configuration (`core/config.py`)
- Centralized configuration management
- Environment-specific settings
- Default fallback values

### Logging (`core/logger.py`)
- Structured logging system
- Performance metrics logging
- Detection event logging

### Camera System (`camera/controls.py`)
- Camera initialization and configuration
- Thread-safe frame capture
- Frame buffering system

### Detection Systems
- **Motion Detection** (`detection/motion.py`): Frame-to-frame motion calculation
- **Fall Detection** (`detection/fall.py`): Function-based fall detection pipeline
- **Seizure Detection** (`detection/seizure.py`): VSViG-based seizure detection

### Visualization (`visualization/display.py`)
- Bounding box rendering
- Keypoint and skeleton drawing
- Statistics overlay

### Alert Management (`alerts/management.py`)
- Multi-channel alert system
- Priority-based alert handling
- Email and webhook notifications

## 📊 Performance Benefits

- **Improved Testability**: Individual functions can be tested in isolation
- **Better Modularity**: Clear separation of concerns between modules
- **Enhanced Maintainability**: Easier to understand and modify components
- **Increased Reusability**: Functions can be reused in different contexts
- **Better Error Handling**: Explicit error handling with fallback mechanisms

## 📚 Documentation

- [Function-based Architecture Guide](healthcare_monitor_functional/README.md)
- [Configuration Guide](docs/configuration_guide.md)
- [Advanced Healthcare Guide](docs/ADVANCED_HEALTHCARE_GUIDE.md)

## 🛠️ Dependencies

- OpenCV (cv2) - Computer vision operations
- NumPy - Numerical computations
- Ultralytics YOLO - Object detection
- Threading - Concurrent camera operations

## 📝 Testing

Each module can be tested independently:
```python
# Test motion detection
motion_level = calculate_motion_level(test_frame1, test_frame2, test_bbox)
assert 0 <= motion_level <= 1

# Test confidence smoothing
smoothed = smooth_confidence_values([0.5, 0.6], 0.8)
assert isinstance(smoothed, float)
```

## 🔄 Migration from Class-based System

### Key Changes
1. **State Management**: Moved from instance variables to explicit parameter passing
2. **Initialization**: Separated initialization functions from processing functions
3. **Error Handling**: Centralized error handling with fallback functions
4. **Configuration**: Externalized configuration with function-based access

## 🤝 Contributing

This function-based architecture makes it easy to:
- Add new detection algorithms
- Implement custom visualization functions
- Create new alert notification channels
- Extend configuration options
- Add performance monitoring functions

---

**Built with ❤️ using Function-based Programming principles**
