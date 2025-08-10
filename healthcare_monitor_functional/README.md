# Healthcare Monitor - Function-based Architecture

## Overview

This is a complete transformation of the healthcare monitoring system from class-based to function-based programming with a modular architecture. The system provides real-time fall detection, seizure detection, and motion analysis using IMOU cameras and AI models.

## Architecture

### Folder Structure
```
healthcare_monitor_functional/
├── main.py                     # Main application entry point
├── core/                       # Core utilities and configuration
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging system
│   └── utils.py               # Common utilities
├── camera/                     # Camera control and management
│   ├── __init__.py
│   └── controls.py            # Camera initialization and control
├── processing/                 # Video processing functions
│   ├── __init__.py
│   └── video.py               # Video analysis and preprocessing
├── detection/                  # Detection algorithms
│   ├── __init__.py
│   ├── motion.py              # Motion detection and analysis
│   ├── fall.py                # Fall detection system
│   └── seizure.py             # Seizure detection system
├── visualization/              # Display and visualization
│   ├── __init__.py
│   └── display.py             # Frame visualization and overlays
└── alerts/                     # Alert management
    ├── __init__.py
    └── management.py          # Alert creation and notification
```

## Key Features

### Function-based Architecture
- **Modular Design**: Each module contains focused functions for specific tasks
- **Pure Functions**: Most functions are pure with predictable inputs/outputs
- **Composable**: Functions can be easily combined for different workflows
- **Testable**: Individual functions can be tested in isolation

### Core Modules

#### Configuration (`core/config.py`)
- Centralized configuration management
- Environment-specific settings
- Default fallback values
- JSON/YAML configuration support

#### Logging (`core/logger.py`)
- Structured logging system
- Performance metrics logging
- Detection event logging
- Configurable log levels and formats

#### Utilities (`core/utils.py`)
- Mathematical utilities for confidence calculations
- Statistical functions for motion analysis
- Runtime performance calculations
- Data smoothing and filtering functions

### Camera System (`camera/controls.py`)
- Camera initialization and configuration
- Thread-safe frame capture
- Frame buffering system
- Camera performance monitoring
- RTSP and webcam support

### Video Processing (`processing/video.py`)
- YOLO object detection
- Frame preprocessing and enhancement
- Motion region detection
- Frame quality assessment
- Person detection and tracking

### Detection Systems

#### Motion Detection (`detection/motion.py`)
- Frame-to-frame motion calculation
- Motion pattern analysis
- Threshold-based motion alerts
- Historical motion tracking

#### Fall Detection (`detection/fall.py`)
- Function-based fall detection pipeline
- Confidence smoothing and validation
- Multi-frame confirmation system
- Integration with motion data

#### Seizure Detection (`detection/seizure.py`)
- VSViG-based seizure detection
- Temporal pattern analysis
- Confidence enhancement with motion
- Real-time seizure prediction

### Visualization (`visualization/display.py`)
- Bounding box rendering
- Keypoint and skeleton drawing
- Motion indicators and alerts
- Statistics overlay
- Confidence history graphs

### Alert Management (`alerts/management.py`)
- Multi-channel alert system
- Priority-based alert handling
- Email and webhook notifications
- Alert suppression and cooldown
- Alert history and statistics

## Usage

### Basic Usage
```bash
# Run with default configuration
python main.py

# Run with custom configuration
python main.py --config config.json

# Run with debug logging
python main.py --debug
```

### Configuration Example
```json
{
  "camera": {
    "type": "rtsp",
    "rtsp_url": "rtsp://admin:password@192.168.1.100/cam/realmonitor?channel=1&subtype=0",
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
    "alert_cooldown": 30,
    "email_notifications": true,
    "webhook_notifications": false
  },
  "logging": {
    "level": "INFO",
    "file": "healthcare_monitor.log"
  }
}
```

## Function-based Design Principles

### 1. Single Responsibility
Each function has a single, well-defined purpose:
```python
def calculate_motion_level(frame1, frame2, bbox):
    """Calculate motion level between two frames within bounding box"""
    # Single responsibility: motion calculation only

def enhance_confidence_with_motion(base_confidence, motion_level):
    """Enhance confidence using motion information"""
    # Single responsibility: confidence enhancement only
```

### 2. Pure Functions
Most functions are pure with no side effects:
```python
def smooth_confidence_values(confidence_history, current_confidence):
    """Pure function - same inputs always produce same outputs"""
    # No side effects, predictable behavior
```

### 3. Composability
Functions can be easily combined:
```python
# Compose functions for complex workflows
motion_level = calculate_motion_level(prev_frame, curr_frame, bbox)
enhanced_confidence = enhance_confidence_with_motion(base_conf, motion_level)
smoothed_confidence = smooth_confidence_values(history, enhanced_confidence)
```

### 4. Dependency Injection
Functions receive their dependencies as parameters:
```python
def process_fall_detection(detector, predictor, frame, bbox, history, frames, motion, config):
    """All dependencies injected as parameters"""
    # No hidden dependencies, fully testable
```

## Migration from Class-based System

### Key Changes

1. **State Management**: Moved from instance variables to explicit parameter passing
2. **Initialization**: Separated initialization functions from processing functions
3. **Error Handling**: Centralized error handling with fallback functions
4. **Configuration**: Externalized configuration with function-based access
5. **Logging**: Structured logging with function-based event tracking

### Benefits

- **Improved Testability**: Individual functions can be tested in isolation
- **Better Modularity**: Clear separation of concerns between modules
- **Enhanced Maintainability**: Easier to understand and modify individual components
- **Increased Reusability**: Functions can be reused in different contexts
- **Better Error Handling**: Explicit error handling with fallback mechanisms

## Dependencies

- OpenCV (cv2) - Computer vision operations
- NumPy - Numerical computations
- Ultralytics YOLO - Object detection
- Logging - Event tracking
- Threading - Concurrent camera operations
- JSON/YAML - Configuration management

## Performance Considerations

- Thread-safe frame buffering for camera capture
- Efficient motion calculation using frame differences
- Confidence smoothing to reduce false positives
- Alert cooldown to prevent spam
- Memory-efficient frame processing

## Extensibility

The function-based architecture makes it easy to:
- Add new detection algorithms
- Implement custom visualization functions
- Create new alert notification channels
- Extend configuration options
- Add performance monitoring functions

## Testing

Each module can be tested independently:
```python
# Test motion detection
motion_level = calculate_motion_level(test_frame1, test_frame2, test_bbox)
assert 0 <= motion_level <= 1

# Test confidence smoothing
smoothed = smooth_confidence_values([0.5, 0.6], 0.8)
assert isinstance(smoothed, float)
```

This function-based architecture provides a solid foundation for healthcare monitoring with improved maintainability, testability, and extensibility compared to the original class-based approach.
