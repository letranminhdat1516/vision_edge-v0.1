#!/usr/bin/env python3
"""
Core Configuration Management
Function-based healthcare monitor configuration - Fall Detection Only
"""

from typing import Dict, Any


def get_camera_config() -> Dict[str, Any]:
    """Get IMOU camera configuration"""
    return {
        'rtsp_url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
        'buffer_size': 1,
        'fps': 30,
        'resolution': (640, 480),
        'auto_reconnect': True
    }


def get_fall_detection_config() -> Dict[str, Any]:
    """Get fall detection configuration"""
    return {
        'model_path': 'models/ai_models/lite-model_movenet_singlepose_thunder_3.tflite',
        'threshold': 0.85,  # Tăng threshold để giảm false positive
        'confidence_threshold': 0.6
    }
