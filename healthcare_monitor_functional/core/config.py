#!/usr/bin/env python3
"""
Core Configuration Management
Function-based healthcare monitor configuration
"""

from pathlib import Path
from typing import Dict, Any


def get_camera_config() -> Dict[str, Any]:
    """Get IMOU camera configuration"""
    return {
        'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
        'buffer_size': 1,
        'fps': 15,
        'resolution': (640, 480),
        'auto_reconnect': True
    }


def get_detection_config() -> Dict[str, Any]:
    """Get detection system configuration"""
    return {
        'fall': {
            'confidence_threshold': 0.4,
            'confirmation_frames': 1,
            'threshold': 0.3
        },
        'seizure': {
            'confidence_threshold': 0.7,
            'temporal_window': 25,
            'alert_threshold': 0.7,
            'warning_threshold': 0.5,
            'confirmation_frames': 5,
            'threshold': 0.6
        },
        'motion': {
            'threshold': 120,
            'significant_threshold': 0.3,
            'max_history': 10
        }
    }


def get_api_config() -> Dict[str, Any]:
    """Get API integration configuration"""
    return {
        'websocket': {
            'url': 'ws://localhost:8086',
            'enabled': True
        },
        'demo_api': {
            'url': 'http://localhost:8003/api/demo/add-event',
            'enabled': True
        },
        'mobile_api': {
            'url': 'http://localhost:8002/api/events',
            'enabled': True
        }
    }


def get_display_config() -> Dict[str, Any]:
    """Get display configuration"""
    return {
        'show_keypoints': True,
        'show_statistics': True,
        'show_dual_windows': True,
        'statistics_panel': {
            'width': 300,
            'height': 350,
            'position': 'top-right'
        }
    }


def get_storage_config() -> Dict[str, Path]:
    """Get storage paths configuration"""
    base_path = Path("healthcare_monitor_functional/data")
    
    paths = {
        'alerts': base_path / "saved_frames" / "alerts",
        'detections': base_path / "saved_frames" / "detections", 
        'keyframes': base_path / "saved_frames" / "keyframes",
        'logs': Path("healthcare_monitor_functional/logs")
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return {
        'processor_config': 120,
        'frame_skip_threshold': 0.8,
        'memory_cleanup_interval': 100,
        'max_detection_history': 10
    }
