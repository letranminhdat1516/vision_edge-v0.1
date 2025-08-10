#!/usr/bin/env python3
"""
Core Configuration Management
Function-based healthcare monitor configuration
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


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


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the healthcare monitor"""
    return {
        'camera': {
            'type': 'rtsp',
            'rtsp_url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
            'width': 640,
            'height': 480,
            'fps': 15,
            'device_index': 0,
            'buffer_size': 1,
            'auto_reconnect': True
        },
        'detection': {
            'yolo_model': 'yolov8s.pt',
            'confidence_threshold': 0.5,
            'fall_threshold': 0.7,
            'seizure_threshold': 0.7,
            'confirmation_frames': 5,
            'max_history': 10,
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
            }
        },
        'alerts': {
            'alert_cooldown': 30,
            'max_history': 1000,
            'email_notifications': False,
            'webhook_notifications': False,
            'log_notifications': True
        },
        'logging': {
            'level': 'INFO',
            'file': 'healthcare_monitor.log',
            'max_file_size': '10MB',
            'backup_count': 5
        },
        'display': {
            'show_keypoints': True,
            'show_statistics': True,
            'show_bounding_boxes': True,
            'show_motion_indicators': True
        }
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults
    
    Args:
        config_path: Path to configuration file (JSON)
        
    Returns:
        Configuration dictionary
    """
    default_config = get_default_config()
    
    if config_path is None:
        return default_config
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        
        # Merge file config with defaults
        merged_config = default_config.copy()
        for key, value in file_config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        return merged_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
        
    Returns:
        True if successful
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
