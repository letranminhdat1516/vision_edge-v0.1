"""
Database Configuration Service
Replaces config_loader to use database and environment variables instead of config.json
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfigService:
    """Configuration service that uses database and environment variables"""
    
    def __init__(self):
        self._detection_settings = None
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration from environment variables"""
        return {
            'connection': {
                'host': os.getenv('DB_HOST', 'aws-1-ap-southeast-1.pooler.supabase.com'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER', 'postgres.undznprwlqjpnxqsgyiv'),
                'password': os.getenv('DB_PASSWORD', '12345678')
            },
            'url': os.getenv('DATABASE_URL', ''),
            'logging': os.getenv('DB_LOGGING', 'false').lower() == 'true'
        }
    
    def load_system_config(self) -> Dict[str, Any]:
        """
        Load system configuration from environment variables and defaults
        Replaces config.json dependency
        """
        return {
            'database': {
                'connection': self.get_database_config()['connection'],
                'default_ids': {
                    'user_id': os.getenv('DEFAULT_USER_ID', '37cbad15-483d-42ff-b07d-fbf3cd1cc863'),
                    'room_id': 'default_room',
                    'subscription_id': 'default_subscription'
                }
            },
            'system': {
                'logging': {
                    'level': os.getenv('LOG_LEVEL', 'INFO'),
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file_logging': os.getenv('FILE_LOGGING', 'false').lower() == 'true',
                    'console_logging': True
                },
                'ai_models': {
                    'fall_detection': {
                        'enabled': os.getenv('FALL_DETECTION_ENABLED', 'true').lower() == 'true',
                        'confidence_threshold': float(os.getenv('FALL_DETECTION_CONFIDENCE', '0.7')),
                        'device': os.getenv('FALL_DETECTION_DEVICE', 'cpu'),
                        'cooldown_seconds': float(os.getenv('FALL_DETECTION_COOLDOWN_SECONDS', '5.0'))
                    },
                    'seizure_detection': {
                        'enabled': os.getenv('SEIZURE_DETECTION_ENABLED', 'true').lower() == 'true',
                        'confidence_threshold': float(os.getenv('SEIZURE_DETECTION_CONFIDENCE', '0.6'))
                    }
                },
                'performance': {
                    'max_concurrent_cameras': int(os.getenv('MONITOR_MAX_WORKERS', '2')),
                    'frame_processing_timeout': 30,
                    'memory_cleanup_interval': 300,
                    'queue_size': int(os.getenv('MONITOR_QUEUE_SIZE', '200'))
                }
            },
            'paths': {
                'alert_images': [
                    'data/saved_frames/alerts',
                    'examples/data/saved_frames/alerts',
                    'uploads/alerts'
                ],
                'logs': 'logs',
                'uploads': 'uploads'
            },
            'api': {
                'base_url': os.getenv('API_BASE_URL', 'https://healthcare-system.com'),
                'snapshot_endpoint': '/snapshots',
                'alert_endpoint': '/alerts'
            }
        }
    
    def load_detection_settings(self) -> Dict[str, Any]:
        """Load detection settings from JSON file (keep this as is)"""
        if self._detection_settings is None:
            # Import here to avoid circular imports
            import json
            from pathlib import Path
            
            config_dir = Path(os.getenv('VISION_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
            settings_path = config_dir / "detection_settings.json"
            
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    self._detection_settings = json.load(f)
            except FileNotFoundError:
                # Default fallback settings
                self._detection_settings = {
                    "detection_thresholds": {
                        "fall": {
                            "severity_mapping": {"high": 0.7, "medium": 0.5, "low": 0.3},
                            "notification_threshold": 0.6
                        },
                        "seizure": {
                            "severity_mapping": {"high": 0.6, "medium": 0.4, "low": 0.2},
                            "notification_threshold": 0.5
                        }
                    }
                }
        
        return self._detection_settings
    
    def get_detection_thresholds(self, event_type: str) -> Dict[str, Any]:
        """Get detection thresholds for specific event type"""
        settings = self.load_detection_settings()
        return settings.get('detection_thresholds', {}).get(event_type, {})
    
    def get_camera_settings(self, camera_id: str) -> Dict[str, Any]:
        """Get camera-specific settings"""
        settings = self.load_detection_settings()
        camera_configs = settings.get('camera_specific', {})
        return camera_configs.get(camera_id, camera_configs.get('default', {}))

# Create global instance to replace config_loader
db_config_service = DatabaseConfigService()

# Backward compatibility aliases
config_loader = db_config_service