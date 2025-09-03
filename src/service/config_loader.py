"""
Configuration Loader Service
Loads system configuration and detection settings from JSON files
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Service for loading and managing configuration files"""
    
    def __init__(self, config_dir: Optional[str] = None):
        # Use config directory from environment variable or constructor parameter
        if config_dir is None:
            config_dir = os.environ.get('VISION_CONFIG_DIR')
            if config_dir is None:
                raise ValueError(
                    "❌ CRITICAL: No config directory specified. "
                    "Please set VISION_CONFIG_DIR environment variable or pass config_dir parameter."
                )
        
        self.config_dir = Path(config_dir)
        
        # Verify config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"❌ CRITICAL: Config directory not found: {self.config_dir}. "
                f"Please create the directory and add required config files."
            )
        
        self._system_config = None
        self._detection_settings = None
        
    def load_system_config(self) -> Dict[str, Any]:
        """Load system configuration from config.json"""
        if self._system_config is None:
            config_path = self.config_dir / "config.json"
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._system_config = json.load(f)
                logger.info(f"✅ System config loaded from {config_path}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"❌ CRITICAL: System config file not found at {config_path}. "
                    f"Please ensure config.json exists with all required settings."
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"❌ CRITICAL: Invalid JSON in config file {config_path}: {e}. "
                    f"Please fix the JSON syntax."
                )
                
        return self._system_config
    
    def load_detection_settings(self) -> Dict[str, Any]:
        """Load detection settings from detection_settings.json"""
        if self._detection_settings is None:
            settings_path = self.config_dir / "detection_settings.json"
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    self._detection_settings = json.load(f)
                logger.info(f"✅ Detection settings loaded from {settings_path}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"❌ CRITICAL: Detection settings file not found at {settings_path}. "
                    f"Please ensure detection_settings.json exists with all required thresholds."
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"❌ CRITICAL: Invalid JSON in detection settings {settings_path}: {e}. "
                    f"Please fix the JSON syntax."
                )
                
        return self._detection_settings
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        config = self.load_system_config()
        return config.get('api', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        config = self.load_system_config()
        return config.get('database', {})
    
    def get_detection_thresholds(self, event_type: str) -> Dict[str, float]:
        """Get detection thresholds for specific event type"""
        settings = self.load_detection_settings()
        return settings.get('detection_thresholds', {}).get(event_type, {})
    
    def get_camera_settings(self, camera_id: str) -> Dict[str, Any]:
        """Get camera-specific settings"""
        settings = self.load_detection_settings()
        camera_configs = settings.get('camera_specific', {})
        
        # Return specific camera config or default
        return camera_configs.get(camera_id, camera_configs.get('default', {}))
    
    def apply_camera_sensitivity(self, confidence: float, event_type: str, camera_id: str) -> float:
        """Apply camera-specific sensitivity multiplier to confidence"""
        camera_settings = self.get_camera_settings(camera_id)
        
        # Get multiplier from config, no defaults
        if event_type == 'fall':
            multiplier_key = 'fall_sensitivity_multiplier'
        elif event_type == 'seizure':
            multiplier_key = 'seizure_sensitivity_multiplier'
        else:
            raise ValueError(f"❌ Unknown event type: {event_type}")
            
        if multiplier_key not in camera_settings:
            raise KeyError(
                f"❌ Missing {multiplier_key} for camera {camera_id} in detection_settings.json"
            )
            
        multiplier = camera_settings[multiplier_key]
            
        # Apply multiplier (lower multiplier = more sensitive)
        adjusted_confidence = confidence / multiplier
        return min(1.0, max(0.0, adjusted_confidence))  # Clamp to [0, 1]
    
    def reload_config(self):
        """Reload configuration from files"""
        self._system_config = None
        self._detection_settings = None
        logger.info("🔄 Configuration reloaded")

# Global config loader instance  
def create_config_loader() -> ConfigLoader:
    """Create config loader - requires VISION_CONFIG_DIR environment variable"""
    config_dir = os.environ.get('VISION_CONFIG_DIR')
    
    if config_dir is None:
        # Temporary fallback for development
        default_config_dir = Path(__file__).parent.parent / "config"
        config_dir = str(default_config_dir)
        logger.warning(f"⚠️ VISION_CONFIG_DIR not set, using fallback: {config_dir}")
    
    return ConfigLoader(config_dir)

config_loader = create_config_loader()
