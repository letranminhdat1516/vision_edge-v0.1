"""
Database Camera Configuration Migration Helper
Helps existing services transition from hardcoded to database-based camera configs
"""

from typing import Dict, List, Optional, Union
from service.camera_config_service import camera_config_service
from camera.config import IMOUCameraConfig

class DatabaseCameraConfig:
    """Helper class to provide database-based camera configurations"""
    
    @staticmethod
    def get_camera_config(camera_id: Optional[str] = None) -> Dict:
        """
        Get camera configuration for service usage
        
        Args:
            camera_id: Specific camera ID, if None returns first available camera
            
        Returns:
            Dict with camera configuration compatible with existing services
        """
        if camera_id:
            camera_data = camera_config_service.get_camera_by_id(camera_id)
        else:
            cameras = camera_config_service.get_all_cameras()
            camera_data = cameras[0] if cameras else None
        
        if not camera_data:
            raise ValueError(f"Camera not found: {camera_id or 'any'}")
        
        # Convert to service-compatible format
        return {
            'url': camera_data['rtsp_url'],
            'camera_id': camera_data['id'],
            'camera_name': camera_data['name'],
            'resolution': tuple(map(int, camera_data.get('resolution', '640x480').split('x'))),
            'fps': camera_data.get('fps', 30),
            'buffer_size': 1,
            'auto_reconnect': True,
            'location': camera_data.get('location', ''),
            'type': camera_data.get('type', 'ip')
        }
    
    @staticmethod
    def get_all_camera_configs() -> List[Dict]:
        """Get all camera configurations"""
        cameras = camera_config_service.get_all_cameras()
        return [DatabaseCameraConfig.get_camera_config(cam['id']) for cam in cameras]
    
    @staticmethod
    def get_imou_camera_config(camera_id: Optional[str] = None) -> IMOUCameraConfig:
        """
        Get IMOUCameraConfig object from database
        
        Args:
            camera_id: Specific camera ID, if None returns first available camera
            
        Returns:
            IMOUCameraConfig object
        """
        if camera_id:
            camera_data = camera_config_service.get_camera_by_id(camera_id)
        else:
            cameras = camera_config_service.get_all_cameras()
            camera_data = cameras[0] if cameras else None
        
        if not camera_data:
            # Fallback to default config (no deprecated from_env)
            return IMOUCameraConfig(
                rtsp_url='',
                username='admin',
                password='',
                ip_address='',
                port=554,
                frame_width=640,
                frame_height=480,
                fps=30,
                buffer_size=1,
                confidence_threshold=0.5
            )
        
        return IMOUCameraConfig.from_database_camera(camera_data)
    
    @staticmethod
    def get_allowed_camera_ids() -> List[str]:
        """Get list of allowed camera IDs"""
        return camera_config_service.get_allowed_camera_ids()
    
    @staticmethod
    def validate_camera_access(camera_id: str) -> bool:
        """Check if camera ID is allowed/accessible"""
        allowed_ids = DatabaseCameraConfig.get_allowed_camera_ids()
        return camera_id in allowed_ids
    
    @staticmethod
    def get_rtsp_urls() -> Dict[str, str]:
        """Get mapping of camera_id -> rtsp_url for all cameras"""
        cameras = camera_config_service.get_all_cameras()
        return {cam['id']: cam['rtsp_url'] for cam in cameras}

# Convenience functions for backward compatibility
def get_primary_camera_config() -> Dict:
    """Get primary camera configuration (first available camera)"""
    return DatabaseCameraConfig.get_camera_config()

def get_camera_rtsp_url(camera_id: Optional[str] = None) -> str:
    """Get RTSP URL for specific camera or primary camera"""
    config = DatabaseCameraConfig.get_camera_config(camera_id)
    return config['url']

def get_all_rtsp_urls() -> Dict[str, str]:
    """Get all camera RTSP URLs as dict {camera_id: rtsp_url}"""
    return DatabaseCameraConfig.get_rtsp_urls()