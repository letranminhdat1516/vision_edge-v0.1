from typing import Dict, Optional
import numpy as np
from ...domain.repositories.camera_repository import ICameraRepository
from ..camera.camera_manager import CameraManager

class CameraRepository(ICameraRepository):
    """Infrastructure implementation of camera repository"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
    
    def add_camera(self, camera_id: str, source, fps: int = 30) -> bool:
        """Add camera implementation"""
        return self.camera_manager.add_camera(camera_id, source, fps)
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera implementation"""
        return self.camera_manager.remove_camera(camera_id)
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get frame implementation"""
        return self.camera_manager.get_frame(camera_id)
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Get all frames implementation"""
        return self.camera_manager.get_all_frames()
    
    def is_camera_active(self, camera_id: str) -> bool:
        """Check if camera is active implementation"""
        return self.camera_manager.is_camera_active(camera_id)
    
    def stop_all_cameras(self) -> bool:
        """Stop all cameras implementation"""
        return self.camera_manager.stop_all_cameras()