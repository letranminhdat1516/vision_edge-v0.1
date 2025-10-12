from typing import Dict, Optional, List
import numpy as np
import cv2
from ..repositories.camera_repository import ICameraRepository

class CameraService:
    """Domain service for camera business logic"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
        self.active_cameras: Dict[str, bool] = {}
        self.frame_quality_threshold = 0.7
    
    def start_monitoring(self, camera_configs: List[dict]) -> bool:
        """Business logic for starting camera monitoring"""
        for config in camera_configs:
            if not self.validate_camera_config(config):
                return False
            
            camera_id = config['camera_id']
            if self.camera_repository.add_camera(camera_id, config['source'], config.get('fps', 30)):
                self.active_cameras[camera_id] = True
        
        return len(self.active_cameras) > 0
    
    def stop_monitoring(self, camera_id: str) -> bool:
        """Business logic for stopping camera monitoring"""
        if camera_id in self.active_cameras:
            success = self.camera_repository.remove_camera(camera_id)
            if success:
                del self.active_cameras[camera_id]
            return success
        return False
    
    def get_processed_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get frame with business validation"""
        if camera_id not in self.active_cameras:
            return None
        
        raw_frame = self.camera_repository.get_frame(camera_id)
        if raw_frame is None:
            return None
        
        # Business rule: Only return high quality frames
        if self.assess_frame_quality(raw_frame) < self.frame_quality_threshold:
            return None
        
        return raw_frame
    
    def validate_camera_config(self, config: dict) -> bool:
        """Business rules for camera configuration"""
        required_fields = ['camera_id', 'source']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Business rule: FPS must be reasonable
        fps = config.get('fps', 30)
        if fps < 1 or fps > 60:
            return False
        
        return True
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """Business logic for frame quality assessment"""
        if frame is None or frame.size == 0:
            return 0.0
        
        # Simple quality metrics
        # 1. Check brightness
        brightness = np.mean(frame)
        brightness_score = 1.0 if 50 <= brightness <= 200 else 0.5
        
        # 2. Check contrast
        contrast = np.std(frame)
        contrast_score = 1.0 if contrast > 30 else 0.5
        
        # 3. Check sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = 1.0 if sharpness > 100 else 0.5
        
        return (brightness_score + contrast_score + sharpness_score) / 3.0
    
    def get_all_active_cameras(self) -> List[str]:
        """Get list of active camera IDs"""
        return list(self.active_cameras.keys())
    
    def get_all_processed_frames(self) -> Dict[str, np.ndarray]:
        """Get all frames with business validation"""
        frames = {}
        for camera_id in self.active_cameras.keys():
            frame = self.get_processed_frame(camera_id)
            if frame is not None:
                frames[camera_id] = frame
        return frames