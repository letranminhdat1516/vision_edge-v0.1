from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np

class ICameraRepository(ABC):
    """Domain interface for camera operations"""
    
    @abstractmethod
    def add_camera(self, camera_id: str, source: str, fps: int) -> bool:
        pass
    
    @abstractmethod
    def remove_camera(self, camera_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def is_camera_active(self, camera_id: str) -> bool:
        pass
    
    @abstractmethod
    def stop_all_cameras(self):
        pass