from typing import Dict, Optional, List, Callable
import numpy as np
from .camera_device import CameraDevice, CameraConfig
from .frame_processor import FrameProcessor

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, CameraDevice] = {}
        self.frame_processor = FrameProcessor()
        self.frame_callbacks: Dict[str, List[Callable]] = {}
    
    def add_camera(self, camera_id: str, source: str, fps: int = 30) -> bool:
        config = CameraConfig(camera_id, source, fps)
        camera = CameraDevice(config)
        
        if camera.start():
            self.cameras[camera_id] = camera
            self.frame_callbacks[camera_id] = []
            return True
        return False
    
    def remove_camera(self, camera_id: str) -> bool:
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            if camera_id in self.frame_callbacks:
                del self.frame_callbacks[camera_id]
            return True
        return False
    
    def get_frame(self, camera_id: str, preprocess: bool = True) -> Optional[np.ndarray]:
        if camera_id not in self.cameras:
            return None
            
        frame = self.cameras[camera_id].get_frame()
        if frame is not None and preprocess:
            frame = self.frame_processor.preprocess(frame)
            
        return frame
    
    def get_all_frames(self, preprocess: bool = True) -> Dict[str, np.ndarray]:
        frames = {}
        for camera_id in self.cameras.keys():
            frame = self.get_frame(camera_id, preprocess)
            if frame is not None:
                frames[camera_id] = frame
        return frames
    
    def get_keyframe(self, camera_id: str, frame_count: int = 5) -> Optional[np.ndarray]:
        if camera_id not in self.cameras:
            return None
            
        frames = []
        for _ in range(frame_count):
            frame = self.get_frame(camera_id, preprocess=False)
            if frame is not None:
                frames.append(frame)
        
        if frames:
            keyframe = self.frame_processor.extract_keyframe(frames)
            return self.frame_processor.preprocess(keyframe)
        return None
    
    def add_frame_callback(self, camera_id: str, callback: Callable[[str, np.ndarray], None]):
        if camera_id in self.frame_callbacks:
            self.frame_callbacks[camera_id].append(callback)
    
    def get_camera_ids(self) -> List[str]:
        return list(self.cameras.keys())
    
    def is_camera_active(self, camera_id: str) -> bool:
        return camera_id in self.cameras and self.cameras[camera_id].is_running
    
    def stop_all_cameras(self) -> bool:
        try:
            for camera in self.cameras.values():
                camera.stop()
            self.cameras.clear()
            self.frame_callbacks.clear()
            return True
        except Exception:
            return False