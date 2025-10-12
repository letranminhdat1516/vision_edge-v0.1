import cv2
import numpy as np
from typing import Tuple

class FrameProcessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        self.target_size = target_size
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = self._resize_frame(frame)
        frame = self._reduce_noise(frame)
        frame = self._normalize_lighting(frame)
        return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self.target_size)
    
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (3, 3), 0)
    
    def _normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def extract_keyframe(self, frames: list, threshold: float = 0.3) -> np.ndarray:
        if len(frames) < 2:
            return frames[0] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
            
        max_diff = 0
        keyframe = frames[0]
        
        for i in range(1, len(frames)):
            diff = self._calculate_frame_difference(frames[i-1], frames[i])
            if diff > max_diff:
                max_diff = diff
                keyframe = frames[i]
        
        return keyframe if max_diff > threshold else frames[0]
    
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(diff.mean()) / 255.0