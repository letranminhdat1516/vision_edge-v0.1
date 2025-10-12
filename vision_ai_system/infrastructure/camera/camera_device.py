import cv2
import threading
import numpy as np
from queue import Queue, Empty
from typing import Optional
from dataclasses import dataclass

@dataclass
class CameraConfig:
    camera_id: str
    source: str
    fps: int = 30
    width: int = 640
    height: int = 480

class CameraDevice:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=3)
        self.thread = None
        
    def start(self) -> bool:
        try:
            if self.config.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.config.source))
            else:
                self.cap = cv2.VideoCapture(self.config.source)
                
            if not self.cap.isOpened():
                return False
                
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except:
            return False
    
    def _capture_loop(self):
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(frame)
    
    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None