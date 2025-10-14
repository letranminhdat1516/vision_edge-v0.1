# infrastructure/camera/camera_device.py
import cv2
import time
from typing import Optional
from models.generated_all import Cameras

class CameraDevice:
    """Đại diện cho 1 camera vật lý, chịu trách nhiệm đọc frame."""

    def __init__(self, cam: Cameras):
        self.meta = cam
        self.rtsp_url = cam.rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self._opened = False

    def open(self) -> bool:
        if not self.rtsp_url:
            print(f"[{self.meta.camera_name}] No RTSP URL configured")
            return False
        print(f"[{self.meta.camera_name}] Connecting to {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self._opened = bool(self.cap and self.cap.isOpened())
        if self._opened:
            print(f"[{self.meta.camera_name}] Connected successfully")
        else:
            print(f"[{self.meta.camera_name}] Failed to connect")
        return self._opened

    def read(self):
        """Đọc 1 frame, trả (success, frame)."""
        if not (self.cap and self._opened):
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            self._opened = False
        return ok, frame

    def reopen_with_backoff(self, max_wait=30):
        """Auto reconnect on connection loss"""
        wait = 1
        while not self.open():
            print(f"[{self.meta.camera_name}] Reconnecting in {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, max_wait)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._opened = False
