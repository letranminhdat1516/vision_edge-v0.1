# infrastructure/camera/camera_manager.py
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Any
from models.generated_all import Cameras
from .camera_device import CameraDevice

class CameraManager:
    """Manage multiple cameras concurrently"""

    def __init__(self, on_frame: Callable[[str, Any, datetime], None]):
        self.on_frame = on_frame
        self._threads: Dict[str, threading.Thread] = {}
        self._devices: Dict[str, CameraDevice] = {}
        self._stop = threading.Event()

    def load_from_list(self, cams: List[Cameras]):
        """Create device for each camera"""
        for cam in cams:
            self._devices[str(cam.camera_id)] = CameraDevice(cam)

    def start_all(self):
        """Run all cameras in separate threads"""
        for dev in self._devices.values():
            print(f"Starting thread for {dev.meta.camera_name}")
            t = threading.Thread(target=self._loop, args=(dev,), daemon=True)
            t.start()
            self._threads[str(dev.meta.camera_id)] = t

    def _loop(self, dev: CameraDevice):
        if not dev.open():
            dev.reopen_with_backoff()
        frames = 0; t0 = time.time()
        while not self._stop.is_set():
            ok, frame = dev.read()
            if not ok:
                dev.reopen_with_backoff()
                continue
            frames += 1
            ts = datetime.utcnow()
            self.on_frame(str(dev.meta.camera_id), frame, ts)
        dev.release()

    def stop_all(self):
        """Stop all cameras"""
        self._stop.set()
        for t in self._threads.values():
            if t.is_alive():
                try:
                    t.join(timeout=1)
                except KeyboardInterrupt:
                    break
        self._threads.clear()
        for dev in self._devices.values():
            dev.release()
