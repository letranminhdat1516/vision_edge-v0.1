# runtime/monitor_runner.py
import os
import cv2
import threading
import time
from datetime import datetime
from core.services.camera_service import CameraService
from infrastructure.camera.camera_manager import CameraManager

class MonitorRunner:
    """Camera monitoring pipeline with viewer support"""

    def __init__(self, cam_service: CameraService, cam_manager: CameraManager):
        self.cam_service = cam_service
        self.cam_manager = cam_manager
        self.show_video = os.getenv("MONITOR_SHOW_VIDEO", "false").lower() == "true"
        self.camera_windows = {}

    def on_frame(self, camera_id: str, frame, ts: datetime):
        """Callback when new frame arrives"""
        # Process frame for AI
        # self.cam_service.process_frame(camera_id, frame, ts)
        
        # Show video if enabled
        if self.show_video and camera_id not in self.camera_windows:
            self.start_video_display(camera_id, frame)
        elif self.show_video and camera_id in self.camera_windows:
            self.update_video_display(camera_id, frame)

    def start_video_display(self, camera_id: str, frame):
        """Start video display for a camera"""
        # Find camera name
        camera_name = f"Camera {camera_id[:8]}"
        for cam in self.cam_service.get_active_cameras():
            if str(cam.camera_id) == camera_id:
                camera_name = cam.camera_name
                break
        
        window_name = camera_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        self.camera_windows[camera_id] = window_name
        print(f"Video display started for {camera_name}")

    def update_video_display(self, camera_id: str, frame):
        """Update video display with new frame"""
        if camera_id in self.camera_windows:
            window_name = self.camera_windows[camera_id]
            
            # Display frame without timestamp overlay
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    def run(self):
        """Start camera monitoring"""
        print(f"Monitor video display: {'ON' if self.show_video else 'OFF'}")
        
        # Get all active cameras first
        all_cams = self.cam_service.get_active_cameras()
        print(f"Total active cameras: {len(all_cams)}")
        
        # Filter by MONITOR_ALLOWED_CAM_IDS if configured
        allowed_ids = os.getenv("MONITOR_ALLOWED_CAM_IDS", "").split(",")
        allowed_ids = [id.strip() for id in allowed_ids if id.strip()]
        
        if allowed_ids:
            print(f"Filtering by allowed IDs: {allowed_ids}")
            cams = [cam for cam in all_cams if str(cam.camera_id) in allowed_ids]
            print(f"Cameras after filtering: {len(cams)}")
        else:
            cams = all_cams
            print("No filter applied - using all active cameras")
        
        if not cams:
            print("No cameras found to monitor!")
            return
        
        for cam in cams:
            print(f"Will monitor: {cam.camera_name} (ID: {str(cam.camera_id)[:8]}...)")
        
        self.cam_manager.on_frame = self.on_frame
        self.cam_manager.load_from_list(cams)
        self.cam_manager.start_all()
        
        if self.show_video:
            print("Video windows will open automatically. Press 'q' in any window to close it.")

    def stop(self):
        """Stop monitoring"""
        self.cam_manager.stop_all()
        if self.show_video:
            cv2.destroyAllWindows()
            self.camera_windows.clear()
