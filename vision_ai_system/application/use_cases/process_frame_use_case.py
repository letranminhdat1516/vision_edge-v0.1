from typing import Optional, Dict
import numpy as np
from ...domain.services.camera_service import CameraService

class ProcessFrameUseCase:
    """Use case for processing camera frames workflow"""
    
    def __init__(self, camera_service: CameraService):
        self.camera_service = camera_service
    
    def execute_single_camera(self, camera_id: str) -> dict:
        """Process frame from single camera"""
        try:
            frame = self.camera_service.get_processed_frame(camera_id)
            
            if frame is not None:
                return {
                    "success": True,
                    "camera_id": camera_id,
                    "frame_shape": frame.shape,
                    "frame_ready": True,
                    "message": f"Frame processed for camera {camera_id}"
                }
            else:
                return {
                    "success": False,
                    "camera_id": camera_id,
                    "frame_ready": False,
                    "error": "No valid frame available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "camera_id": camera_id,
                "error": f"Frame processing error: {str(e)}"
            }
    
    def execute_all_cameras(self) -> dict:
        """Process frames from all active cameras"""
        try:
            frames = self.camera_service.get_all_processed_frames()
            active_cameras = self.camera_service.get_all_active_cameras()
            
            return {
                "success": True,
                "total_cameras": len(active_cameras),
                "frames_received": len(frames),
                "camera_status": {camera_id: camera_id in frames for camera_id in active_cameras},
                "message": f"Processed {len(frames)}/{len(active_cameras)} camera frames"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Batch frame processing error: {str(e)}"
            }