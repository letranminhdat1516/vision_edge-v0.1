from typing import Optional
from ...domain.services.camera_service import CameraService

class StopMonitoringUseCase:
    """Use case for stopping camera monitoring workflow"""
    
    def __init__(self, camera_service: CameraService):
        self.camera_service = camera_service
    
    def execute(self, camera_id: Optional[str] = None) -> dict:
        """Execute stop monitoring workflow"""
        try:
            if camera_id:
                # Stop specific camera
                success = self.camera_service.stop_monitoring(camera_id)
                
                if success:
                    return {
                        "success": True,
                        "camera_id": camera_id,
                        "message": f"Successfully stopped monitoring camera {camera_id}"
                    }
                else:
                    return {
                        "success": False,
                        "camera_id": camera_id,
                        "error": f"Failed to stop camera {camera_id} or camera not found"
                    }
            else:
                # Stop all cameras
                active_cameras = self.camera_service.get_all_active_cameras()
                stopped_cameras = []
                
                for cam_id in active_cameras:
                    if self.camera_service.stop_monitoring(cam_id):
                        stopped_cameras.append(cam_id)
                
                return {
                    "success": len(stopped_cameras) == len(active_cameras),
                    "stopped_cameras": stopped_cameras,
                    "total_cameras": len(active_cameras),
                    "message": f"Stopped {len(stopped_cameras)}/{len(active_cameras)} cameras"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Stop monitoring error: {str(e)}"
            }