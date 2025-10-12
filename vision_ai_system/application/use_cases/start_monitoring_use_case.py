from typing import List, Dict
from ...domain.services.camera_service import CameraService

class StartMonitoringUseCase:
    """Use case for starting camera monitoring workflow"""
    
    def __init__(self, camera_service: CameraService):
        self.camera_service = camera_service
    
    def execute(self, camera_configs: List[dict]) -> dict:
        """Execute start monitoring workflow"""
        try:
            # Validate input
            if not camera_configs:
                return {
                    "success": False,
                    "error": "No camera configurations provided"
                }
            
            # Execute business logic
            success = self.camera_service.start_monitoring(camera_configs)
            
            if success:
                active_cameras = self.camera_service.get_all_active_cameras()
                return {
                    "success": True,
                    "active_cameras": active_cameras,
                    "message": f"Successfully started monitoring {len(active_cameras)} cameras"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to start camera monitoring"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }