"""
Camera Controller - Presentation Layer
FastAPI controller cho camera operations
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ...application.use_cases.start_monitoring_use_case import StartMonitoringUseCase
from ...application.use_cases.process_frame_use_case import ProcessFrameUseCase
from ...application.use_cases.stop_monitoring_use_case import StopMonitoringUseCase

class CameraConfigRequest(BaseModel):
    camera_id: str
    source: str | int
    fps: int = 30

class StartMonitoringRequest(BaseModel):
    cameras: List[CameraConfigRequest]

class CameraController:
    """REST API controller for camera operations"""
    
    def __init__(
        self,
        start_monitoring_use_case: StartMonitoringUseCase,
        process_frame_use_case: ProcessFrameUseCase,
        stop_monitoring_use_case: StopMonitoringUseCase
    ):
        self.start_monitoring_use_case = start_monitoring_use_case
        self.process_frame_use_case = process_frame_use_case
        self.stop_monitoring_use_case = stop_monitoring_use_case
        self.router = APIRouter(prefix="/camera", tags=["Camera"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.post("/start")
        async def start_monitoring(request: StartMonitoringRequest) -> Dict[str, Any]:
            """Start camera monitoring"""
            camera_configs = [
                {
                    "camera_id": camera.camera_id,
                    "source": camera.source,
                    "fps": camera.fps
                }
                for camera in request.cameras
            ]
            
            result = self.start_monitoring_use_case.execute(camera_configs)
            
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result.get("error", "Failed to start monitoring"))
            
            return result
        
        @self.router.get("/frame/{camera_id}")
        async def get_frame(camera_id: str) -> Dict[str, Any]:
            """Get processed frame from specific camera"""
            result = self.process_frame_use_case.execute_single_camera(camera_id)
            
            if not result["success"]:
                raise HTTPException(status_code=404, detail=result.get("error", "Camera not found"))
            
            return result
        
        @self.router.get("/frames")
        async def get_all_frames() -> Dict[str, Any]:
            """Get processed frames from all cameras"""
            result = self.process_frame_use_case.execute_all_cameras()
            return result
        
        @self.router.delete("/stop/{camera_id}")
        async def stop_camera(camera_id: str) -> Dict[str, Any]:
            """Stop specific camera"""
            result = self.stop_monitoring_use_case.execute(camera_id)
            
            if not result["success"]:
                raise HTTPException(status_code=404, detail=result.get("error", "Camera not found"))
            
            return result
        
        @self.router.delete("/stop")
        async def stop_all_cameras() -> Dict[str, Any]:
            """Stop all cameras"""
            result = self.stop_monitoring_use_case.execute()
            return result