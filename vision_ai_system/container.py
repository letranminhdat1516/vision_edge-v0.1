"""
Dependency Injection Container for DDD Vision AI System
"""
from .domain.repositories.camera_repository import ICameraRepository
from .domain.services.camera_service import CameraService
from .infrastructure.persistence.camera_repository import CameraRepository
from .application.use_cases.start_monitoring_use_case import StartMonitoringUseCase
from .application.use_cases.process_frame_use_case import ProcessFrameUseCase
from .application.use_cases.stop_monitoring_use_case import StopMonitoringUseCase

class DIContainer:
    """Dependency Injection Container for clean architecture"""
    
    def __init__(self):
        self._instances = {}
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        """Setup dependency injection"""
        # Infrastructure layer
        self._instances['camera_repository'] = CameraRepository()
        
        # Domain layer
        self._instances['camera_service'] = CameraService(
            self._instances['camera_repository']
        )
        
        # Application layer
        self._instances['start_monitoring_use_case'] = StartMonitoringUseCase(
            self._instances['camera_service']
        )
        self._instances['process_frame_use_case'] = ProcessFrameUseCase(
            self._instances['camera_service']
        )
        self._instances['stop_monitoring_use_case'] = StopMonitoringUseCase(
            self._instances['camera_service']
        )
    
    def get_camera_repository(self) -> ICameraRepository:
        return self._instances['camera_repository']
    
    def get_camera_service(self) -> CameraService:
        return self._instances['camera_service']
    
    def get_start_monitoring_use_case(self) -> StartMonitoringUseCase:
        return self._instances['start_monitoring_use_case']
    
    def get_process_frame_use_case(self) -> ProcessFrameUseCase:
        return self._instances['process_frame_use_case']
    
    def get_stop_monitoring_use_case(self) -> StopMonitoringUseCase:
        return self._instances['stop_monitoring_use_case']

# Global container instance
container = DIContainer()