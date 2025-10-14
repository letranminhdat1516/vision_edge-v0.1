# core/services/camera_service.py
import os
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
from models.generated_all import Cameras  # Import from generated_all

from core.services.pose_detection_service import PoseDetectionDomainService
from core.entities.pose_detection import PoseDetectionResult, FallDetectionEvent
from infrastructure.ai_models.pose_detection_engine_threadsafe import PoseDetectionEngine, PoseVisualizationService


class CameraService:
    """Xử lý nghiệp vụ camera với pose detection và fall detection."""

    def __init__(self, repo=None, pose_detection_service=None):
        """
        Initialize CameraService with flexible parameters.
        
        Args:
            repo: Camera repository for database operations (optional)
            pose_detection_service: AI pose detection service (optional)
        """
        self.repo = repo
        
        # Initialize AI components
        if pose_detection_service:
            self.pose_service = pose_detection_service
            self.pose_engine = pose_detection_service.pose_engine
        else:
            # Fallback: create AI components
            self.pose_engine = PoseDetectionEngine()
            self.pose_service = PoseDetectionDomainService(self.pose_engine)
        
        self.visualization_service = PoseVisualizationService()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []

    def get_active_cameras(self) -> List[Cameras]:
        """Trả về danh sách camera đang active."""
        if not self.repo:
            raise ValueError("Repository not initialized. Cannot get cameras from database.")
        
        cams = self.repo.get_all()
        return [c for c in cams if c.status == "active"]

    def update_status(self, camera_id: str, status: str):
        """Cập nhật trạng thái camera (active/offline)."""
        if not self.repo:
            raise ValueError("Repository not initialized. Cannot update camera status.")
        
        self.repo.update_status(camera_id, status)

    def process_frame(self, camera_id: str, frame: np.ndarray, timestamp: datetime) -> dict:
        """
        Process frame from camera với pose detection
        
        Args:
            camera_id: ID của camera
            frame: OpenCV frame
            timestamp: Thời gian frame
            
        Returns:
            Dict chứa kết quả processing
        """
        self.frame_count += 1
        
        try:
            # Process frame để detect pose và fall
            pose_result, fall_event = self.pose_service.process_frame_for_monitoring(frame, camera_id)
            
            # Tạo response
            result = {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'frame_processed': True,
                'pose_detected': pose_result is not None,
                'fall_detected': fall_event is not None,
                'pose_result': pose_result,
                'fall_event': fall_event,
                'frame_count': self.frame_count
            }
            
            # Thêm stats nếu có pose result
            if pose_result:
                result['stats'] = self.pose_service.get_monitoring_stats(pose_result)
                self.processing_times.append(pose_result.inference_time)
                
                # Log fall detection
                if fall_event:
                    print(f"🚨 FALL DETECTED - Camera: {camera_id}, Probability: {fall_event.fall_probability:.2f}, Alert: {fall_event.alert_level}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing frame for camera {camera_id}: {e}")
            return {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'frame_processed': False,
                'error': str(e),
                'frame_count': self.frame_count
            }

    def visualize_frame(
        self, 
        frame: np.ndarray, 
        pose_result: Optional[PoseDetectionResult] = None,
        show_skeleton: bool = True,
        show_labels: bool = True,
        show_info: bool = True,
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Vẽ visualization lên frame
        
        Args:
            frame: OpenCV frame
            pose_result: Kết quả pose detection
            show_skeleton: Hiện skeleton
            show_labels: Hiện labels
            show_info: Hiện info panel
            fps: FPS để hiển thị
            
        Returns:
            Frame đã vẽ visualization
        """
        if not pose_result:
            return frame
            
        # Use unified draw_pose_on_frame method
        return self.visualization_service.draw_pose_on_frame(
            frame, pose_result, show_skeleton, show_labels, show_info, fps
        )

    def get_processing_stats(self) -> dict:
        """Lấy thống kê processing"""
        if not self.processing_times:
            return {
                'frames_processed': self.frame_count,
                'avg_inference_time': 0.0,
                'pose_engine_ready': self.pose_engine.is_ready()
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            'frames_processed': self.frame_count,
            'avg_inference_time': avg_time,
            'avg_inference_time_ms': avg_time * 1000,
            'pose_engine_ready': self.pose_engine.is_ready(),
            'total_detections': len(self.processing_times)
        }

    def is_ai_ready(self) -> bool:
        """Kiểm tra AI engine đã sẵn sàng chưa"""
        return self.pose_engine.is_ready()
