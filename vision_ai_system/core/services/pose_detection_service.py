"""
Domain Service cho Pose Detection
Orchestrate việc detect pose và tạo events
"""
from typing import Optional, List
import numpy as np
from datetime import datetime

from core.entities.pose_detection import PoseDetectionResult, FallDetectionEvent
from infrastructure.ai_models.pose_detection_engine_threadsafe import PoseDetectionEngine


class PoseDetectionDomainService:
    """
    Domain service xử lý logic nghiệp vụ pose detection
    """
    
    def __init__(self, pose_engine: PoseDetectionEngine):
        self.pose_engine = pose_engine
        self.fall_threshold = 0.6  # Ngưỡng phát hiện fall
        
    def detect_pose_from_frame(self, frame: np.ndarray, camera_id: str) -> Optional[PoseDetectionResult]:
        """
        Detect pose từ frame
        
        Args:
            frame: OpenCV frame
            camera_id: ID camera
            
        Returns:
            PoseDetectionResult hoặc None
        """
        if not self.pose_engine.is_ready():
            return None
            
        return self.pose_engine.detect_pose(frame, camera_id)
    
    def analyze_for_fall_detection(self, pose_result: PoseDetectionResult) -> Optional[FallDetectionEvent]:
        """
        Phân tích pose result để phát hiện fall với logic cải tiến - giảm false positive
        
        Args:
            pose_result: Kết quả pose detection
            
        Returns:
            FallDetectionEvent nếu phát hiện khả năng fall thực sự
        """
        if not pose_result:
            return None
        
        # Kiểm tra pose có đủ valid keypoints không (tối thiểu 6 keypoints)
        valid_keypoints = [kp for kp in pose_result.keypoints if kp.confidence > 0.4]
        if len(valid_keypoints) < 6:
            # Không đủ keypoints để phân tích fall - không báo fall
            return None
        
        # Kiểm tra có keypoints cốt lõi không (vai, hông)
        core_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        detected_core = [kp for kp in valid_keypoints if kp.name in core_keypoints]
        
        if len(detected_core) < 2:
            # Không có đủ core keypoints - không thể phân tích fall chính xác
            return None
        
        # Tính body angle với logic cải tiến
        body_angle = self._calculate_improved_body_angle(valid_keypoints)
        
        if body_angle is None:
            # Không thể tính body angle - không báo fall
            return None
        
        # Logic fall detection cải tiến - strict hơn
        fall_probability = self._calculate_fall_probability_strict(valid_keypoints, body_angle)
        
        # Chỉ tạo event nếu fall probability >= 0.8 (very strict)
        if fall_probability >= 0.8:
            alert_level = self._determine_alert_level(fall_probability)
            
            fall_event = FallDetectionEvent(
                camera_id=pose_result.camera_id,
                timestamp=pose_result.timestamp,
                pose_result=pose_result,
                fall_probability=fall_probability,
                body_angle=body_angle,
                alert_level=alert_level
            )
            
            return fall_event
            
        return None

    def _calculate_improved_body_angle(self, keypoints) -> Optional[float]:
        """Tính body angle với logic cải tiến"""
        # Tìm keypoints cần thiết
        shoulders = [kp for kp in keypoints if 'shoulder' in kp.name and kp.confidence > 0.5]
        hips = [kp for kp in keypoints if 'hip' in kp.name and kp.confidence > 0.5]
        
        if len(shoulders) < 1 or len(hips) < 1:
            return None
        
        # Lấy center points
        shoulder_center_y = sum(kp.y for kp in shoulders) / len(shoulders)
        hip_center_y = sum(kp.y for kp in hips) / len(hips)
        
        # Kiểm tra thứ tự hợp lý (vai phải ở trên hông trong tư thế đứng)
        if abs(shoulder_center_y - hip_center_y) < 30:  # Quá gần nhau - không hợp lý
            return None
        
        # Tính góc nghiêng cơ thể
        if len(shoulders) >= 2 and len(hips) >= 2:
            # Có đủ cả 2 vai và 2 hông
            shoulder_slope = (shoulders[1].y - shoulders[0].y) / max(abs(shoulders[1].x - shoulders[0].x), 1)
            hip_slope = (hips[1].y - hips[0].y) / max(abs(hips[1].x - hips[0].x), 1)
            avg_slope = (shoulder_slope + hip_slope) / 2
            body_angle = abs(np.degrees(np.arctan(avg_slope)))
        else:
            # Chỉ có 1 vai hoặc 1 hông - tính góc đơn giản
            body_height = abs(shoulder_center_y - hip_center_y)
            if body_height < 80:  # Quá thấp - có thể đã nằm
                body_angle = 75  # Nghiêng nhiều
            else:
                body_angle = 15  # Đứng gần thẳng
        
        return body_angle

    def _calculate_fall_probability_strict(self, keypoints, body_angle: float) -> float:
        """Tính fall probability với logic rất strict để tránh false positive"""
        
        # Base score từ body angle (rất strict)
        if body_angle > 80:
            angle_score = 0.8  # Gần như nằm ngang
        elif body_angle > 65:
            angle_score = 0.5  # Nghiêng rất nhiều  
        elif body_angle > 45:
            angle_score = 0.2  # Nghiêng vừa
        else:
            angle_score = 0.0  # Đứng thẳng hoặc nghiêng ít
        
        # Kiểm tra head position relative to body center
        head_keypoints = [kp for kp in keypoints if kp.name in ['nose'] and kp.confidence > 0.6]
        body_keypoints = [kp for kp in keypoints if kp.name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'] and kp.confidence > 0.5]
        
        position_score = 0.0
        if head_keypoints and len(body_keypoints) >= 2:
            head_y = head_keypoints[0].y
            body_center_y = sum(kp.y for kp in body_keypoints) / len(body_keypoints)
            
            # Head phải thấp hơn body center đáng kể mới tính là fall
            if head_y > body_center_y + 50:  # Head significantly lower
                position_score = 0.3
            elif head_y > body_center_y + 20:  # Head moderately lower  
                position_score = 0.1
        
        # Final score với weight conservative
        final_score = (
            angle_score * 0.7 +      # Body angle là main factor
            position_score * 0.3     # Head position secondary
        )
        
        # Additional penalties for uncertain cases
        if len(keypoints) < 10:  # Quá ít keypoints
            final_score *= 0.6  # Significant penalty
        
        # Confidence threshold - chỉ báo fall khi rất chắc chắn
        avg_confidence = sum(kp.confidence for kp in keypoints) / len(keypoints)
        if avg_confidence < 0.5:
            final_score *= 0.4  # Heavy penalty cho low confidence
        
        return min(final_score, 1.0)

    def _determine_alert_level(self, fall_probability: float) -> str:
        """Xác định mức độ cảnh báo"""
        if fall_probability >= 0.95:
            return "critical"
        elif fall_probability >= 0.85:
            return "danger"  
        else:
            return "warning"
    
    def process_frame_for_monitoring(
        self, 
        frame: np.ndarray, 
        camera_id: str
    ) -> tuple[Optional[PoseDetectionResult], Optional[FallDetectionEvent]]:
        """
        Xử lý frame cho monitoring - detect pose và check fall
        
        Args:
            frame: OpenCV frame
            camera_id: ID camera
            
        Returns:
            Tuple (pose_result, fall_event)
        """
        # Detect pose
        pose_result = self.detect_pose_from_frame(frame, camera_id)
        
        if not pose_result:
            return None, None
        
        # Analyze for fall
        fall_event = self.analyze_for_fall_detection(pose_result)
        
        return pose_result, fall_event
    
    def is_pose_valid(self, pose_result: PoseDetectionResult, min_keypoints: int = 4) -> bool:
        """
        Kiểm tra pose result có hợp lệ không
        
        Args:
            pose_result: Kết quả pose detection
            min_keypoints: Số keypoints tối thiểu
            
        Returns:
            True nếu pose hợp lệ
        """
        if not pose_result:
            return False
            
        valid_keypoints = pose_result.get_valid_keypoints()
        return len(valid_keypoints) >= min_keypoints
    
    def get_monitoring_stats(self, pose_result: PoseDetectionResult) -> dict:
        """
        Lấy thống kê cho monitoring
        
        Args:
            pose_result: Kết quả pose detection
            
        Returns:
            Dictionary chứa stats
        """
        if not pose_result:
            return {
                'total_keypoints': 0,
                'valid_keypoints': 0,
                'pose_confidence': 0.0,
                'inference_time_ms': 0.0,
                'body_angle': None,
                'fall_scores': {}
            }
        
        valid_kps = pose_result.get_valid_keypoints()
        body_angle = pose_result.calculate_body_angle()
        
        return {
            'total_keypoints': len(pose_result.keypoints),
            'valid_keypoints': len(valid_kps),
            'pose_confidence': pose_result.pose_confidence,
            'inference_time_ms': pose_result.inference_time * 1000,
            'body_angle': body_angle,
            'fall_scores': pose_result.fall_detection_scores
        }