"""
Domain Entity cho Pose Detection
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class Keypoint:
    """Value object đại diện cho 1 keypoint"""
    index: int
    name: str
    x: float
    y: float
    confidence: float
    
    def is_valid(self, threshold: float = 0.2) -> bool:
        """Kiểm tra keypoint có confidence đủ cao không"""
        return self.confidence >= threshold


@dataclass
class PoseDetectionResult:
    """Entity đại diện cho kết quả pose detection"""
    camera_id: str
    timestamp: datetime
    keypoints: List[Keypoint]
    frame_metadata: Dict
    inference_time: float
    
    # Fall detection specific scores
    fall_detection_scores: Dict[str, float] = field(default_factory=dict)
    pose_confidence: float = 0.0
    
    def get_valid_keypoints(self, threshold: float = 0.2) -> List[Keypoint]:
        """Lấy các keypoints có confidence đủ cao"""
        return [kp for kp in self.keypoints if kp.is_valid(threshold)]
    
    def get_keypoint_by_name(self, name: str) -> Optional[Keypoint]:
        """Lấy keypoint theo tên"""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
    
    def get_body_keypoints(self) -> Dict[str, Optional[Keypoint]]:
        """Lấy các keypoints quan trọng cho fall detection"""
        return {
            'left_shoulder': self.get_keypoint_by_name('left shoulder'),
            'right_shoulder': self.get_keypoint_by_name('right shoulder'),
            'left_hip': self.get_keypoint_by_name('left hip'),
            'right_hip': self.get_keypoint_by_name('right hip')
        }
    
    def calculate_body_angle(self) -> Optional[float]:
        """Tính góc nghiêng của cơ thể"""
        body_kps = self.get_body_keypoints()
        left_shoulder = body_kps['left_shoulder']
        left_hip = body_kps['left_hip']
        
        if left_shoulder and left_hip and left_shoulder.is_valid() and left_hip.is_valid():
            dx = left_hip.x - left_shoulder.x
            dy = left_hip.y - left_shoulder.y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            return angle
        return None


@dataclass
class FallDetectionEvent:
    """Domain Event cho fall detection"""
    camera_id: str
    timestamp: datetime
    pose_result: PoseDetectionResult
    fall_probability: float
    body_angle: Optional[float]
    alert_level: str  # 'normal', 'warning', 'danger'
    
    @classmethod
    def from_pose_result(cls, pose_result: PoseDetectionResult) -> 'FallDetectionEvent':
        """Tạo fall detection event từ pose result"""
        body_angle = pose_result.calculate_body_angle()
        
        # Logic đơn giản để xác định fall probability
        fall_probability = 0.0
        alert_level = 'normal'
        
        if body_angle is not None:
            # Nếu góc > 45 độ có thể là fall
            if abs(body_angle) > 45:
                fall_probability = min(abs(body_angle) / 90, 1.0)
                alert_level = 'warning' if fall_probability < 0.7 else 'danger'
        
        return cls(
            camera_id=pose_result.camera_id,
            timestamp=pose_result.timestamp,
            pose_result=pose_result,
            fall_probability=fall_probability,
            body_angle=body_angle,
            alert_level=alert_level
        )