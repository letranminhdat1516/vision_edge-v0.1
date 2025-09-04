"""
Seizure Detection Module using VSViG
Real-time seizure detection from video surveillance for healthcare monitoring
"""

from .vsvig_detector import VSViGSeizureDetector
from .pose_estimator import UltimatePoseEstimator  
from .yolov8_pose_estimator import YOLOv8PoseEstimator

__all__ = [
    'VSViGSeizureDetector',
    'UltimatePoseEstimator',
    'YOLOv8PoseEstimator'
]

__version__ = '1.0.0'
__author__ = 'Vision Edge Healthcare Team'
