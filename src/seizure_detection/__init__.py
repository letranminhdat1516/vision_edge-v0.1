"""
Seizure Detection Module using VSViG
Real-time seizure detection from video surveillance for healthcare monitoring
"""

from .vsvig_detector import VSViGSeizureDetector
from .pose_estimator import CustomPoseEstimator  
from .seizure_predictor import SeizurePredictor

__all__ = [
    'VSViGSeizureDetector',
    'CustomPoseEstimator', 
    'SeizurePredictor'
]

__version__ = '1.0.0'
__author__ = 'Vision Edge Healthcare Team'
