#!/usr/bin/env python3
"""
Seizure Detection Functions  
Function-based seizure detection system
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def initialize_seizure_detector(confidence_threshold: float = 0.7) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize seizure detection system
    
    Args:
        confidence_threshold: Confidence threshold for seizure detection
        
    Returns:
        Tuple of (detector, predictor) instances or fallbacks
    """
    try:
        from seizure_detection.vsvig_detector import VSViGSeizureDetector
        from seizure_detection.seizure_predictor import SeizurePredictor
        
        detector = VSViGSeizureDetector(confidence_threshold=confidence_threshold)
        predictor = SeizurePredictor(
            temporal_window=25,
            alert_threshold=0.7,
            warning_threshold=0.5
        )
        
        return detector, predictor
        
    except ImportError:
        # Return fallback implementations
        return FallbackSeizureDetector(confidence_threshold), FallbackSeizurePredictor()


class FallbackSeizureDetector:
    """Fallback seizure detector when main detector is not available"""
    
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
    
    def detect_seizure(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """Fallback seizure detection - always returns no seizure detected"""
        return {
            'temporal_ready': False,
            'keypoints': None,
            'confidence': 0.0
        }


class FallbackSeizurePredictor:
    """Fallback seizure predictor when main predictor is not available"""
    
    def __init__(self, temporal_window: int = 30, alert_threshold: float = 0.65, warning_threshold: float = 0.45):
        self.temporal_window = temporal_window
        self.alert_threshold = alert_threshold
        self.warning_threshold = warning_threshold
    
    def update_prediction(self, confidence: float) -> Dict[str, Any]:
        """Fallback prediction update - always returns normal state"""
        return {
            'smoothed_confidence': 0.0,
            'seizure_detected': False,
            'alert_level': 'normal',
            'ready': False
        }


def detect_seizure(detector: Any, predictor: Any, frame: np.ndarray, person_bbox: List[int],
                  motion_level: float = 0.0) -> Dict[str, Any]:
    """
    Perform seizure detection on a frame
    
    Args:
        detector: Seizure detector instance
        predictor: Seizure predictor instance
        frame: Input frame
        person_bbox: Person bounding box coordinates
        motion_level: Current motion level for enhancement
        
    Returns:
        Seizure detection result dictionary
    """
    if detector is None:
        return {
            'seizure_detected': False,
            'confidence': 0.0,
            'seizure_ready': False,
            'keypoints': None,
            'processing_time': 0.0
        }
    
    start_time = time.time()
    
    try:
        # Basic seizure detection
        seizure_result = detector.detect_seizure(frame, person_bbox)
        
        result = {
            'seizure_detected': False,
            'confidence': 0.0,
            'seizure_ready': seizure_result['temporal_ready'],
            'keypoints': seizure_result['keypoints'],
            'processing_time': 0.0
        }
        
        # If temporal buffer is ready, process prediction
        if seizure_result['temporal_ready'] and predictor is not None:
            pred_result = predictor.update_prediction(seizure_result['confidence'])
            
            base_confidence = pred_result['smoothed_confidence']
            enhanced_confidence = enhance_seizure_confidence_with_motion(base_confidence, motion_level)
            
            result.update({
                'confidence': enhanced_confidence,
                'base_confidence': base_confidence
            })
        
        result['processing_time'] = time.time() - start_time
        return result
        
    except Exception as e:
        logging.error(f"Seizure detection error: {str(e)}")
        return {
            'seizure_detected': False,
            'confidence': 0.0,
            'seizure_ready': False,
            'keypoints': None,
            'processing_time': time.time() - start_time,
            'error': str(e)
        }


def enhance_seizure_confidence_with_motion(base_confidence: float, motion_level: float) -> float:
    """
    Enhance seizure confidence using motion information
    
    Args:
        base_confidence: Base confidence from seizure detector
        motion_level: Current motion level (0-1)
        
    Returns:
        Enhanced confidence value
    """
    motion_boost = 0.0
    
    if motion_level > 0.8:
        motion_boost = 0.1  # Very high motion boosts seizure confidence
    elif motion_level < 0.1:
        motion_boost = -0.02  # Very low motion slightly reduces confidence
    else:
        motion_boost = 0.02  # Normal motion slightly boosts confidence
    
    enhanced_confidence = base_confidence + motion_boost
    return max(0.0, min(1.0, enhanced_confidence))


def smooth_seizure_confidences(confidence_history: List[float], current_confidence: float,
                              max_history: int = 10) -> Tuple[float, List[float]]:
    """
    Apply temporal smoothing to seizure confidence values
    
    Args:
        confidence_history: History of confidence values
        current_confidence: New confidence value
        max_history: Maximum history length
        
    Returns:
        Tuple of (smoothed_confidence, updated_history)
    """
    # Update history
    updated_history = confidence_history.copy()
    updated_history.append(current_confidence)
    
    if len(updated_history) > max_history:
        updated_history = updated_history[-max_history:]
    
    # Apply smoothing
    if len(updated_history) <= 1:
        return current_confidence, updated_history
    
    # Weighted average with more recent values having higher weight
    weights = np.linspace(0.5, 1.0, len(updated_history)).tolist()
    smoothed = float(np.average(updated_history, weights=weights))
    
    return smoothed, updated_history


def check_seizure_confirmation(smoothed_confidence: float, confirmation_frames: int,
                              threshold: float = 0.6, min_frames: int = 5,
                              warning_threshold: float = 0.4, motion_level: float = 0.0) -> Tuple[str, int]:
    """
    Check seizure confirmation and determine alert level
    
    Args:
        smoothed_confidence: Smoothed confidence value
        confirmation_frames: Current confirmation frame count
        threshold: Confidence threshold for seizure detection
        min_frames: Minimum frames needed for confirmation
        warning_threshold: Threshold for warning alerts
        motion_level: Current motion level
        
    Returns:
        Tuple of (alert_level, updated_confirmation_frames)
    """
    if smoothed_confidence > threshold:
        updated_frames = confirmation_frames + 1
    else:
        updated_frames = max(0, confirmation_frames - 1)
    
    # Determine alert level
    if updated_frames >= min_frames:
        alert_level = 'critical'  # Seizure detected
    elif smoothed_confidence > warning_threshold and motion_level > 0.7:
        alert_level = 'warning'  # Seizure warning
    else:
        alert_level = 'normal'
    
    return alert_level, updated_frames


def process_seizure_detection(detector: Any, predictor: Any, frame: np.ndarray,
                             person_bbox: List[int], confidence_history: List[float],
                             confirmation_frames: int, motion_level: float,
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete seizure detection processing pipeline
    
    Args:
        detector: Seizure detector instance
        predictor: Seizure predictor instance
        frame: Input frame
        person_bbox: Person bounding box coordinates
        confidence_history: History of confidence values
        confirmation_frames: Current confirmation frame count
        motion_level: Current motion level
        config: Detection configuration
        
    Returns:
        Complete seizure detection result
    """
    result = {
        'seizure_detected': False,
        'confidence': 0.0,
        'smoothed_confidence': 0.0,
        'confirmation_frames': confirmation_frames,
        'confidence_history': confidence_history,
        'seizure_ready': False,
        'keypoints': None,
        'alert_level': 'normal',
        'processing_time': 0.0
    }
    
    if detector is None:
        return result
    
    # Perform seizure detection
    detection_result = detect_seizure(detector, predictor, frame, person_bbox, motion_level)
    
    result.update({
        'seizure_ready': detection_result['seizure_ready'],
        'keypoints': detection_result['keypoints'],
        'processing_time': detection_result['processing_time']
    })
    
    # Process only if temporal buffer is ready
    if detection_result['seizure_ready']:
        # Smooth confidence values
        enhanced_confidence = detection_result['confidence']
        smoothed_confidence, updated_history = smooth_seizure_confidences(
            confidence_history, enhanced_confidence, config.get('max_history', 10)
        )
        
        # Check confirmation and alert level
        alert_level, updated_frames = check_seizure_confirmation(
            smoothed_confidence,
            confirmation_frames,
            config.get('threshold', 0.6),
            config.get('confirmation_frames', 5),
            config.get('warning_threshold', 0.4),
            motion_level
        )
        
        # Update result
        result.update({
            'seizure_detected': alert_level == 'critical',
            'confidence': enhanced_confidence,
            'smoothed_confidence': smoothed_confidence,
            'confirmation_frames': updated_frames,
            'confidence_history': updated_history,
            'alert_level': alert_level
        })
    
    return result


def get_seizure_statistics(seizure_detections: int, seizure_warnings: int, 
                          confidence_avg: float, pose_failures: int,
                          last_seizure_time: Optional[float]) -> Dict[str, Any]:
    """
    Get seizure detection statistics
    
    Args:
        seizure_detections: Total number of seizure detections
        seizure_warnings: Total number of seizure warnings
        confidence_avg: Average confidence of detections
        pose_failures: Number of pose extraction failures
        last_seizure_time: Timestamp of last seizure detection
        
    Returns:
        Seizure statistics dictionary
    """
    last_seizure_str = "None"
    if last_seizure_time is not None:
        from datetime import datetime
        last_seizure_str = datetime.fromtimestamp(last_seizure_time).strftime('%H:%M:%S')
    
    return {
        'seizures_detected': seizure_detections,
        'seizure_warnings': seizure_warnings,
        'average_confidence': confidence_avg,
        'pose_extraction_failures': pose_failures,
        'last_seizure_time': last_seizure_str,
        'last_seizure_timestamp': last_seizure_time
    }


def update_seizure_statistics(current_stats: Dict[str, Any], detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update seizure detection statistics
    
    Args:
        current_stats: Current statistics
        detection_result: Latest detection result
        
    Returns:
        Updated statistics
    """
    updated_stats = current_stats.copy()
    
    if detection_result.get('seizure_detected', False):
        # Update detection count
        updated_stats['seizure_detections'] = current_stats.get('seizure_detections', 0) + 1
        updated_stats['last_seizure_time'] = time.time()
        
        # Update average confidence
        current_avg = current_stats.get('seizure_confidence_avg', 0.0)
        current_count = current_stats.get('seizure_detections', 0)
        new_confidence = detection_result.get('smoothed_confidence', 0.0)
        
        if current_count == 0:
            updated_stats['seizure_confidence_avg'] = new_confidence
        else:
            updated_stats['seizure_confidence_avg'] = (
                (current_avg * (current_count - 1) + new_confidence) / current_count
            )
    
    elif detection_result.get('alert_level') == 'warning':
        # Update warning count
        updated_stats['seizure_warnings'] = current_stats.get('seizure_warnings', 0) + 1
    
    # Track pose extraction failures
    if 'error' in detection_result:
        updated_stats['pose_extraction_failures'] = current_stats.get('pose_extraction_failures', 0) + 1
    
    return updated_stats
