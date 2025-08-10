#!/usr/bin/env python3
"""
Fall Detection Functions
Function-based fall detection system
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


def initialize_fall_detector(confidence_threshold: float = 0.4) -> Tuple[Any, Any]:
    """
    Initialize fall detection system
    
    Args:
        confidence_threshold: Confidence threshold for fall detection
        
    Returns:
        Tuple of (detector, predictor) - for fallback, predictor is None
    """
    try:
        from fall_detection.simple_fall_detector_v2 import SimpleFallDetector
        detector = SimpleFallDetector(confidence_threshold=confidence_threshold)
        return detector, None
    except ImportError:
        try:
            from fall_detection.simple_fall_detector import SimpleFallDetector
            detector = SimpleFallDetector(confidence_threshold=confidence_threshold)
            return detector, None
        except ImportError:
            # Fallback implementation
            detector = FallbackFallDetector(confidence_threshold)
            return detector, None


class FallbackFallDetector:
    """Smart fallback fall detector with motion analysis"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.motion_history = []
        self.aspect_ratio_history = []
    
    def detect_fall(self, frame: np.ndarray, person: Dict) -> Dict[str, Any]:
        """Smart fall detection based on aspect ratio and motion patterns"""
        if not person or 'bbox' not in person:
            return {'fall_detected': False, 'confidence': 0.0}
        
        bbox = person['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate aspect ratio (width/height)
        width = x2 - x1
        height = y2 - y1
        
        if height == 0:
            return {'fall_detected': False, 'confidence': 0.0}
            
        aspect_ratio = width / height
        
        # Update aspect ratio history
        self.aspect_ratio_history.append(aspect_ratio)
        if len(self.aspect_ratio_history) > 10:
            self.aspect_ratio_history.pop(0)
        
        # Fall detection logic
        confidence = 0.0
        
        # Check if aspect ratio suggests horizontal position (fall)
        if aspect_ratio > 1.2:  # More wide than tall
            confidence += 0.4
        
        # Check for sudden change in aspect ratio
        if len(self.aspect_ratio_history) >= 5:
            recent_avg = sum(self.aspect_ratio_history[-3:]) / 3
            older_avg = sum(self.aspect_ratio_history[-5:-2]) / 3
            
            if recent_avg > older_avg * 1.3:  # 30% increase in width/height ratio
                confidence += 0.3
        
        # Check if person is in lower part of frame (on ground)
        frame_height = frame.shape[0] if frame is not None else 480
        person_center_y = (y1 + y2) / 2
        
        if person_center_y > frame_height * 0.6:  # In lower 40% of frame
            confidence += 0.2
        
        # Check if person bbox is very wide (lying down)
        if width > height * 1.5:
            confidence += 0.3
            
        fall_detected = confidence >= self.confidence_threshold
        
        return {
            'fall_detected': fall_detected, 
            'confidence': min(confidence, 1.0),
            'aspect_ratio': aspect_ratio,
            'person_center_y': person_center_y / frame_height if frame is not None else 0
        }


def detect_fall(detector: Any, frame: np.ndarray, person_detection: Dict, 
               motion_level: float = 0.0) -> Dict[str, Any]:
    """
    Perform fall detection on a frame
    
    Args:
        detector: Fall detector instance
        frame: Input frame
        person_detection: Person detection result
        motion_level: Current motion level for enhancement
        
    Returns:
        Fall detection result dictionary
    """
    if detector is None:
        return {
            'fall_detected': False,
            'confidence': 0.0,
            'enhanced_confidence': 0.0,
            'processing_time': 0.0
        }
    
    start_time = time.time()
    
    try:
        # Basic fall detection
        fall_result = detector.detect_fall(frame, person_detection)
        base_confidence = fall_result.get('confidence', 0.0)
        
        # Enhance confidence with motion
        enhanced_confidence = enhance_fall_confidence_with_motion(base_confidence, motion_level)
        
        processing_time = time.time() - start_time
        
        return {
            'fall_detected': fall_result.get('fall_detected', False),
            'confidence': base_confidence,
            'enhanced_confidence': enhanced_confidence,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logging.error(f"Fall detection error: {str(e)}")
        return {
            'fall_detected': False,
            'confidence': 0.0,
            'enhanced_confidence': 0.0,
            'processing_time': time.time() - start_time,
            'error': str(e)
        }


def enhance_fall_confidence_with_motion(base_confidence: float, motion_level: float) -> float:
    """
    Enhance fall confidence using motion information
    
    Args:
        base_confidence: Base confidence from fall detector
        motion_level: Current motion level (0-1)
        
    Returns:
        Enhanced confidence value
    """
    motion_boost = 0.0
    
    if motion_level > 0.6:
        motion_boost = 0.3  # High motion significantly boosts fall confidence
    elif motion_level > 0.3:
        motion_boost = 0.2  # Moderate motion moderately boosts confidence
    elif motion_level > 0.1:
        motion_boost = 0.05  # Low motion slightly boosts confidence
    else:
        motion_boost = -0.05  # Very low motion slightly reduces confidence
    
    enhanced_confidence = base_confidence + motion_boost
    return max(0.0, min(1.0, enhanced_confidence))


def smooth_fall_confidences(confidence_history: List[float], current_confidence: float, 
                           max_history: int = 10) -> Tuple[float, List[float]]:
    """
    Apply temporal smoothing to fall confidence values
    
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


def check_fall_confirmation(smoothed_confidence: float, confirmation_frames: int,
                          threshold: float = 0.3, min_frames: int = 1) -> Tuple[bool, int]:
    """
    Check fall confirmation based on confidence and frame count
    
    Args:
        smoothed_confidence: Smoothed confidence value
        confirmation_frames: Current confirmation frame count
        threshold: Confidence threshold for fall detection
        min_frames: Minimum frames needed for confirmation
        
    Returns:
        Tuple of (fall_confirmed, updated_confirmation_frames)
    """
    if smoothed_confidence > threshold:
        updated_frames = confirmation_frames + 1
    else:
        updated_frames = max(0, confirmation_frames - 1)
    
    fall_confirmed = updated_frames >= min_frames
    
    return fall_confirmed, updated_frames


def process_fall_detection(detector: Any, frame: np.ndarray, person_detections: List[Dict],
                          confidence_history: List[float], confirmation_frames: int,
                          motion_level: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete fall detection processing pipeline
    
    Args:
        detector: Fall detector instance
        frame: Input frame
        person_detections: List of person detections
        confidence_history: History of confidence values
        confirmation_frames: Current confirmation frame count
        motion_level: Current motion level
        config: Detection configuration
        
    Returns:
        Complete fall detection result
    """
    result = {
        'fall_detected': False,
        'confidence': 0.0,
        'smoothed_confidence': 0.0,
        'confirmation_frames': confirmation_frames,
        'confidence_history': confidence_history,
        'processing_time': 0.0
    }
    
    if not person_detections:
        # Reset confirmation frames when no person detected
        result['confirmation_frames'] = 0
        return result
    
    # Get primary person for detection
    primary_person = max(person_detections, 
                        key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
    
    # Perform fall detection
    detection_result = detect_fall(detector, frame, primary_person, motion_level)
    
    # Smooth confidence values
    enhanced_confidence = detection_result['enhanced_confidence']
    smoothed_confidence, updated_history = smooth_fall_confidences(
        confidence_history, enhanced_confidence, config.get('max_history', 10)
    )
    
    # Check confirmation
    fall_confirmed, updated_frames = check_fall_confirmation(
        smoothed_confidence, 
        confirmation_frames,
        config.get('threshold', 0.3),
        config.get('confirmation_frames', 1)
    )
    
    # Update result
    result.update({
        'fall_detected': fall_confirmed,
        'confidence': detection_result['confidence'],
        'smoothed_confidence': smoothed_confidence,
        'confirmation_frames': updated_frames,
        'confidence_history': updated_history,
        'processing_time': detection_result['processing_time']
    })
    
    return result


def get_fall_statistics(fall_detections: int, confidence_avg: float, 
                       last_fall_time: Optional[float]) -> Dict[str, Any]:
    """
    Get fall detection statistics
    
    Args:
        fall_detections: Total number of fall detections
        confidence_avg: Average confidence of detections
        last_fall_time: Timestamp of last fall detection
        
    Returns:
        Fall statistics dictionary
    """
    last_fall_str = "None"
    if last_fall_time is not None:
        from datetime import datetime
        last_fall_str = datetime.fromtimestamp(last_fall_time).strftime('%H:%M:%S')
    
    return {
        'falls_detected': fall_detections,
        'average_confidence': confidence_avg,
        'last_fall_time': last_fall_str,
        'last_fall_timestamp': last_fall_time
    }


def update_fall_statistics(current_stats: Dict[str, Any], detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update fall detection statistics
    
    Args:
        current_stats: Current statistics
        detection_result: Latest detection result
        
    Returns:
        Updated statistics
    """
    updated_stats = current_stats.copy()
    
    if detection_result.get('fall_detected', False):
        # Update detection count
        updated_stats['fall_detections'] = current_stats.get('fall_detections', 0) + 1
        updated_stats['last_fall_time'] = time.time()
        
        # Update average confidence
        current_avg = current_stats.get('fall_confidence_avg', 0.0)
        current_count = current_stats.get('fall_detections', 0)
        new_confidence = detection_result.get('smoothed_confidence', 0.0)
        
        if current_count == 0:
            updated_stats['fall_confidence_avg'] = new_confidence
        else:
            updated_stats['fall_confidence_avg'] = (
                (current_avg * (current_count - 1) + new_confidence) / current_count
            )
    
    return updated_stats
