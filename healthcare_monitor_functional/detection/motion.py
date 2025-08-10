#!/usr/bin/env python3
"""
Motion Analysis Functions
Function-based motion detection and analysis
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any


def calculate_motion_level(person_detections: List[Dict], person_positions: List[Tuple[float, float]], 
                          max_history: int = 10) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Calculate motion level based on person position changes
    
    Args:
        person_detections: List of current person detections
        person_positions: History of person positions
        max_history: Maximum position history to maintain
        
    Returns:
        Tuple of (motion_level, updated_person_positions)
    """
    if not person_detections:
        return 0.0, person_positions
        
    # Get primary person position
    primary_person = max(person_detections, 
                        key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
    bbox = primary_person['bbox']
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2
    current_position = (center_x, center_y)
    
    # Update position history
    updated_positions = person_positions.copy()
    updated_positions.append(current_position)
    
    # Maintain max history
    if len(updated_positions) > max_history:
        updated_positions = updated_positions[-max_history:]
    
    # Calculate motion level
    if len(updated_positions) < 3:
        return 0.5, updated_positions
        
    total_displacement = 0.0
    for i in range(1, len(updated_positions)):
        dx = updated_positions[i][0] - updated_positions[i-1][0]
        dy = updated_positions[i][1] - updated_positions[i-1][1]
        displacement = np.sqrt(dx*dx + dy*dy)
        total_displacement += displacement
    
    avg_displacement = total_displacement / (len(updated_positions) - 1)
    motion_level = min(avg_displacement / 50.0, 1.0)
    
    return motion_level, updated_positions


def update_motion_history(motion_levels: List[float], current_motion: float, 
                         max_history: int = 10) -> List[float]:
    """
    Update motion level history
    
    Args:
        motion_levels: Current motion history
        current_motion: New motion level
        max_history: Maximum history length
        
    Returns:
        Updated motion history
    """
    updated_history = motion_levels.copy()
    updated_history.append(current_motion)
    
    if len(updated_history) > max_history:
        updated_history = updated_history[-max_history:]
        
    return updated_history


def check_significant_motion(motion_level: float, threshold: float = 0.3) -> bool:
    """
    Check if motion level is significant
    
    Args:
        motion_level: Current motion level
        threshold: Threshold for significant motion
        
    Returns:
        True if motion is significant
    """
    return motion_level > threshold


def get_motion_statistics(motion_levels: List[float]) -> Dict[str, float]:
    """
    Calculate motion statistics
    
    Args:
        motion_levels: List of motion levels
        
    Returns:
        Motion statistics dictionary
    """
    if not motion_levels:
        return {
            'current_motion': 0.0,
            'avg_motion': 0.0,
            'max_motion': 0.0,
            'motion_samples': 0
        }
    
    return {
        'current_motion': motion_levels[-1],
        'avg_motion': sum(motion_levels) / len(motion_levels),
        'max_motion': max(motion_levels),
        'motion_samples': len(motion_levels)
    }


def analyze_motion_patterns(motion_levels: List[float], window_size: int = 5) -> Dict[str, Any]:
    """
    Analyze motion patterns over time
    
    Args:
        motion_levels: List of motion levels
        window_size: Window size for pattern analysis
        
    Returns:
        Motion pattern analysis results
    """
    if len(motion_levels) < window_size:
        return {
            'trend': 'insufficient_data',
            'volatility': 0.0,
            'stability': 0.0,
            'pattern_type': 'unknown'
        }
    
    recent_window = motion_levels[-window_size:]
    
    # Calculate trend
    trend = 'stable'
    if len(recent_window) >= 3:
        start_avg = sum(recent_window[:2]) / 2
        end_avg = sum(recent_window[-2:]) / 2
        
        if end_avg > start_avg + 0.1:
            trend = 'increasing'
        elif end_avg < start_avg - 0.1:
            trend = 'decreasing'
    
    # Calculate volatility (standard deviation)
    avg_motion = sum(recent_window) / len(recent_window)
    volatility = np.std(recent_window) if len(recent_window) > 1 else 0.0
    
    # Calculate stability (inverse of volatility)
    stability = max(0.0, 1.0 - float(volatility))
    
    # Determine pattern type
    pattern_type = 'calm'
    if avg_motion > 0.7:
        pattern_type = 'active'
    elif avg_motion > 0.4:
        pattern_type = 'moderate'
    elif volatility > 0.3:
        pattern_type = 'irregular'
    
    return {
        'trend': trend,
        'volatility': float(volatility),
        'stability': stability,
        'pattern_type': pattern_type,
        'avg_motion_window': avg_motion
    }


def is_motion_concerning(motion_analysis: Dict[str, Any], detection_context: str = 'general') -> bool:
    """
    Determine if motion patterns are concerning for healthcare monitoring
    
    Args:
        motion_analysis: Motion pattern analysis results
        detection_context: Context for evaluation ('fall', 'seizure', 'general')
        
    Returns:
        True if motion is concerning
    """
    avg_motion = motion_analysis.get('avg_motion_window', 0.0)
    volatility = motion_analysis.get('volatility', 0.0)
    pattern_type = motion_analysis.get('pattern_type', 'unknown')
    
    if detection_context == 'fall':
        # High motion with high volatility could indicate fall
        return avg_motion > 0.6 and volatility > 0.2
        
    elif detection_context == 'seizure':
        # Very high motion or irregular patterns could indicate seizure
        return (avg_motion > 0.8) or (pattern_type == 'irregular' and volatility > 0.4)
        
    else:  # general
        # Any extremely high motion or very irregular patterns
        return avg_motion > 0.9 or volatility > 0.5


def get_motion_color(motion_level: float) -> Tuple[int, int, int]:
    """
    Get color coding for motion level visualization
    
    Args:
        motion_level: Motion level (0-1)
        
    Returns:
        BGR color tuple for OpenCV
    """
    if motion_level < 0.3:
        return (0, 255, 0)  # Green - calm
    elif motion_level < 0.7:
        return (0, 255, 255)  # Yellow - moderate  
    else:
        return (0, 0, 255)  # Red - high motion
