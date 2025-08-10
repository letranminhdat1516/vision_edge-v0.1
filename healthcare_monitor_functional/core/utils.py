#!/usr/bin/env python3
"""
Core Utilities
Common utility functions for healthcare monitoring
"""

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def calculate_runtime_stats(start_time: float) -> Dict[str, float]:
    """
    Calculate runtime statistics
    
    Args:
        start_time: Start time timestamp
        
    Returns:
        Runtime statistics dictionary
    """
    runtime = time.time() - start_time
    return {
        'runtime': runtime,
        'runtime_minutes': runtime / 60.0,
        'runtime_hours': runtime / 3600.0
    }


def calculate_fps(total_frames: int, runtime: float) -> float:
    """
    Calculate frames per second
    
    Args:
        total_frames: Total number of frames processed
        runtime: Runtime in seconds
        
    Returns:
        FPS value
    """
    if runtime <= 0:
        return 0.0
    return total_frames / runtime


def calculate_processing_efficiency(frames_processed: int, total_frames: int) -> float:
    """
    Calculate processing efficiency percentage
    
    Args:
        frames_processed: Number of frames actually processed
        total_frames: Total number of frames received
        
    Returns:
        Efficiency percentage (frames skipped)
    """
    if total_frames <= 0:
        return 0.0
    return (1 - frames_processed / total_frames) * 100


def smooth_confidence_values(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Apply temporal smoothing to confidence values
    
    Args:
        values: List of confidence values
        weights: Optional weights for values
        
    Returns:
        Smoothed confidence value
    """
    if not values:
        return 0.0
        
    if len(values) <= 1:
        return values[0] if values else 0.0
    
    if weights is None:
        weights = np.linspace(0.5, 1.0, len(values)).tolist()
        
    return float(np.average(values, weights=weights))


def enhance_confidence_with_motion(base_confidence: float, motion_level: float, 
                                 detection_type: str) -> float:
    """
    Enhance detection confidence based on motion patterns
    
    Args:
        base_confidence: Base confidence value
        motion_level: Current motion level (0-1)
        detection_type: Type of detection ('fall' or 'seizure')
        
    Returns:
        Enhanced confidence value
    """
    motion_boost = 0.0
    
    if detection_type == 'fall':
        if motion_level > 0.6:
            motion_boost = 0.3
        elif motion_level > 0.3:
            motion_boost = 0.2
        elif motion_level > 0.1:
            motion_boost = 0.05
        else:
            motion_boost = -0.05
    else:  # seizure
        if motion_level > 0.8:
            motion_boost = 0.1
        elif motion_level < 0.1:
            motion_boost = -0.02
        else:
            motion_boost = 0.02
    
    enhanced_confidence = base_confidence + motion_boost
    return max(0.0, min(1.0, enhanced_confidence))


def calculate_motion_level(person_positions: List[Tuple[float, float]], 
                          max_history: int = 10) -> float:
    """
    Calculate motion level based on position history
    
    Args:
        person_positions: List of (x, y) position tuples
        max_history: Maximum history length to consider
        
    Returns:
        Motion level (0-1)
    """
    if len(person_positions) < 3:
        return 0.5
        
    # Limit to max history
    positions = person_positions[-max_history:] if len(person_positions) > max_history else person_positions
    
    total_displacement = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        displacement = np.sqrt(dx*dx + dy*dy)
        total_displacement += displacement
    
    avg_displacement = total_displacement / (len(positions) - 1)
    motion_level = min(avg_displacement / 50.0, 1.0)
    
    return motion_level


def get_primary_person(person_detections: List[Dict]) -> Optional[Dict]:
    """
    Get the primary person from detections (largest bounding box)
    
    Args:
        person_detections: List of person detection dictionaries
        
    Returns:
        Primary person detection or None
    """
    if not person_detections:
        return None
        
    return max(person_detections, 
              key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])


def bbox_to_coordinates(bbox: List[float]) -> List[int]:
    """
    Convert bbox [x, y, w, h] to [x1, y1, x2, y2] coordinates
    
    Args:
        bbox: Bounding box as [x, y, width, height]
        
    Returns:
        Coordinates as [x1, y1, x2, y2]
    """
    return [
        int(bbox[0]),
        int(bbox[1]), 
        int(bbox[0] + bbox[2]),
        int(bbox[1] + bbox[3])
    ]


def save_frame_metadata(filepath: Path, metadata: Dict[str, Any]) -> bool:
    """
    Save frame metadata as JSON
    
    Args:
        filepath: Path to the image file
        metadata: Metadata dictionary
        
    Returns:
        Success status
    """
    try:
        metadata_file = filepath.with_suffix('.jpg_metadata.json')
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
            
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception:
        return False


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: Unix timestamp (optional, uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def update_moving_average(current_avg: float, new_value: float, count: int) -> float:
    """
    Update moving average with new value
    
    Args:
        current_avg: Current average value
        new_value: New value to include
        count: Number of values before this update
        
    Returns:
        Updated average
    """
    if count == 0:
        return new_value
    else:
        return (current_avg * count + new_value) / (count + 1)


def cleanup_old_files(directory: Path, max_files: int = 1000) -> int:
    """
    Clean up old files if directory has too many files
    
    Args:
        directory: Directory to clean up
        max_files: Maximum number of files to keep
        
    Returns:
        Number of files deleted
    """
    try:
        if not directory.exists():
            return 0
            
        files = sorted(directory.glob("*"), key=lambda x: x.stat().st_mtime)
        
        if len(files) <= max_files:
            return 0
            
        files_to_delete = files[:-max_files]
        deleted_count = 0
        
        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
            except Exception:
                pass
                
        return deleted_count
    except Exception:
        return 0
