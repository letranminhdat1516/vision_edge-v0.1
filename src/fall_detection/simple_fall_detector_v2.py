"""
Simplified Fall Detection for Healthcare Monitoring System
Direct integration with existing pipeline
"""
import os
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np

log = logging.getLogger(__name__)

class SimpleFallDetector:
    """
    Simplified fall detection component for healthcare monitoring.
    Uses lightweight approach for integration with existing system.
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize simplified fall detector.
        
        Args:
            confidence_threshold: Minimum confidence for fall detection
        """
        self.confidence_threshold = confidence_threshold
        self.previous_frame = None
        self.previous_timestamp = None
        self.min_time_interval = 1.5  # Minimum 1.5 seconds between fall checks
        self.frame_buffer = []
        self.max_buffer_size = 3
        
        log.info(f"🩺 Simplified fall detector initialized (confidence: {confidence_threshold})")
    
    def detect_fall(self, current_frame, timestamp=None, person_bbox=None):
        """
        Detect fall in current frame using simplified approach.
        
        Args:
            current_frame: Current video frame (numpy array)
            timestamp: Frame timestamp (optional)
            person_bbox: Person bounding box from YOLO (optional)
            
        Returns:
            dict: Fall detection result
        """
        start_time = time.time()
        
        # Default result
        result = {
            'fall_detected': False,
            'confidence': 0.0,
            'angle': 0.0,
            'category': 'no-fall',
            'processing_time': 0.0,
            'method': 'simplified'
        }
        
        try:
            # Ensure timestamp is a valid number
            if timestamp is not None:
                if hasattr(timestamp, '__iter__') and not isinstance(timestamp, (str, bytes)):
                    # If timestamp is iterable (like list), take first element
                    try:
                        current_time = float(list(timestamp)[0])
                    except (ValueError, IndexError, TypeError):
                        current_time = time.time()
                else:
                    try:
                        current_time = float(timestamp)
                    except (ValueError, TypeError):
                        current_time = time.time()
            else:
                current_time = time.time()
            
            # Add frame to buffer with safe bbox conversion
            safe_bbox = self._safe_bbox_conversion(person_bbox)
            
            frame_data = {
                'frame': current_frame,
                'timestamp': current_time,
                'bbox': safe_bbox
            }
            
            self.frame_buffer.append(frame_data)
            
            # Keep buffer size manageable
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Check if we have enough frames and time interval
            if (len(self.frame_buffer) >= 2):
                try:
                    # Safely get timestamps
                    first_timestamp = self.frame_buffer[0]['timestamp']
                    if hasattr(first_timestamp, '__iter__') and not isinstance(first_timestamp, (str, bytes)):
                        first_timestamp = float(list(first_timestamp)[0])
                    else:
                        first_timestamp = float(first_timestamp)
                    
                    # Check time interval
                    if (current_time - first_timestamp) >= self.min_time_interval:
                        # Process simplified fall detection
                        fall_result = self._analyze_movement_pattern()
                        
                        if fall_result:
                            result.update(fall_result)
                            log.warning(f"🚨 Fall detected! Confidence: {result['confidence']:.2f}")
                except (ValueError, TypeError, KeyError) as time_error:
                    log.debug(f"Timestamp processing error: {time_error}")
                    # Continue without fall detection for this frame
                
        except Exception as e:
            log.error(f"Fall detection error in detect_fall: {e}")
            print(f"Fall detection error in detect_fall: {e}")
            print(f"Debug: person_bbox type={type(person_bbox)}, value={person_bbox}")
            
            # Stack trace for better debugging
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _safe_bbox_conversion(self, bbox):
        """
        Safely convert bbox to standard format [x1, y1, x2, y2].
        
        Args:
            bbox: Input bbox in various formats
            
        Returns:
            list or None: Safe bbox format or None if invalid
        """
        if bbox is None:
            return None
            
        try:
            # Handle different input types
            if hasattr(bbox, '__iter__') and not isinstance(bbox, str):
                bbox_list = list(bbox)
                
                # Ensure we have at least 4 elements
                if len(bbox_list) >= 4:
                    # Convert to floats and ensure they are valid numbers
                    safe_bbox = []
                    for i in range(4):
                        try:
                            val = float(bbox_list[i])
                            if not np.isnan(val) and not np.isinf(val):
                                safe_bbox.append(val)
                            else:
                                return None
                        except (ValueError, TypeError):
                            return None
                    
                    # Validate bbox coordinates make sense
                    x1, y1, x2, y2 = safe_bbox
                    if x2 > x1 and y2 > y1 and all(coord >= 0 for coord in safe_bbox):
                        return safe_bbox
                        
        except Exception as e:
            log.debug(f"Bbox conversion error: {e}")
            
        return None
    
    def _analyze_movement_pattern(self):
        """
        Analyze movement pattern for fall detection using simplified heuristics.
        
        Returns:
            dict or None: Fall detection result
        """
        if len(self.frame_buffer) < 2:
            return None
            
        try:
            # Get first and last frames
            first_frame = self.frame_buffer[0]
            last_frame = self.frame_buffer[-1]
            
            # Simplified fall detection based on bbox changes
            if (first_frame['bbox'] is not None and 
                last_frame['bbox'] is not None):
                
                return self._analyze_bbox_changes(first_frame['bbox'], last_frame['bbox'])
            
            # Fallback to frame difference analysis
            return self._analyze_frame_difference(first_frame['frame'], last_frame['frame'])
            
        except Exception as e:
            log.error(f"Movement analysis error: {e}")
            print(f"Movement analysis error: {e}")  # Add console output for debugging
            return None
    
    def _analyze_bbox_changes(self, bbox1, bbox2):
        """
        Analyze bounding box changes to detect falls.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            dict or None: Fall result
        """
        try:
            # Use safe conversion for both bboxes
            safe_bbox1 = self._safe_bbox_conversion(bbox1)
            safe_bbox2 = self._safe_bbox_conversion(bbox2)
            
            if safe_bbox1 is None or safe_bbox2 is None:
                return None
            
            # Convert to numpy arrays for safe arithmetic operations
            bbox1_arr = np.array(safe_bbox1, dtype=np.float64)
            bbox2_arr = np.array(safe_bbox2, dtype=np.float64)
            
            # Calculate dimensions safely
            w1 = bbox1_arr[2] - bbox1_arr[0]
            h1 = bbox1_arr[3] - bbox1_arr[1]
            w2 = bbox2_arr[2] - bbox2_arr[0]
            h2 = bbox2_arr[3] - bbox2_arr[1]
            
            # Validate dimensions
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return None
            
            # Calculate aspect ratios
            aspect_ratio1 = w1 / h1
            aspect_ratio2 = w2 / h2
            
            # Calculate center positions
            center1_y = (bbox1_arr[1] + bbox1_arr[3]) / 2
            center2_y = (bbox2_arr[1] + bbox2_arr[3]) / 2
            
            # Fall detection criteria
            aspect_change = aspect_ratio2 / aspect_ratio1 if aspect_ratio1 > 0 else 1.0
            vertical_movement = abs(center2_y - center1_y)
            
            # Heuristic: Person becomes wider and moves down significantly
            if (aspect_change > 1.5 and  # Person becomes much wider
                vertical_movement > 20 and  # Significant vertical movement
                center2_y > center1_y):  # Moving downward
                
                confidence = min(0.9, 0.5 + (aspect_change - 1.5) * 0.3 + min(vertical_movement / 100, 0.4))
                
                if confidence >= self.confidence_threshold:
                    return {
                        'fall_detected': True,
                        'confidence': confidence,
                        'angle': 90.0 - (45.0 / aspect_change),  # Estimated angle
                        'category': 'fall',
                        'method': 'bbox_analysis'
                    }
                    
        except Exception as e:
            log.error(f"Bbox analysis error: {e}")
            print(f"Bbox analysis error: {e}, bbox1={bbox1}, bbox2={bbox2}")  # Debug info
            
        return None
    
    def _analyze_frame_difference(self, frame1, frame2):
        """
        Analyze frame differences for fall detection.
        
        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)
            
        Returns:
            dict or None: Fall result
        """
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = np.mean(frame1, axis=2).astype(np.uint8)
            else:
                gray1 = frame1
                
            if len(frame2.shape) == 3:
                gray2 = np.mean(frame2, axis=2).astype(np.uint8)
            else:
                gray2 = frame2
                
            # Calculate frame difference
            diff = np.abs(gray2.astype(np.float32) - gray1.astype(np.float32))
            
            # Analyze movement patterns
            movement_intensity = np.mean(diff)
            horizontal_movement = np.mean(np.abs(np.diff(diff, axis=1)))
            vertical_movement = np.mean(np.abs(np.diff(diff, axis=0)))
            
            # Simple fall detection heuristic
            movement_ratio = horizontal_movement / (vertical_movement + 1e-6)
            
            if (movement_intensity > 15 and  # Significant movement
                movement_ratio > 1.3):  # More horizontal than vertical movement
                
                confidence = min(0.85, movement_intensity / 50 + movement_ratio / 5)
                
                if confidence >= self.confidence_threshold:
                    return {
                        'fall_detected': True,
                        'confidence': confidence,
                        'angle': 60.0 + movement_ratio * 10,  # Estimated angle
                        'category': 'fall',
                        'method': 'frame_difference'
                    }
                    
        except Exception as e:
            log.error(f"Frame difference analysis error: {e}")
            
        return None
    
    def reset(self):
        """Reset detector state."""
        self.frame_buffer.clear()
        self.previous_frame = None
        self.previous_timestamp = None
        log.debug("Fall detector state reset")
    
    def get_stats(self):
        """Get detector statistics."""
        return {
            'confidence_threshold': self.confidence_threshold,
            'min_time_interval': self.min_time_interval,
            'buffer_size': len(self.frame_buffer),
            'max_buffer_size': self.max_buffer_size
        }
