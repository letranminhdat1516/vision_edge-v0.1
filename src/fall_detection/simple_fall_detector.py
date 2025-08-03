"""
Simple Fall Detection for Healthcare Monitoring System
Integrates with Vision Edge architecture
"""
import os
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Import fall detection components
from .pipeline.fall_detect import FallDetector

log = logging.getLogger(__name__)

class SimpleFallDetector:
    """
    Fall detection component for healthcare monitoring.
    Uses pose estimation and angle analysis to detect person falls.
    """
    
    def __init__(self, model_name='mobilenet', confidence_threshold=0.6):
        """
        Initialize fall detector.
        
        Args:
            model_name: 'mobilenet' or 'movenet'
            confidence_threshold: Minimum confidence for fall detection
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.previous_frame = None
        self.previous_timestamp = None
        self.min_time_interval = 1.0  # Minimum 1 second between fall checks
        
        # Initialize detector
        self._setup_fall_detector()
        
        log.info(f"🩺 Fall detector initialized (model: {model_name}, confidence: {confidence_threshold})")
    
    def _setup_fall_detector(self):
        """Setup fall detection model paths and config."""
        # Get absolute paths to models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'ai_models')
        
        # Model configuration
        tflite_model = os.path.join(models_dir, 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
        edgetpu_model = os.path.join(models_dir, 'posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
        labels_file = os.path.join(models_dir, 'pose_labels.txt')
        
        # Validate model files exist
        if not os.path.exists(tflite_model):
            raise FileNotFoundError(f"TFLite model not found: {tflite_model}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
            
        config = {
            'model': {
                'tflite': tflite_model,
                'edgetpu': edgetpu_model if os.path.exists(edgetpu_model) else None,
            },
            'labels': labels_file,
            'confidence_threshold': self.confidence_threshold,
            'model_name': self.model_name
        }
        
        # Initialize fall detector
        self.fall_detector = FallDetector(**config)
        log.debug("Fall detection models loaded successfully")
    
    def detect_fall(self, current_frame, timestamp=None):
        """
        Detect fall in current frame compared to previous frame.
        
        Args:
            current_frame: Current video frame (numpy array or PIL Image)
            timestamp: Frame timestamp (optional)
            
        Returns:
            dict: Fall detection result
            {
                'fall_detected': bool,
                'confidence': float,
                'angle': float,
                'category': str,
                'keypoints': dict,
                'processing_time': float
            }
        """
        start_time = time.time()
        
        # Default result
        result = {
            'fall_detected': False,
            'confidence': 0.0,
            'angle': 0.0,
            'category': 'no-fall',
            'keypoints': {},
            'processing_time': 0.0
        }
        
        try:
            # Convert frame to PIL Image if needed
            if isinstance(current_frame, np.ndarray):
                current_frame = Image.fromarray(current_frame)
            
            current_time = timestamp or time.time()
            
            # Check if we have a previous frame and enough time has passed
            if (self.previous_frame is not None and 
                self.previous_timestamp is not None and 
                (current_time - self.previous_timestamp) >= self.min_time_interval):
                
                # Process fall detection
                fall_result = self._process_fall_detection(
                    self.previous_frame, current_frame
                )
                
                if fall_result:
                    result.update(fall_result)
                    log.info(f"🚨 Fall detected! Confidence: {result['confidence']:.2f}, Angle: {result['angle']:.1f}°")
                else:
                    log.debug("No fall detected in frame pair")
            
            # Update previous frame
            self.previous_frame = current_frame
            self.previous_timestamp = current_time
            
        except Exception as e:
            log.error(f"Fall detection error: {e}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _process_fall_detection(self, prev_frame, curr_frame):
        """
        Process fall detection using two consecutive frames.
        
        Args:
            prev_frame: Previous frame (PIL Image)
            curr_frame: Current frame (PIL Image)
            
        Returns:
            dict or None: Fall detection result
        """
        try:
            # Use the original fall prediction function
            from .fall_prediction import Fall_prediction
            
            # Process with fall detection algorithm
            fall_result = Fall_prediction(prev_frame, curr_frame)
            
            if fall_result and fall_result.get('category') == 'fall':
                return {
                    'fall_detected': True,
                    'confidence': fall_result.get('confidence', 0.0),
                    'angle': fall_result.get('angle', 0.0),
                    'category': fall_result.get('category', 'fall'),
                    'keypoints': fall_result.get('keypoint_corr', {})
                }
                
        except Exception as e:
            log.error(f"Fall detection processing error: {e}")
            
        return None
    
    def reset(self):
        """Reset detector state."""
        self.previous_frame = None
        self.previous_timestamp = None
        log.debug("Fall detector state reset")
    
    def get_stats(self):
        """Get detector statistics."""
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'min_time_interval': self.min_time_interval,
            'has_previous_frame': self.previous_frame is not None
        }
