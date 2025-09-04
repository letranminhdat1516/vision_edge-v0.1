"""
MediaPipe Pose Estimator Integration
High-accuracy pose estimation using Google's MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple
import logging

class MediaPipePoseEstimator:
    """
    MediaPipe-based pose estimation for high accuracy keypoint detection
    """
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose estimator
        
        Args:
            model_complexity: 0=Light, 1=Full, 2=Heavy (default: 1)
            min_detection_confidence: Minimum detection confidence (0.5-1.0)
            min_tracking_confidence: Minimum tracking confidence (0.5-1.0)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create pose estimator
        self.pose_estimator = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,  # Disable segmentation for speed
            smooth_landmarks=True       # Enable landmark smoothing
        )
        
        # MediaPipe landmark indices mapping to our 15-point format
        self.landmark_mapping = {
            0: 0,   # nose -> nose
            1: 2,   # left_eye_inner -> left_eye (approximation)
            2: 1,   # right_eye_inner -> right_eye (approximation)  
            3: 7,   # left_ear -> left_ear
            4: 8,   # right_ear -> right_ear
            5: 11,  # left_shoulder -> left_shoulder
            6: 12,  # right_shoulder -> right_shoulder
            7: 13,  # left_elbow -> left_elbow
            8: 14,  # right_elbow -> right_elbow
            9: 15,  # left_wrist -> left_wrist
            10: 16, # right_wrist -> right_wrist
            11: 23, # left_hip -> left_hip
            12: 24, # right_hip -> right_hip
            13: 25, # left_knee -> left_knee
            14: 26, # right_knee -> right_knee
        }
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee'
        ]
        
        self.is_initialized = True
        self.logger.info("MediaPipe Pose Estimator initialized successfully")
    
    def extract_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Extract keypoints using MediaPipe
        
        Args:
            frame: Input frame (H, W, 3)
            person_bbox: Optional person bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Keypoints (15, 3) with [x, y, confidence] or None
        """
        try:
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop to person region if bbox provided
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                person_crop = rgb_frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    return None
                process_frame = person_crop
                offset_x, offset_y = x1, y1
            else:
                process_frame = rgb_frame
                offset_x, offset_y = 0, 0
            
            # Run MediaPipe pose detection
            results = self.pose_estimator.process(process_frame)
            
            if results.pose_landmarks is None:
                return None
            
            # Convert MediaPipe landmarks to our format
            keypoints = self._convert_mediapipe_landmarks(
                results.pose_landmarks.landmark,
                process_frame.shape,
                offset_x, offset_y
            )
            
            return keypoints
            
        except Exception as e:
            self.logger.error(f"MediaPipe keypoint extraction failed: {e}")
            return None
    
    def _convert_mediapipe_landmarks(self, landmarks, frame_shape, offset_x=0, offset_y=0) -> np.ndarray:
        """
        Convert MediaPipe landmarks to our 15-keypoint format
        
        Args:
            landmarks: MediaPipe landmark list
            frame_shape: Shape of the processed frame (H, W, C)
            offset_x, offset_y: Offset to add back to coordinates
            
        Returns:
            np.ndarray: Converted keypoints (15, 3)
        """
        height, width = frame_shape[:2]
        keypoints = np.zeros((15, 3))
        
        for our_idx, mp_idx in self.landmark_mapping.items():
            if mp_idx < len(landmarks):
                landmark = landmarks[mp_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * width + offset_x
                y = landmark.y * height + offset_y
                confidence = landmark.visibility  # MediaPipe uses visibility as confidence
                
                keypoints[our_idx] = [x, y, confidence]
        
        return keypoints
    
    def visualize_pose(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None, 
                      use_mediapipe_drawing: bool = True) -> np.ndarray:
        """
        Visualize pose with MediaPipe's high-quality drawing
        
        Args:
            frame: Input frame
            keypoints: Our format keypoints (optional)
            use_mediapipe_drawing: Use MediaPipe's native drawing
            
        Returns:
            np.ndarray: Frame with pose visualization
        """
        if keypoints is None:
            # Extract keypoints first
            keypoints = self.extract_keypoints(frame)
            if keypoints is None:
                return frame
        
        frame_vis = frame.copy()
        
        if use_mediapipe_drawing:
            # Use MediaPipe's native high-quality drawing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame_vis,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
        else:
            # Use our custom drawing (fallback)
            frame_vis = self._draw_custom_pose(frame_vis, keypoints)
        
        return frame_vis
    
    def _draw_custom_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Custom pose drawing with our style
        
        Args:
            frame: Input frame
            keypoints: Keypoints to draw (15, 3)
            
        Returns:
            np.ndarray: Frame with custom pose drawing
        """
        conf_threshold = 0.7  # High confidence threshold for MediaPipe
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > conf_threshold:
                # High confidence color coding
                if conf > 0.9:
                    color = (0, 255, 0)    # Bright green
                elif conf > 0.8:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                radius = 5 if conf > 0.9 else 4
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                
                # Add keypoint label
                if conf > 0.8:
                    cv2.putText(frame, self.keypoint_names[i], (int(x), int(y-8)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw skeleton connections
        connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Torso  
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs
            (11, 13), (12, 14)
        ]
        
        for p1, p2 in connections:
            if (keypoints[p1, 2] > conf_threshold and keypoints[p2, 2] > conf_threshold):
                pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
                pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
                
                # Connection color based on confidence
                min_conf = min(keypoints[p1, 2], keypoints[p2, 2])
                color = (0, 255, 0) if min_conf > 0.9 else (0, 255, 255) if min_conf > 0.8 else (0, 165, 255)
                thickness = 3 if min_conf > 0.9 else 2
                
                cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.7) -> bool:
        """
        Validate MediaPipe keypoints quality
        
        Args:
            keypoints: Keypoints to validate (15, 3)
            min_confidence: Minimum confidence threshold
            
        Returns:
            bool: True if keypoints are high quality
        """
        if keypoints is None or keypoints.shape[0] != 15:
            return False
        
        # MediaPipe is generally very reliable, so use stricter criteria
        high_conf_points = np.sum(keypoints[:, 2] > min_confidence)
        
        # Essential keypoints for pose analysis
        essential_keypoints = [5, 6, 11, 12]  # shoulders and hips
        essential_detected = np.sum(keypoints[essential_keypoints, 2] > min_confidence)
        
        # MediaPipe quality requirements (stricter)
        basic_requirement = high_conf_points >= 10  # At least 10/15 keypoints
        essential_requirement = essential_detected >= 4  # All essential keypoints
        
        # Average confidence should be high for MediaPipe
        avg_confidence = np.mean(keypoints[keypoints[:, 2] > 0, 2])
        quality_check = avg_confidence > 0.8
        
        return bool(basic_requirement and essential_requirement and quality_check)
    
    def get_statistics(self) -> dict:
        """Get MediaPipe pose estimator statistics"""
        return {
            'model_type': 'MediaPipe Pose',
            'model_loaded': self.is_initialized,
            'accuracy': 'High (95%+)',
            'speed': 'Real-time',
            'keypoints_count': len(self.keypoint_names),
            'confidence_threshold': 0.7,
            'status': 'loaded' if self.is_initialized else 'not_loaded'
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.close()


# Integration helper function
def get_mediapipe_pose_estimator() -> MediaPipePoseEstimator:
    """
    Get MediaPipe pose estimator instance
    
    Returns:
        MediaPipePoseEstimator: Initialized pose estimator
    """
    return MediaPipePoseEstimator(
        model_complexity=1,           # Balanced accuracy/speed
        min_detection_confidence=0.7, # High confidence for healthcare
        min_tracking_confidence=0.5   # Smooth tracking
    )
