"""
Enhanced Pose Estimator with MediaPipe Integration
Combines high-accuracy MediaPipe with existing fallback methods
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Union
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

class EnhancedPoseEstimator:
    """
    Enhanced pose estimator with MediaPipe primary and fallback methods
    """
    
    def __init__(self, use_mediapipe: bool = True, fallback_enabled: bool = True):
        """
        Initialize enhanced pose estimator
        
        Args:
            use_mediapipe: Whether to use MediaPipe as primary method
            fallback_enabled: Whether to enable fallback methods
        """
        self.logger = logging.getLogger(__name__)
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.fallback_enabled = fallback_enabled
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'mediapipe_success': 0,
            'fallback_success': 0,
            'total_failures': 0,
            'avg_processing_time': 0,
            'last_method_used': None
        }
        
        # Initialize MediaPipe if available
        self.mediapipe_estimator = None
        if self.use_mediapipe:
            self._initialize_mediapipe()
        
        # Initialize fallback methods
        self.fallback_methods = []
        if self.fallback_enabled:
            self._initialize_fallback_methods()
        
        self.logger.info(f"Enhanced Pose Estimator initialized: "
                        f"MediaPipe={'✅' if self.mediapipe_estimator else '❌'}, "
                        f"Fallbacks={len(self.fallback_methods)}")
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe pose estimation"""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Create pose estimator with optimized settings for healthcare
            self.mediapipe_estimator = self.mp_pose.Pose(
                model_complexity=1,           # Balanced accuracy/speed
                min_detection_confidence=0.7, # High confidence for healthcare
                min_tracking_confidence=0.5,  # Smooth tracking
                enable_segmentation=False,    # Disable for speed
                smooth_landmarks=True         # Enable smoothing
            )
            
            # MediaPipe to our format mapping (15 keypoints)
            self.mediapipe_mapping = {
                0: 0,   # nose
                2: 1,   # left_eye (using left_eye_inner)
                5: 2,   # right_eye (using right_eye_inner) 
                7: 3,   # left_ear
                8: 4,   # right_ear
                11: 5,  # left_shoulder
                12: 6,  # right_shoulder
                13: 7,  # left_elbow
                14: 8,  # right_elbow
                15: 9,  # left_wrist
                16: 10, # right_wrist
                23: 11, # left_hip
                24: 12, # right_hip
                25: 13, # left_knee
                26: 14, # right_knee
            }
            
            self.logger.info("MediaPipe pose estimator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")
            self.mediapipe_estimator = None
    
    def _initialize_fallback_methods(self):
        """Initialize fallback pose estimation methods"""
        # Method 1: OpenCV DNN-based pose estimation
        try:
            # This would be your existing pose estimation methods
            # For now, we'll use a simple placeholder
            self.fallback_methods.append('opencv_dnn')
            self.logger.info("OpenCV DNN fallback method added")
        except Exception as e:
            self.logger.warning(f"OpenCV DNN fallback failed to initialize: {e}")
        
        # Method 2: Simple contour-based estimation (very basic fallback)
        self.fallback_methods.append('contour_based')
        self.logger.info("Contour-based fallback method added")
    
    def extract_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Extract keypoints using best available method
        
        Args:
            frame: Input frame (H, W, 3)
            person_bbox: Optional person bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Keypoints (15, 3) with [x, y, confidence] or None
        """
        start_time = time.time()
        self.stats['total_frames'] += 1
        
        keypoints = None
        method_used = None
        
        # Try MediaPipe first (highest accuracy)
        if self.mediapipe_estimator:
            try:
                keypoints = self._extract_mediapipe_keypoints(frame, person_bbox)
                if keypoints is not None and self._validate_keypoints(keypoints):
                    method_used = 'mediapipe'
                    self.stats['mediapipe_success'] += 1
            except Exception as e:
                self.logger.warning(f"MediaPipe extraction failed: {e}")
        
        # Try fallback methods if MediaPipe failed
        if keypoints is None and self.fallback_enabled:
            for method in self.fallback_methods:
                try:
                    if method == 'opencv_dnn':
                        keypoints = self._extract_opencv_keypoints(frame, person_bbox)
                    elif method == 'contour_based':
                        keypoints = self._extract_contour_keypoints(frame, person_bbox)
                    
                    if keypoints is not None and self._validate_keypoints(keypoints):
                        method_used = f'fallback_{method}'
                        self.stats['fallback_success'] += 1
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Fallback method {method} failed: {e}")
        
        # Update statistics
        processing_time = time.time() - start_time
        if keypoints is None:
            self.stats['total_failures'] += 1
            method_used = 'failed'
        
        self.stats['last_method_used'] = method_used
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['total_frames'] - 1) + processing_time) 
            / self.stats['total_frames']
        )
        
        return keypoints
    
    def _extract_mediapipe_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Extract keypoints using MediaPipe"""
        # Convert BGR to RGB
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
        results = self.mediapipe_estimator.process(process_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Convert to our format
        keypoints = self._convert_mediapipe_landmarks(
            results.pose_landmarks.landmark,
            process_frame.shape,
            offset_x, offset_y
        )
        
        return keypoints
    
    def _convert_mediapipe_landmarks(self, landmarks, frame_shape, offset_x=0, offset_y=0) -> np.ndarray:
        """Convert MediaPipe landmarks to our 15-keypoint format"""
        height, width = frame_shape[:2]
        keypoints = np.zeros((15, 3))
        
        for our_idx, mp_idx in self.mediapipe_mapping.items():
            if mp_idx < len(landmarks):
                landmark = landmarks[mp_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * width + offset_x
                y = landmark.y * height + offset_y
                confidence = landmark.visibility  # MediaPipe uses visibility
                
                keypoints[our_idx] = [x, y, confidence]
        
        return keypoints
    
    def _extract_opencv_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Fallback: OpenCV-based pose estimation (placeholder)"""
        # This would integrate with your existing OpenCV pose methods
        # For now, return None to trigger next fallback
        return None
    
    def _extract_contour_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Fallback: Very basic contour-based pose estimation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple thresholding to find person shape
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour (assume it's the person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 1000:  # Too small
                return None
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Generate basic keypoints based on contour geometry
            keypoints = np.zeros((15, 3))
            
            # Head (top center)
            keypoints[0] = [x + w//2, y + h//8, 0.5]  # nose
            keypoints[1] = [x + w//3, y + h//8, 0.4]  # left_eye
            keypoints[2] = [x + 2*w//3, y + h//8, 0.4]  # right_eye
            
            # Shoulders (upper body)
            keypoints[5] = [x + w//4, y + h//4, 0.5]  # left_shoulder
            keypoints[6] = [x + 3*w//4, y + h//4, 0.5]  # right_shoulder
            
            # Hips (mid body)
            keypoints[11] = [x + w//3, y + 2*h//3, 0.4]  # left_hip
            keypoints[12] = [x + 2*w//3, y + 2*h//3, 0.4]  # right_hip
            
            # Other keypoints with low confidence
            for i in range(15):
                if keypoints[i, 2] == 0:  # Not set yet
                    keypoints[i] = [x + w//2, y + h//2, 0.2]  # Default to center
            
            return keypoints
            
        except Exception as e:
            self.logger.error(f"Contour-based estimation failed: {e}")
            return None
    
    def _validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.3) -> bool:
        """Validate keypoints quality"""
        if keypoints is None or keypoints.shape[0] != 15:
            return False
        
        # Check if we have enough confident keypoints
        confident_points = np.sum(keypoints[:, 2] > min_confidence)
        
        # Essential keypoints for pose analysis
        essential_indices = [5, 6, 11, 12]  # shoulders and hips
        essential_detected = np.sum(keypoints[essential_indices, 2] > min_confidence)
        
        # Validation criteria
        basic_requirement = confident_points >= 6  # At least 6/15 keypoints
        essential_requirement = essential_detected >= 2  # At least 2/4 essential keypoints
        
        return bool(basic_requirement and essential_requirement)
    
    def visualize_pose(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None, 
                      method_info: bool = True) -> np.ndarray:
        """
        Visualize pose with method-specific styling
        
        Args:
            frame: Input frame
            keypoints: Keypoints to visualize
            method_info: Whether to show method info
            
        Returns:
            np.ndarray: Frame with pose visualization
        """
        if keypoints is None:
            keypoints = self.extract_keypoints(frame)
            if keypoints is None:
                return frame
        
        frame_vis = frame.copy()
        
        # Use MediaPipe's native drawing if it was the method used
        if (self.stats['last_method_used'] == 'mediapipe' and 
            self.mediapipe_estimator and hasattr(self, 'mp_drawing')):
            
            # Convert back to MediaPipe format for native drawing
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mediapipe_estimator.process(rgb_frame)
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_vis,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
            except:
                # Fallback to custom drawing
                frame_vis = self._draw_custom_pose(frame_vis, keypoints)
        else:
            # Use custom drawing for fallback methods
            frame_vis = self._draw_custom_pose(frame_vis, keypoints)
        
        # Add method information
        if method_info:
            self._add_method_info(frame_vis)
        
        return frame_vis
    
    def _draw_custom_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Custom pose drawing for fallback methods"""
        conf_threshold = 0.3
        
        # Method-specific colors
        method = self.stats['last_method_used'] or 'unknown'
        if 'mediapipe' in method:
            base_color = (0, 255, 0)    # Green for MediaPipe
        elif 'fallback' in method:
            base_color = (0, 255, 255)  # Yellow for fallback
        else:
            base_color = (255, 0, 0)    # Red for failed
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > conf_threshold:
                # Color based on confidence
                if conf > 0.7:
                    color = base_color
                elif conf > 0.5:
                    color = tuple(c//2 + 128 for c in base_color)  # Lighter
                else:
                    color = tuple(c//3 + 85 for c in base_color)   # Even lighter
                
                radius = 5 if conf > 0.7 else 3
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        
        # Draw skeleton connections
        connections = [
            (0, 1), (0, 2),  # Head connections
            (5, 6),          # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10), # Right arm
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14)  # Legs
        ]
        
        for p1, p2 in connections:
            if (keypoints[p1, 2] > conf_threshold and keypoints[p2, 2] > conf_threshold):
                pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
                pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
                
                cv2.line(frame, pt1, pt2, base_color, 2)
        
        return frame
    
    def _add_method_info(self, frame: np.ndarray):
        """Add method and performance information to frame"""
        height, width = frame.shape[:2]
        
        # Background for info
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Method info
        method = self.stats['last_method_used'] or 'No detection'
        method_color = (0, 255, 0) if 'mediapipe' in method else (0, 255, 255) if 'fallback' in method else (0, 0, 255)
        
        cv2.putText(frame, f"Method: {method}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_color, 1)
        
        # Performance stats
        if self.stats['total_frames'] > 0:
            mp_rate = self.stats['mediapipe_success'] / self.stats['total_frames'] * 100
            fallback_rate = self.stats['fallback_success'] / self.stats['total_frames'] * 100
            failure_rate = self.stats['total_failures'] / self.stats['total_frames'] * 100
            avg_time = self.stats['avg_processing_time'] * 1000  # ms
            
            cv2.putText(frame, f"MediaPipe: {mp_rate:.1f}%", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Fallback: {fallback_rate:.1f}%", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Failed: {failure_rate:.1f}%", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Avg: {avg_time:.1f}ms", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_statistics(self) -> dict:
        """Get detailed statistics"""
        if self.stats['total_frames'] == 0:
            return {'status': 'No frames processed yet'}
        
        return {
            'total_frames': self.stats['total_frames'],
            'mediapipe_success_rate': self.stats['mediapipe_success'] / self.stats['total_frames'] * 100,
            'fallback_success_rate': self.stats['fallback_success'] / self.stats['total_frames'] * 100,
            'failure_rate': self.stats['total_failures'] / self.stats['total_frames'] * 100,
            'avg_processing_time_ms': self.stats['avg_processing_time'] * 1000,
            'estimated_fps': 1.0 / self.stats['avg_processing_time'] if self.stats['avg_processing_time'] > 0 else 0,
            'last_method_used': self.stats['last_method_used'],
            'mediapipe_available': self.mediapipe_estimator is not None,
            'fallback_methods': len(self.fallback_methods)
        }
    
    def __del__(self):
        """Cleanup resources"""
        if self.mediapipe_estimator:
            try:
                self.mediapipe_estimator.close()
            except:
                pass


# Factory function for easy integration
def get_enhanced_pose_estimator(use_mediapipe: bool = True, fallback_enabled: bool = True) -> EnhancedPoseEstimator:
    """
    Factory function to get enhanced pose estimator
    
    Args:
        use_mediapipe: Whether to use MediaPipe as primary method
        fallback_enabled: Whether to enable fallback methods
        
    Returns:
        EnhancedPoseEstimator: Configured pose estimator
    """
    return EnhancedPoseEstimator(use_mediapipe=use_mediapipe, fallback_enabled=fallback_enabled)
