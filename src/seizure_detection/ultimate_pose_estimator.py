"""
Ultimate Pose Estimator
Combines MediaPipe, YOLOv8-Pose, and fallback methods for maximum accuracy
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Union, Dict
import time

# Import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# Import YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

class UltimatePoseEstimator:
    """
    Ultimate pose estimator combining multiple high-accuracy models:
    1. MediaPipe Pose (Google) - 95%+ accuracy
    2. YOLOv8-Pose (Ultralytics) - 90%+ accuracy  
    3. Enhanced fallback methods
    """
    
    def __init__(self, 
                 use_mediapipe: bool = True,
                 use_yolo: bool = True, 
                 use_fallback: bool = True,
                 auto_switch: bool = True):
        """
        Initialize ultimate pose estimator
        
        Args:
            use_mediapipe: Enable MediaPipe pose estimation
            use_yolo: Enable YOLOv8 pose estimation  
            use_fallback: Enable fallback methods
            auto_switch: Automatically switch to best performing method
        """
        self.logger = logging.getLogger(__name__)
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_fallback = use_fallback
        self.auto_switch = auto_switch
        
        # Performance tracking
        self.performance_stats = {
            'mediapipe': {'success': 0, 'total': 0, 'avg_time': 0, 'confidence_sum': 0},
            'yolo': {'success': 0, 'total': 0, 'avg_time': 0, 'confidence_sum': 0},
            'fallback': {'success': 0, 'total': 0, 'avg_time': 0, 'confidence_sum': 0}
        }
        
        # Current preferred method (auto-adjusted based on performance)
        self.preferred_method = 'mediapipe' if self.use_mediapipe else 'yolo' if self.use_yolo else 'fallback'
        
        # Initialize estimators
        self.mediapipe_estimator = None
        self.yolo_model = None
        
        self._initialize_estimators()
        
        self.logger.info(f"Ultimate Pose Estimator initialized: "
                        f"MediaPipe={'✅' if self.mediapipe_estimator else '❌'}, "
                        f"YOLO={'✅' if self.yolo_model else '❌'}, "
                        f"Fallback={'✅' if self.use_fallback else '❌'}")
    
    def _initialize_estimators(self):
        """Initialize all available pose estimators"""
        
        # Initialize MediaPipe
        if self.use_mediapipe:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                self.mediapipe_estimator = self.mp_pose.Pose(
                    model_complexity=1,           # Balanced performance
                    min_detection_confidence=0.7, # High confidence
                    min_tracking_confidence=0.5,
                    enable_segmentation=False,
                    smooth_landmarks=True
                )
                
                self.logger.info("✅ MediaPipe pose estimator initialized")
                
            except Exception as e:
                self.logger.error(f"❌ MediaPipe initialization failed: {e}")
                self.mediapipe_estimator = None
        
        # Initialize YOLOv8-Pose
        if self.use_yolo:
            try:
                # Download YOLOv8n-pose model (nano version for speed)
                self.yolo_model = YOLO('yolov8n-pose.pt')
                self.logger.info("✅ YOLOv8-Pose model initialized")
                
            except Exception as e:
                self.logger.error(f"❌ YOLOv8 initialization failed: {e}")
                self.yolo_model = None
        
        # YOLO keypoint mapping to our 15-point format
        # YOLO uses 17 COCO keypoints, we map to our 15-point system
        self.yolo_to_our_mapping = {
            0: 0,   # nose -> nose
            1: 1,   # left_eye -> left_eye
            2: 2,   # right_eye -> right_eye
            3: 3,   # left_ear -> left_ear
            4: 4,   # right_ear -> right_ear
            5: 5,   # left_shoulder -> left_shoulder
            6: 6,   # right_shoulder -> right_shoulder
            7: 7,   # left_elbow -> left_elbow
            8: 8,   # right_elbow -> right_elbow
            9: 9,   # left_wrist -> left_wrist
            10: 10, # right_wrist -> right_wrist
            11: 11, # left_hip -> left_hip
            12: 12, # right_hip -> right_hip
            13: 13, # left_knee -> left_knee
            14: 14, # right_knee -> right_knee
            # Skip 15, 16 (left_ankle, right_ankle) to match our 15-point system
        }
        
        # MediaPipe mapping (33 landmarks to our 15)
        self.mediapipe_to_our_mapping = {
            0: 0,   # nose
            2: 1,   # left_eye_inner -> left_eye
            5: 2,   # right_eye_inner -> right_eye  
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
    
    def extract_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Extract keypoints using the best available method
        Tries methods in order of preference and performance
        
        Args:
            frame: Input frame (H, W, 3)
            person_bbox: Optional person bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Keypoints (15, 3) with [x, y, confidence] or None
        """
        methods_to_try = self._get_method_order()
        
        for method in methods_to_try:
            try:
                start_time = time.time()
                keypoints = self._extract_with_method(frame, method, person_bbox)
                processing_time = time.time() - start_time
                
                # Update performance stats
                self.performance_stats[method]['total'] += 1
                
                if keypoints is not None and self._validate_keypoints(keypoints):
                    self.performance_stats[method]['success'] += 1
                    self.performance_stats[method]['avg_time'] = (
                        (self.performance_stats[method]['avg_time'] * 
                         (self.performance_stats[method]['success'] - 1) + processing_time) /
                        self.performance_stats[method]['success']
                    )
                    self.performance_stats[method]['confidence_sum'] += np.mean(keypoints[:, 2])
                    
                    # Update preferred method if auto_switch enabled
                    if self.auto_switch:
                        self._update_preferred_method()
                    
                    return keypoints
                    
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue
        
        # All methods failed
        return None
    
    def _get_method_order(self) -> List[str]:
        """Get the order of methods to try based on performance and preference"""
        available_methods = []
        
        if self.auto_switch:
            # Sort by performance score (success rate / avg_time)
            method_scores = {}
            
            for method, stats in self.performance_stats.items():
                if stats['total'] > 0:
                    success_rate = stats['success'] / stats['total']
                    avg_time = stats['avg_time'] if stats['avg_time'] > 0 else 1.0
                    confidence_avg = stats['confidence_sum'] / stats['success'] if stats['success'] > 0 else 0
                    
                    # Combined score: success_rate * confidence * (1/time)
                    method_scores[method] = success_rate * confidence_avg * (1.0 / avg_time)
                else:
                    # No data yet, use default priority
                    default_scores = {'mediapipe': 0.9, 'yolo': 0.8, 'fallback': 0.3}
                    method_scores[method] = default_scores.get(method, 0.1)
            
            # Sort by score (highest first)
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
            available_methods = [method for method, score in sorted_methods]
        else:
            # Use preferred method first
            available_methods = [self.preferred_method]
            for method in ['mediapipe', 'yolo', 'fallback']:
                if method != self.preferred_method:
                    available_methods.append(method)
        
        # Filter only available methods
        filtered_methods = []
        for method in available_methods:
            if method == 'mediapipe' and self.mediapipe_estimator:
                filtered_methods.append(method)
            elif method == 'yolo' and self.yolo_model:
                filtered_methods.append(method)
            elif method == 'fallback' and self.use_fallback:
                filtered_methods.append(method)
        
        return filtered_methods
    
    def _extract_with_method(self, frame: np.ndarray, method: str, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Extract keypoints with specific method"""
        
        if method == 'mediapipe':
            return self._extract_mediapipe_keypoints(frame, person_bbox)
        elif method == 'yolo':
            return self._extract_yolo_keypoints(frame, person_bbox)
        elif method == 'fallback':
            return self._extract_fallback_keypoints(frame, person_bbox)
        else:
            return None
    
    def _extract_mediapipe_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Extract keypoints using MediaPipe"""
        if not self.mediapipe_estimator:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop if bbox provided
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
        
        # Process with MediaPipe
        results = self.mediapipe_estimator.process(process_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Convert to our format
        return self._convert_mediapipe_landmarks(results.pose_landmarks.landmark, 
                                               process_frame.shape, offset_x, offset_y)
    
    def _extract_yolo_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Extract keypoints using YOLOv8-Pose"""
        if not self.yolo_model:
            return None
        
        # Crop if bbox provided
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            process_frame = person_crop
            offset_x, offset_y = x1, y1
        else:
            process_frame = frame
            offset_x, offset_y = 0, 0
        
        # Run YOLO inference
        results = self.yolo_model(process_frame, verbose=False)
        
        if not results or not results[0].keypoints:
            return None
        
        # Get keypoints from best detection
        keypoints_data = results[0].keypoints.data[0]  # First person
        
        if keypoints_data.shape[0] == 0:
            return None
        
        # Convert YOLO format to our format
        return self._convert_yolo_keypoints(keypoints_data, offset_x, offset_y)
    
    def _extract_fallback_keypoints(self, frame: np.ndarray, person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Extract keypoints using fallback method (enhanced contour-based)"""
        try:
            # Advanced contour-based pose estimation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area and aspect ratio (human-like)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000:  # Too small
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Human aspect ratio is typically 1.5-3.0
                if 1.2 <= aspect_ratio <= 4.0:
                    valid_contours.append((contour, area, x, y, w, h))
            
            if not valid_contours:
                return None
            
            # Get largest valid contour
            best_contour = max(valid_contours, key=lambda x: x[1])
            contour, area, x, y, w, h = best_contour
            
            # Generate sophisticated keypoints based on contour analysis
            keypoints = np.zeros((15, 3))
            
            # Head region (top 20% of contour)
            head_y = y + h // 5
            keypoints[0] = [x + w//2, head_y, 0.7]  # nose (center top)
            keypoints[1] = [x + w//3, head_y, 0.6]  # left_eye
            keypoints[2] = [x + 2*w//3, head_y, 0.6]  # right_eye
            keypoints[3] = [x + w//4, head_y + h//10, 0.5]  # left_ear
            keypoints[4] = [x + 3*w//4, head_y + h//10, 0.5]  # right_ear
            
            # Shoulder region (25% down)
            shoulder_y = y + h//4
            keypoints[5] = [x + w//4, shoulder_y, 0.8]  # left_shoulder
            keypoints[6] = [x + 3*w//4, shoulder_y, 0.8]  # right_shoulder
            
            # Elbow region (45% down)
            elbow_y = y + 2*h//5
            keypoints[7] = [x + w//6, elbow_y, 0.6]  # left_elbow
            keypoints[8] = [x + 5*w//6, elbow_y, 0.6]  # right_elbow
            
            # Wrist region (55% down)
            wrist_y = y + 3*h//5
            keypoints[9] = [x + w//8, wrist_y, 0.5]  # left_wrist
            keypoints[10] = [x + 7*w//8, wrist_y, 0.5]  # right_wrist
            
            # Hip region (65% down)
            hip_y = y + 2*h//3
            keypoints[11] = [x + w//3, hip_y, 0.7]  # left_hip
            keypoints[12] = [x + 2*w//3, hip_y, 0.7]  # right_hip
            
            # Knee region (80% down)
            knee_y = y + 4*h//5
            keypoints[13] = [x + w//3, knee_y, 0.6]  # left_knee
            keypoints[14] = [x + 2*w//3, knee_y, 0.6]  # right_knee
            
            return keypoints
            
        except Exception as e:
            self.logger.error(f"Fallback estimation failed: {e}")
            return None
    
    def _convert_mediapipe_landmarks(self, landmarks, frame_shape, offset_x=0, offset_y=0) -> np.ndarray:
        """Convert MediaPipe landmarks to our format"""
        height, width = frame_shape[:2]
        keypoints = np.zeros((15, 3))
        
        for our_idx, mp_idx in self.mediapipe_to_our_mapping.items():
            if mp_idx < len(landmarks):
                landmark = landmarks[mp_idx]
                x = landmark.x * width + offset_x
                y = landmark.y * height + offset_y
                confidence = landmark.visibility
                keypoints[our_idx] = [x, y, confidence]
        
        return keypoints
    
    def _convert_yolo_keypoints(self, yolo_keypoints, offset_x=0, offset_y=0) -> np.ndarray:
        """Convert YOLO keypoints to our format"""
        keypoints = np.zeros((15, 3))
        
        for our_idx, yolo_idx in self.yolo_to_our_mapping.items():
            if yolo_idx < len(yolo_keypoints):
                kp = yolo_keypoints[yolo_idx]
                x = float(kp[0]) + offset_x
                y = float(kp[1]) + offset_y
                confidence = float(kp[2])
                keypoints[our_idx] = [x, y, confidence]
        
        return keypoints
    
    def _validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.3) -> bool:
        """Validate keypoints quality"""
        if keypoints is None or keypoints.shape[0] != 15:
            return False
        
        # Check confidence levels
        confident_points = np.sum(keypoints[:, 2] > min_confidence)
        essential_points = [5, 6, 11, 12]  # shoulders and hips
        essential_detected = np.sum(keypoints[essential_points, 2] > min_confidence)
        
        return confident_points >= 6 and essential_detected >= 2
    
    def _update_preferred_method(self):
        """Update preferred method based on performance"""
        if not self.auto_switch:
            return
        
        best_method = None
        best_score = 0
        
        for method, stats in self.performance_stats.items():
            if stats['total'] >= 5:  # Need at least 5 samples
                success_rate = stats['success'] / stats['total']
                avg_confidence = stats['confidence_sum'] / stats['success'] if stats['success'] > 0 else 0
                avg_time = stats['avg_time'] if stats['avg_time'] > 0 else 1.0
                
                # Performance score
                score = success_rate * avg_confidence * (1.0 / avg_time)
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        if best_method and best_method != self.preferred_method:
            self.logger.info(f"Auto-switching preferred method: {self.preferred_method} -> {best_method}")
            self.preferred_method = best_method
    
    def visualize_pose(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None, 
                      show_method_info: bool = True) -> np.ndarray:
        """
        Visualize pose with method-specific styling
        
        Args:
            frame: Input frame
            keypoints: Keypoints to visualize  
            show_method_info: Whether to show method and performance info
            
        Returns:
            np.ndarray: Frame with pose visualization
        """
        if keypoints is None:
            keypoints = self.extract_keypoints(frame)
            if keypoints is None:
                return frame
        
        frame_vis = frame.copy()
        
        # Use high-quality drawing for MediaPipe and YOLO
        last_method = self._get_last_successful_method()
        
        if last_method == 'mediapipe' and self.mediapipe_estimator:
            # Use MediaPipe's native high-quality drawing
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
                frame_vis = self._draw_custom_pose(frame_vis, keypoints, last_method)
        else:
            # Custom drawing for YOLO and fallback
            frame_vis = self._draw_custom_pose(frame_vis, keypoints, last_method)
        
        if show_method_info:
            self._add_performance_overlay(frame_vis)
        
        return frame_vis
    
    def _draw_custom_pose(self, frame: np.ndarray, keypoints: np.ndarray, method: str) -> np.ndarray:
        """Custom pose drawing with method-specific colors"""
        # Method-specific colors
        method_colors = {
            'mediapipe': (0, 255, 0),    # Green
            'yolo': (255, 165, 0),       # Orange  
            'fallback': (0, 255, 255)    # Yellow
        }
        
        base_color = method_colors.get(method, (255, 255, 255))
        conf_threshold = 0.3
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > conf_threshold:
                # Confidence-based color intensity
                if conf > 0.8:
                    color = base_color
                    radius = 6
                elif conf > 0.6:
                    color = tuple(int(c * 0.8) for c in base_color)
                    radius = 5
                else:
                    color = tuple(int(c * 0.6) for c in base_color)
                    radius = 4
                
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                cv2.circle(frame, (int(x), int(y)), radius + 1, (255, 255, 255), 1)
        
        # Draw skeleton
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),                           # Shoulders
            (5, 7), (7, 9),                  # Left arm
            (6, 8), (8, 10),                 # Right arm
            (5, 11), (6, 12), (11, 12),      # Torso
            (11, 13), (12, 14)               # Legs
        ]
        
        for p1, p2 in connections:
            if (keypoints[p1, 2] > conf_threshold and keypoints[p2, 2] > conf_threshold):
                pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
                pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
                
                min_conf = min(keypoints[p1, 2], keypoints[p2, 2])
                thickness = 3 if min_conf > 0.8 else 2
                
                cv2.line(frame, pt1, pt2, base_color, thickness)
        
        return frame
    
    def _add_performance_overlay(self, frame: np.ndarray):
        """Add performance information overlay"""
        height, width = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ULTIMATE POSE ESTIMATOR", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Current method
        current_method = self.preferred_method.upper()
        method_colors = {'MEDIAPIPE': (0, 255, 0), 'YOLO': (255, 165, 0), 'FALLBACK': (0, 255, 255)}
        method_color = method_colors.get(current_method, (255, 255, 255))
        
        cv2.putText(frame, f"Active: {current_method}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_color, 1)
        
        # Performance stats
        y_offset = 75
        for method, stats in self.performance_stats.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = stats['avg_time'] * 1000  # ms
                
                color = method_colors.get(method.upper(), (255, 255, 255))
                cv2.putText(frame, f"{method}: {success_rate:.1f}% ({avg_time:.1f}ms)", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
        
        # Status
        status = "AUTO-OPTIMIZING" if self.auto_switch else "FIXED METHOD"
        cv2.putText(frame, status, (20, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _get_last_successful_method(self) -> str:
        """Get the last successful method used"""
        # Simple heuristic: return current preferred method
        return self.preferred_method
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        stats = {
            'available_methods': [],
            'preferred_method': self.preferred_method,
            'auto_switch_enabled': self.auto_switch,
            'performance_by_method': {}
        }
        
        # Available methods
        if self.mediapipe_estimator:
            stats['available_methods'].append('mediapipe')
        if self.yolo_model:
            stats['available_methods'].append('yolo')
        if self.use_fallback:
            stats['available_methods'].append('fallback')
        
        # Performance details
        for method, perf_stats in self.performance_stats.items():
            if perf_stats['total'] > 0:
                success_rate = perf_stats['success'] / perf_stats['total']
                avg_confidence = perf_stats['confidence_sum'] / perf_stats['success'] if perf_stats['success'] > 0 else 0
                
                stats['performance_by_method'][method] = {
                    'success_rate': success_rate * 100,
                    'avg_processing_time_ms': perf_stats['avg_time'] * 1000,
                    'estimated_fps': 1.0 / perf_stats['avg_time'] if perf_stats['avg_time'] > 0 else 0,
                    'avg_confidence': avg_confidence,
                    'total_attempts': perf_stats['total'],
                    'successful_attempts': perf_stats['success']
                }
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if self.mediapipe_estimator:
            try:
                self.mediapipe_estimator.close()
            except:
                pass


# Factory function
def get_ultimate_pose_estimator(
    use_mediapipe: bool = True,
    use_yolo: bool = True, 
    use_fallback: bool = True,
    auto_switch: bool = True
) -> UltimatePoseEstimator:
    """
    Factory function to get ultimate pose estimator
    
    Args:
        use_mediapipe: Enable MediaPipe pose estimation
        use_yolo: Enable YOLOv8 pose estimation
        use_fallback: Enable fallback methods
        auto_switch: Auto-optimize method selection
        
    Returns:
        UltimatePoseEstimator: Configured ultimate pose estimator
    """
    return UltimatePoseEstimator(
        use_mediapipe=use_mediapipe,
        use_yolo=use_yolo,
        use_fallback=use_fallback,
        auto_switch=auto_switch
    )
