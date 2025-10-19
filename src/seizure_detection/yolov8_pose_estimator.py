"""
YOLOv8 Pose Estimator - Professional implementation
Uses proven YOLOv8-Pose model from Ultralytics
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Optional, List, Tuple
import time

class YOLOv8PoseEstimator:
    def __init__(self, model_size: str = 'n'):
        """
        Initialize YOLOv8 Pose Estimator
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        # Setup logging
        self.logger = logging.getLogger(f'{__name__}.{model_size}')
        self.logger.setLevel(logging.DEBUG)  # Enable debug logging temporarily
        self.model_size = model_size
        
        # Load YOLOv8-Pose model
        self._load_yolo_model()
        
        # COCO pose keypoints (17 points standard)
        self.keypoint_names = [
            'nose',           # 0
            'left_eye',       # 1  
            'right_eye',      # 2
            'left_ear',       # 3
            'right_ear',      # 4
            'left_shoulder',  # 5
            'right_shoulder', # 6
            'left_elbow',     # 7
            'right_elbow',    # 8
            'left_wrist',     # 9
            'right_wrist',    # 10
            'left_hip',       # 11
            'right_hip',      # 12
            'left_knee',      # 13
            'right_knee',     # 14
            'left_ankle',     # 15
            'right_ankle',    # 16
        ]
        
        # COCO pose skeleton connections
        self.skeleton_connections = [
            # Head
            (0, 1), (0, 2),     # nose to eyes
            (1, 3), (2, 4),     # eyes to ears
            
            # Arms
            (5, 6),             # shoulders
            (5, 7), (7, 9),     # left arm
            (6, 8), (8, 10),    # right arm
            
            # Torso
            (5, 11), (6, 12),   # shoulders to hips
            (11, 12),           # hips
            
            # Legs  
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]
        
        # Performance tracking
        self.total_detections = 0
        self.successful_detections = 0
        self.avg_inference_time = 0.0
        
    def _load_yolo_model(self):
        """Load YOLOv8-Pose model"""
        try:
            model_name = f'yolov8{self.model_size}-pose.pt'
            self.logger.info(f"🔄 Loading YOLOv8-Pose model: {model_name}")
            
            self.model = YOLO(model_name)
            self.model_loaded = True
            
            self.logger.info(f"✅ YOLOv8-Pose {self.model_size} loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load YOLOv8-Pose: {e}")
            self.model_loaded = False
            self.model = None
    
    def extract_keypoints(self, frame: np.ndarray, confidence_threshold: float = 0.5, person_bbox=None) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from frame
        
        Args:
            frame: Input image (H, W, 3)
            confidence_threshold: Minimum confidence for person detection OR person_bbox for compatibility
            person_bbox: Optional person bounding box (for compatibility - not used)
            
        Returns:
            np.ndarray: Keypoints (17, 3) with [x, y, confidence] or None
        """
        if not self.model_loaded:
            return None
            
        try:
            start_time = time.time()
            
            # Handle compatibility: if second parameter looks like a bbox, use default confidence
            if isinstance(confidence_threshold, (list, tuple, np.ndarray, dict)) and (
                (isinstance(confidence_threshold, (list, tuple, np.ndarray)) and len(confidence_threshold) >= 4) or
                isinstance(confidence_threshold, dict)
            ):
                # Second parameter is actually person_bbox, use default confidence
                actual_confidence_threshold = 0.5
                person_bbox = confidence_threshold
            else:
                # Normal case: second parameter is confidence threshold
                if isinstance(confidence_threshold, (list, tuple)):
                    if len(confidence_threshold) > 0:
                        actual_confidence_threshold = float(confidence_threshold[0])
                    else:
                        # Empty list/tuple, use default
                        actual_confidence_threshold = 0.5
                        print("⚠️ YOLOv8-Pose: Empty confidence threshold list, using default 0.5")
                else:
                    actual_confidence_threshold = float(confidence_threshold)
            
            # Run inference
            results = self.model(frame, verbose=False)
            
            # Update timing
            inference_time = time.time() - start_time
            self.total_detections += 1
            self.avg_inference_time = (
                (self.avg_inference_time * (self.total_detections - 1) + inference_time) 
                / self.total_detections
            )
            
            # Process results
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get the person with highest confidence
                best_person_idx = 0
                best_confidence = 0.0
                
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    # Safely extract confidence
                    if hasattr(box, 'conf') and box.conf is not None:
                        conf_tensor = box.conf
                        try:
                            if hasattr(conf_tensor, 'item'):
                                conf_value = conf_tensor.item()
                            elif isinstance(conf_tensor, (list, tuple)) and len(conf_tensor) > 0:
                                conf_value = float(conf_tensor[0])
                            elif isinstance(conf_tensor, dict):
                                # Handle dict case - try common keys
                                self.logger.debug(f"Confidence tensor is dict: {conf_tensor}")
                                conf_value = conf_tensor.get('confidence', conf_tensor.get('conf', conf_tensor.get(0, 0.0)))
                                conf_value = float(conf_value) if not isinstance(conf_value, dict) else 0.0
                            else:
                                try:
                                    if isinstance(conf_tensor, (int, float)):
                                        conf_value = float(conf_tensor)
                                    elif hasattr(conf_tensor, '__float__'):
                                        conf_value = float(conf_tensor)
                                    else:
                                        self.logger.debug(f"Unknown conf_tensor type: {type(conf_tensor)}, value: {conf_tensor}")
                                        conf_value = 0.0
                                except (TypeError, ValueError):
                                    self.logger.debug(f"Failed to convert conf_tensor: {type(conf_tensor)}, value: {conf_tensor}")
                                    conf_value = 0.0
                        except (TypeError, ValueError, KeyError) as e:
                            self.logger.debug(f"Failed to extract confidence: {e}, tensor: {conf_tensor}")
                            conf_value = 0.0
                        
                        if conf_value > best_confidence:
                            best_confidence = conf_value
                            best_person_idx = i
                
                # Check if confidence is high enough
                if best_confidence < actual_confidence_threshold:
                    return None
                
                # Extract keypoints for best person
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    keypoints_tensor = results[0].keypoints.data
                    if len(keypoints_tensor) > best_person_idx:
                        keypoints_data = keypoints_tensor[best_person_idx]  # Shape: (17, 3)
                        
                        # Convert to numpy array safely
                        try:
                            if hasattr(keypoints_data, 'cpu'):
                                keypoints = keypoints_data.cpu().numpy()
                            elif hasattr(keypoints_data, 'numpy'):
                                keypoints = keypoints_data.numpy()
                            elif isinstance(keypoints_data, dict):
                                self.logger.debug(f"Keypoints data is dict: {keypoints_data}")
                                return None
                            else:
                                keypoints = np.array(keypoints_data)
                        except Exception as conv_e:
                            self.logger.debug(f"Failed to convert keypoints: {conv_e}, type: {type(keypoints_data)}")
                            return None
                        
                        # Validate keypoints
                        if self._validate_keypoints(keypoints):
                            self.successful_detections += 1
                            return keypoints
            
            return None
            
        except Exception as e:
            print(f"🔍 YOLOv8-Pose DEBUG: Exception {type(e).__name__}: {e}")
            import traceback
            print(f"🔍 YOLOv8-Pose DEBUG: Traceback:")
            traceback.print_exc()
            self.logger.warning(f"YOLOv8-Pose extraction failed: {e}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _validate_keypoints(self, keypoints: np.ndarray, min_visible_points: int = 5) -> bool:
        """Validate keypoints quality"""
        try:
            if keypoints is None:
                return False
            
            if not isinstance(keypoints, np.ndarray):
                self.logger.debug(f"Keypoints not numpy array: {type(keypoints)}")
                return False
                
            if keypoints.shape[0] != 17:
                self.logger.debug(f"Keypoints wrong shape: {keypoints.shape}, expected (17, 3)")
                return False
            
            if len(keypoints.shape) < 2 or keypoints.shape[1] != 3:
                self.logger.debug(f"Keypoints wrong dimensions: {keypoints.shape}, expected (17, 3)")
                return False

            # Count visible keypoints (confidence > 0.3)
            try:
                visible_points = np.sum(keypoints[:, 2] > 0.3)
            except Exception as e:
                self.logger.debug(f"Error calculating visible points: {e}")
                return False

            # Must have essential body parts
            essential_points = [5, 6, 11, 12]  # shoulders and hips
            try:
                essential_visible = np.sum(keypoints[essential_points, 2] > 0.3)
            except Exception as e:
                self.logger.debug(f"Error calculating essential points: {e}")
                return False

            return bool(visible_points >= min_visible_points and essential_visible >= 2)
            
        except Exception as e:
            self.logger.debug(f"Keypoints validation error: {e}")
            return False
    
    def validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.3) -> bool:
        """Public method for keypoints validation (compatibility with VSViG)"""
        return self._validate_keypoints(keypoints, min_visible_points=5)
    
    def visualize_pose(self, frame: np.ndarray, keypoints: np.ndarray, 
                      show_confidence: bool = True) -> np.ndarray:
        """
        Visualize pose keypoints and skeleton on frame
        
        Args:
            frame: Input frame
            keypoints: Keypoints array (17, 3)
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with pose visualization
        """
        if keypoints is None:
            return frame
            
        frame_vis = frame.copy()
        
        # Color scheme
        joint_color = (0, 255, 0)      # Green for joints
        skeleton_color = (0, 255, 255)  # Yellow for skeleton
        text_color = (255, 255, 255)    # White for text
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:  # Only draw visible keypoints
                # Determine size based on confidence
                if conf > 0.8:
                    radius = 6
                elif conf > 0.5:
                    radius = 5
                else:
                    radius = 4
                
                # Draw keypoint
                cv2.circle(frame_vis, (int(x), int(y)), radius, joint_color, -1)
                cv2.circle(frame_vis, (int(x), int(y)), radius + 1, text_color, 1)
                
                # Show keypoint number and confidence
                if show_confidence:
                    cv2.putText(frame_vis, f"{i}", 
                              (int(x) + 8, int(y) - 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                    cv2.putText(frame_vis, f"{conf:.2f}", 
                              (int(x) + 8, int(y) + 12),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.25, text_color, 1)
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            # Check if both points are visible
            if (keypoints[pt1_idx, 2] > 0.3 and keypoints[pt2_idx, 2] > 0.3):
                pt1 = (int(keypoints[pt1_idx, 0]), int(keypoints[pt1_idx, 1]))
                pt2 = (int(keypoints[pt2_idx, 0]), int(keypoints[pt2_idx, 1]))
                
                # Line thickness based on confidence
                min_conf = min(keypoints[pt1_idx, 2], keypoints[pt2_idx, 2])
                thickness = 3 if min_conf > 0.7 else 2
                
                cv2.line(frame_vis, pt1, pt2, skeleton_color, thickness)
        
        # Add info overlay
        self._add_info_overlay(frame_vis, keypoints)
        
        return frame_vis
    
    def _add_info_overlay(self, frame: np.ndarray, keypoints: np.ndarray):
        """Add information overlay"""
        height, width = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "YOLOv8-Pose Estimator", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Stats
        valid_keypoints = np.sum(keypoints[:, 2] > 0.3) if keypoints is not None else 0
        success_rate = (self.successful_detections / max(self.total_detections, 1)) * 100
        
        cv2.putText(frame, f"Valid Keypoints: {valid_keypoints}/17", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Success Rate: {success_rate:.1f}%", (15, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Avg Time: {self.avg_inference_time*1000:.1f}ms", (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_keypoint_name(self, index: int) -> str:
        """Get keypoint name by index"""
        if 0 <= index < len(self.keypoint_names):
            return self.keypoint_names[index]
        return f"unknown_{index}"
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'success_rate': (self.successful_detections / max(self.total_detections, 1)) * 100,
            'avg_inference_time_ms': self.avg_inference_time * 1000,
            'model_size': self.model_size
        }
