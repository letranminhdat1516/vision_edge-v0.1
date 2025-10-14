"""
Infrastructure layer - Pose Detection Engine with Thread Safety
Simplified thread-safe version để tránh TF Lite conflicts
"""
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add fall-detection paths
current_dir = os.path.dirname(os.path.abspath(__file__))
fall_detection_dir = os.path.join(current_dir, '../ai_models/fall-detection')
sys.path.append(fall_detection_dir)
sys.path.append(os.path.join(fall_detection_dir, 'src'))
sys.path.append(os.path.join(fall_detection_dir, 'src/pipeline'))

from core.entities.pose_detection import Keypoint, PoseDetectionResult


class PoseDetectionEngine:
    """
    Thread-safe pose detection engine với simplified approach
    """
    
    # Global lock to serialize TF Lite access
    _global_lock = threading.Lock()
    _instance_count = 0
    
    def __init__(self):
        PoseDetectionEngine._instance_count += 1
        self.instance_id = PoseDetectionEngine._instance_count
        
        self.confidence_threshold = 0.2
        self.pose_engine = None
        
        # Keypoint names mapping
        self.keypoint_names = [
            'nose', 'left eye', 'right eye', 'left ear', 'right ear',
            'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 
            'left wrist', 'right wrist', 'left hip', 'right hip',
            'left knee', 'right knee', 'left ankle', 'right ankle'
        ]
        
        print(f"🔧 Initializing Pose Detection Engine #{self.instance_id}...")
        self._initialize_engine()
    
    def _initialize_engine(self) -> bool:
        """Initialize pose engine with error handling"""
        try:
            # Use global lock to prevent concurrent TF Lite loading
            with PoseDetectionEngine._global_lock:
                print(f"🧠 Loading TF Lite model for engine #{self.instance_id}...")
                
                from src.pipeline.inference import TFInferenceEngine
                from src.pipeline.pose_engine import PoseEngine
                
                # Config paths
                fall_detection_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    '../ai_models/fall-detection'
                )
                
                model_path = os.path.join(fall_detection_dir, 'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
                labels_path = os.path.join(fall_detection_dir, 'ai_models/pose_labels.txt')
                
                if not os.path.exists(model_path):
                    print(f"❌ Model not found: {model_path}")
                    return False
                    
                if not os.path.exists(labels_path):
                    print(f"❌ Labels not found: {labels_path}")
                    return False
                
                # Create config
                config = {
                    'model': {'tflite': model_path},
                    'labels': labels_path,
                    'confidence_threshold': 0.6,
                    'model_name': 'mobilenet'
                }
                
                # Initialize engine
                tfengine = TFInferenceEngine(
                    model=config['model'],
                    labels=config['labels'],
                    confidence_threshold=config['confidence_threshold']
                )
                
                self.pose_engine = PoseEngine(tfengine, config['model_name'])
                
                print(f"✅ Pose Detection Engine #{self.instance_id} initialized successfully")
                return True
                
        except Exception as e:
            print(f"❌ Failed to initialize Pose Detection Engine #{self.instance_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.pose_engine is not None
    
    def detect_pose(self, frame: np.ndarray, camera_id: str) -> Optional[PoseDetectionResult]:
        """
        Thread-safe pose detection với global lock
        
        Args:
            frame: OpenCV frame (BGR)
            camera_id: ID của camera
            
        Returns:
            PoseDetectionResult hoặc None nếu failed
        """
        if not self.is_ready():
            print(f"❌ Pose engine #{self.instance_id} not ready")
            return self._create_empty_result(camera_id, "Engine not ready")
        
        try:
            # Use global lock to serialize all TF Lite operations
            with PoseDetectionEngine._global_lock:
                start_time = time.time()
                
                # Convert frame to RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame
                
                pil_image = Image.fromarray(rgb_frame)
                
                # Run pose detection with serialized access
                kps, template_image, thumbnail2, inference_time = self.pose_engine._model.execute_model(pil_image)
                
                # Apply padding correction
                corrected_keypoints = self._apply_padding_correction(
                    kps, template_image, thumbnail2, frame.shape
                )
                
                # Create domain entities
                keypoints = []
                for i, (y, x, confidence) in enumerate(corrected_keypoints):
                    if confidence > 0.1:  # Filter low confidence
                        keypoint = Keypoint(
                            index=i,
                            name=self.keypoint_names[i] if i < len(self.keypoint_names) else f"kp_{i}",
                            y=float(y),
                            x=float(x),
                            confidence=float(confidence)
                        )
                        keypoints.append(keypoint)
                
                # Create result entity
                result = PoseDetectionResult(
                    camera_id=camera_id,
                    timestamp=datetime.fromtimestamp(time.time()),
                    keypoints=keypoints,
                    frame_metadata={
                        'original_shape': frame.shape,
                        'template_shape': template_image.shape if hasattr(template_image, 'shape') else None,
                        'thumbnail_shape': thumbnail2.shape if hasattr(thumbnail2, 'shape') else None,
                        'engine_id': self.instance_id,
                        'processing_time': time.time() - start_time
                    },
                    inference_time=inference_time
                )
                
                return result
                
        except Exception as e:
            print(f"❌ Pose detection failed on engine #{self.instance_id}: {e}")
            return self._create_empty_result(camera_id, str(e))
    
    def _create_empty_result(self, camera_id: str, error_msg: str) -> PoseDetectionResult:
        """Create empty result when detection fails"""
        return PoseDetectionResult(
            camera_id=camera_id,
            timestamp=datetime.fromtimestamp(time.time()),
            keypoints=[],
            frame_metadata={'error': error_msg, 'engine_id': self.instance_id},
            inference_time=0.0
        )
    
    def _apply_padding_correction(self, keypoints_raw, template_image, thumbnail, original_shape) -> List[Tuple[float, float, float]]:
        """
        Apply padding correction logic (validated logic từ tests trước)
        """
        if template_image is None or thumbnail is None:
            return [(0, 0, 0) for _ in range(17)]
        
        # Convert PIL Images to numpy arrays if needed
        if hasattr(template_image, 'size'):  # PIL Image
            template_array = np.array(template_image)
            template_height, template_width = template_array.shape[:2]
        else:  # Already numpy array
            template_height, template_width = template_image.shape[:2]
            
        if hasattr(thumbnail, 'size'):  # PIL Image
            thumbnail_array = np.array(thumbnail)
            thumbnail_height, thumbnail_width = thumbnail_array.shape[:2]
        else:  # Already numpy array
            thumbnail_height, thumbnail_width = thumbnail.shape[:2]
        
        original_height, original_width = original_shape[:2]
        
        # Calculate scaling factors
        scale_x = template_width / thumbnail_width
        scale_y = template_height / thumbnail_height
        
        # Calculate padding for aspect ratio preservation
        aspect_ratio_original = original_width / original_height
        aspect_ratio_template = template_width / template_height
        
        if aspect_ratio_original > aspect_ratio_template:
            # Original is wider - vertical padding in template
            new_height = int(template_width / aspect_ratio_original)
            padding_y = (template_height - new_height) // 2
            padding_x = 0
            final_scale_x = original_width / template_width
            final_scale_y = original_height / new_height
        else:
            # Original is taller - horizontal padding in template
            new_width = int(template_height * aspect_ratio_original)
            padding_x = (template_width - new_width) // 2
            padding_y = 0
            final_scale_x = original_width / new_width
            final_scale_y = original_height / template_height
        
        corrected_keypoints = []
        for kp in keypoints_raw:
            if len(kp) >= 3:
                y_tensor, x_tensor, confidence = kp[0], kp[1], kp[2]
                
                # Scale from thumbnail to template coordinates
                y_template = y_tensor * scale_y
                x_template = x_tensor * scale_x
                
                # Remove padding
                y_no_padding = y_template - padding_y
                x_no_padding = x_template - padding_x
                
                # Scale to original image coordinates
                y_original = y_no_padding * final_scale_y
                x_original = x_no_padding * final_scale_x
                
                # Clamp to image bounds
                y_original = max(0, min(original_height - 1, y_original))
                x_original = max(0, min(original_width - 1, x_original))
                
                corrected_keypoints.append((y_original, x_original, confidence))
            else:
                corrected_keypoints.append((0, 0, 0))
        
        return corrected_keypoints


class PoseVisualizationService:
    """Service cho visualization của pose detection results"""

    def __init__(self):
        self.skeleton_connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Body
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (5, 11), (6, 12), (11, 12),
            # Legs
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        
        self.colors = {
            'keypoint': (0, 255, 0),      # Green keypoints
            'skeleton': (255, 0, 0),      # Red skeleton
            'text': (255, 255, 255),      # White text
            'low_conf': (128, 128, 128)   # Gray for low confidence
        }

    def draw_pose_on_frame(self, frame: np.ndarray, pose_result: PoseDetectionResult, 
                          show_skeleton: bool = True, show_labels: bool = False,
                          show_info: bool = True, fps: Optional[float] = None) -> np.ndarray:
        """Draw pose detection results on frame"""
        annotated_frame = frame.copy()
        
        try:
            # Draw keypoints
            for keypoint in pose_result.keypoints:
                if keypoint.confidence > 0.3:
                    color = self.colors['keypoint'] if keypoint.confidence > 0.5 else self.colors['low_conf']
                    center = (int(keypoint.x), int(keypoint.y))
                    
                    # Draw keypoint circle
                    cv2.circle(annotated_frame, center, 4, color, -1)
                    cv2.circle(annotated_frame, center, 6, (0, 0, 0), 2)
                    
                    # Draw label if requested
                    if show_labels:
                        text = f"{keypoint.name}:{keypoint.confidence:.2f}"
                        cv2.putText(annotated_frame, text, (center[0] + 10, center[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Draw skeleton
            if show_skeleton:
                self._draw_skeleton(annotated_frame, pose_result.keypoints)
            
            # Draw info
            if show_info:
                self._draw_info_text(annotated_frame, pose_result, fps)
                
        except Exception as e:
            print(f"⚠️ Error in pose visualization: {e}")
        
        return annotated_frame

    def _draw_skeleton(self, frame: np.ndarray, keypoints: List[Keypoint]):
        """Draw skeleton connections"""
        kp_dict = {kp.index: kp for kp in keypoints if kp.confidence > 0.3}
        
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx in kp_dict and end_idx in kp_dict:
                start_kp = kp_dict[start_idx]
                end_kp = kp_dict[end_idx]
                
                start_point = (int(start_kp.x), int(start_kp.y))
                end_point = (int(end_kp.x), int(end_kp.y))
                
                cv2.line(frame, start_point, end_point, self.colors['skeleton'], 2)

    def _draw_info_text(self, frame: np.ndarray, pose_result: PoseDetectionResult, fps: Optional[float]):
        """Draw info text"""
        info_lines = []
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        visible_keypoints = len([kp for kp in pose_result.keypoints if kp.confidence > 0.3])
        info_lines.append(f"Keypoints: {visible_keypoints}/17")
        
        engine_id = pose_result.frame_metadata.get('engine_id', 'unknown')
        info_lines.append(f"Engine: #{engine_id}")
        
        # Draw info
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y_offset += 25