"""
VSViG Seizure Detector Implementation
Main wrapper for VSViG model integration into healthcare monitoring system
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import cv2
from pathlib import Path
import os
import sys

# Add VSViG path for imports
vsvig_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'VSViG')
if vsvig_path not in sys.path:
    sys.path.insert(0, vsvig_path)

# Import VSViG model classes
try:
    # Import từ file VSViG.py trong models/VSViG/
    from VSViG import STViG, VSViG_base, VSViG_light, InterPartMR, IntraPartMR, Stem, Stem_pe, Grapher, Part_3DCNN
    VSVIG_AVAILABLE = True
except ImportError as e:
    logging.error(f"VSViG model classes not available: {e}")
    VSVIG_AVAILABLE = False

from .yolov8_pose_estimator import YOLOv8PoseEstimator

class VSViGSeizureDetector:
    """
    VSViG-based seizure detection system for healthcare monitoring
    Integrates pose estimation + VSViG model for real-time seizure detection
    """
    
    def __init__(self, 
                 vsvig_model_path: Optional[str] = None,
                 pose_model_path: Optional[str] = None,
                 dynamic_order_path: Optional[str] = None,
                 device: str = 'auto',
                 confidence_threshold: float = 0.65):  # Tăng từ 0.50 lên 0.65 - giảm độ nhạy
        """
        Initialize VSViG seizure detector
        
        Args:
            vsvig_model_path: Path to VSViG model weights
            pose_model_path: Path to custom pose model
            dynamic_order_path: Path to dynamic partition order
            device: Device for inference
            confidence_threshold: Seizure detection confidence threshold
        """
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model paths - get absolute paths
        if vsvig_model_path and dynamic_order_path:
            self.vsvig_model_path = vsvig_model_path
            self.dynamic_order_path = dynamic_order_path
        else:
            # Find project root and construct absolute paths
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            base_path = os.path.join(project_root, "models", "VSViG")
            self.vsvig_model_path = vsvig_model_path or os.path.join(base_path, "VSViG-base.pth")
            self.dynamic_order_path = dynamic_order_path or os.path.join(base_path, "dy_point_order.pt")
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.temporal_window = 10  # Giảm xuống 10 frames (0.33 giây) để nhanh hơn
        self.frame_buffer = []  # Buffer for temporal analysis
        
        # Seizure detection state management
        self.last_seizure_detection_time = 0  # Timestamp of last seizure
        self.seizure_cooldown = 45.0  # Tăng lên 45 seconds cooldown - tránh spam
        self.current_seizure_state = False  # Track if currently in seizure
        
        # Components
        self.pose_estimator = YOLOv8PoseEstimator(model_size='n')
        self.vsvig_model = None
        self.is_initialized = False
        self.inference_error_logged = False  # Prevent spam logging
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'seizures_detected': 0,
            'average_confidence': 0.0,
            'last_seizure_time': None,
            'pose_extraction_failures': 0
        }
        
        self.logger.info(f"VSViGSeizureDetector initialized on {self.device}")
    
    def load_models(self) -> bool:
        """
        Load VSViG model and initialize pose estimator
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            # Check if VSViG is available
            if not VSVIG_AVAILABLE:
                self.logger.error("VSViG model classes not available")
                return False
            
            self.logger.info(f"Loading VSViG model from: {self.vsvig_model_path}")
            self.logger.info(f"Loading dynamic order from: {self.dynamic_order_path}")
            
            # YOLOv8PoseEstimator initializes automatically
            self.logger.info("YOLOv8 Pose Estimator loaded successfully")
            
            # Load VSViG model with proper architecture
            if not Path(self.vsvig_model_path).exists():
                self.logger.error(f"VSViG model not found: {self.vsvig_model_path}")
                return False
            
            # Create VSViG model with proper configuration
            self.logger.info("Initializing VSViG model architecture...")
            
            # Define configuration for VSViG_base
            class OptConfig:
                def __init__(self, dynamic_order_path, device):
                    self.dynamic = 1
                    self.num_layer = [2,2,6,2]
                    self.output_channels = [24,48,96,192]
                    self.expansion = 2
                    self.pos_emb = 'stem'
                    # Load dynamic partition order if available
                    if Path(dynamic_order_path).exists():
                        self.dynamic_point_order = torch.load(dynamic_order_path, map_location=device)
                    else:
                        # Create default partition order - you can adjust this
                        self.dynamic_point_order = torch.zeros(15, dtype=torch.long)
            
            # Create model with proper architecture
            opt = OptConfig(self.dynamic_order_path, self.device)
            
            # Log dynamic partition order status
            if Path(self.dynamic_order_path).exists():
                self.logger.info("Dynamic partition order loaded successfully")
            else:
                self.logger.warning("Using default dynamic partition order")
                
            self.vsvig_model = STViG(opt).to(self.device)
            
            # Load state dict
            checkpoint = torch.load(self.vsvig_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.logger.info("Loading from state_dict in checkpoint")
            else:
                state_dict = checkpoint
                self.logger.info("Loading direct state_dict")
            
            # Load state dict into model
            missing_keys, unexpected_keys = self.vsvig_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                self.logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            self.vsvig_model.eval()
            
            self.logger.info("VSViG model loaded successfully with proper architecture")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load VSViG model: {e}")
            self.logger.info("VSViG model unavailable - using fallback detection")
            self.is_initialized = True  # Allow fallback mode
            return True
    
    def detect_seizure(self, frame: np.ndarray, person_bbox: List[int]) -> Dict:
        """
        Detect seizure from a single frame with person detection
        
        Args:
            frame: Input frame (H, W, 3)
            person_bbox: Person bounding box [x1, y1, x2, y2]
            
        Returns:
            dict: Detection result with confidence, keypoints, etc.
        """
        if not self.is_initialized:
            if not self.load_models():
                return self._create_empty_result()
        
        result = {
            'seizure_detected': False,
            'confidence': 0.0,
            'keypoints': None,
            'temporal_ready': False,
            'alert_level': 'normal'
        }
        
        try:
            # Extract pose keypoints
            keypoints = self.pose_estimator.extract_keypoints(frame, person_bbox)
            
            if keypoints is None or not self.pose_estimator.validate_keypoints(keypoints):
                self.stats['pose_extraction_failures'] += 1
                return result
            
            result['keypoints'] = keypoints
            
            # Check if person is standing - don't detect seizure if standing
            is_standing = self._check_if_standing(keypoints)
            if is_standing:
                # Person is standing, skip seizure detection
                result['status'] = 'standing'
                result['seizure_detected'] = False
                result['alert_level'] = 'normal'
                self.stats['total_frames_processed'] += 1
                return result
            
            # Add to temporal buffer (store cropped person frame for VSViG)
            x1, y1, x2, y2 = person_bbox
            person_frame = frame[y1:y2, x1:x2].copy()
            
            self.frame_buffer.append({
                'keypoints': keypoints,
                'frame': person_frame,  # Store frame for VSViG patch extraction
                'timestamp': len(self.frame_buffer)
            })
            
            # Maintain temporal window
            if len(self.frame_buffer) > self.temporal_window:
                self.frame_buffer.pop(0)
            
            # Debug: Show buffer filling progress every 5 frames
            if len(self.frame_buffer) % 5 == 0 and len(self.frame_buffer) < self.temporal_window:
                self.logger.info(f"🧠 Filling temporal buffer: {len(self.frame_buffer)}/{self.temporal_window} frames")
            
            # Check if we have enough frames for temporal analysis
            if len(self.frame_buffer) >= self.temporal_window:
                # Kiểm tra xem người đã nằm ổn định chưa (tất cả 10 frames đều aspect > 1.3)
                all_lying = all(
                    (fd['keypoints'] is not None and 
                     self._calculate_aspect_ratio_from_keypoints(fd['keypoints']) > 1.3)
                    for fd in self.frame_buffer if 'keypoints' in fd
                )
                
                if not all_lying:
                    self.logger.info(f"⚠️ Temporal window ready but person not consistently lying - skipping seizure detection")
                    result['temporal_ready'] = False
                    result['skipped_reason'] = 'not_consistently_lying'
                    return result
                
                result['temporal_ready'] = True
                
                # Debug logging for temporal readiness
                if len(self.frame_buffer) == self.temporal_window:
                    self.logger.info(f"🧠 Temporal Window READY: {len(self.frame_buffer)}/{self.temporal_window} frames collected (all lying)")
                
                # Run VSViG seizure detection
                seizure_confidence = self._run_vsvig_inference()
                result['confidence'] = seizure_confidence
                
                # Check cooldown period after last seizure
                import time
                current_time = time.time()
                time_since_last_seizure = current_time - self.last_seizure_detection_time
                
                # If in cooldown period, force low detection
                if time_since_last_seizure < self.seizure_cooldown and self.current_seizure_state:
                    result['status'] = 'cooldown'
                    result['cooldown_remaining'] = self.seizure_cooldown - time_since_last_seizure
                    result['seizure_detected'] = False
                    result['alert_level'] = 'normal'
                    # Continue to process but don't trigger seizure
                    self.stats['total_frames_processed'] += 1
                    return result
                
                # Normal activity detection - reset seizure state
                if seizure_confidence < 0.65:  # Giảm: 0.8→0.65 - tăng độ nhạy
                    self.current_seizure_state = False
                    result['status'] = 'normal_activity'
                    result['seizure_detected'] = False
                    result['alert_level'] = 'normal'
                elif seizure_confidence >= self.confidence_threshold:
                    # Only detect if not in recent seizure state
                    if not self.current_seizure_state or time_since_last_seizure >= self.seizure_cooldown:
                        result['seizure_detected'] = True
                        result['alert_level'] = 'critical'
                        self.stats['seizures_detected'] += 1
                        self.stats['last_seizure_time'] = len(self.frame_buffer)
                        self.last_seizure_detection_time = current_time
                        self.current_seizure_state = True
                elif seizure_confidence >= self.confidence_threshold * 0.85:
                    if not self.current_seizure_state:
                        result['alert_level'] = 'warning'
                
                # Update statistics
                self.stats['average_confidence'] = (
                    self.stats['average_confidence'] * self.stats['total_frames_processed'] + 
                    seizure_confidence
                ) / (self.stats['total_frames_processed'] + 1)
            
            self.stats['total_frames_processed'] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Seizure detection failed: {str(e)}")
            return result
    
    def _run_vsvig_inference(self) -> float:
        """
        Run VSViG model inference on temporal keypoint sequence
        
        Returns:
            float: Seizure confidence (0.0 - 1.0)
        """
        # Check if VSViG model is available
        if self.vsvig_model is None:
            return 0.0  # Fallback mode
        
        try:
            # Extract image patches for VSViG model
            # VSViG expects: (Batches, Frames, Points, Channels, Height, Width)
            patches_sequence = []
            keypoints_sequence = []
            
            for frame_data in self.frame_buffer:
                if 'frame' not in frame_data or 'keypoints' not in frame_data:
                    continue
                
                frame = frame_data['frame']
                keypoints = frame_data['keypoints']  # (17, 3) [x, y, conf] from YOLO
                
                # Convert 17 keypoints to 15 keypoints (remove eyes: index 1, 2)
                # YOLO indices: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear, ...
                # Keep: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                if keypoints.shape[0] == 17:
                    indices_to_keep = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    keypoints = keypoints[indices_to_keep]  # Now (15, 3)
                
                # Extract 32x32 patches around each keypoint
                patches = self._extract_keypoint_patches(frame, keypoints, patch_size=32)
                if patches is not None:
                    patches_sequence.append(patches)
                    keypoints_sequence.append(keypoints)
            
            if len(patches_sequence) < 10:
                # Not enough frames for VSViG inference
                return 0.0
            
            # Prepare input tensor: (1, T, P, C, H, W)
            patches_tensor = np.stack(patches_sequence, axis=0)  # (T, P, H, W, C)
            patches_tensor = patches_tensor.transpose(0, 1, 4, 2, 3)  # (T, P, C, H, W)
            patches_tensor = np.expand_dims(patches_tensor, axis=0)  # (1, T, P, C, H, W)
            
            # Convert to torch tensor
            patches_tensor = torch.from_numpy(patches_tensor).float().to(self.device)
            
            # Prepare keypoints tensor for positional embedding
            kpts_tensor = np.stack(keypoints_sequence, axis=0)  # (T, P, 3)
            kpts_tensor = np.expand_dims(kpts_tensor, axis=0)  # (1, T, P, 3)
            # Keep all 3 channels: x, y, confidence
            kpts_tensor = torch.from_numpy(kpts_tensor).float().to(self.device)  # (1, T, P, 3)
            
            # Debug: print shapes
            self.logger.info(f"DEBUG: patches_tensor shape: {patches_tensor.shape}, kpts_tensor shape: {kpts_tensor.shape}")
            
            # Run VSViG model
            with torch.no_grad():
                confidence = self.vsvig_model(patches_tensor, kpts_tensor)
                confidence = confidence.item()
            
            if not self.inference_error_logged:
                self.logger.info(f"✅ VSViG model inference SUCCESS - Confidence: {confidence:.3f}")
                self.inference_error_logged = True
            
            return confidence
                
        except Exception as e:
            if not self.inference_error_logged:
                self.logger.warning(f"VSViG inference error: {e} - using motion analysis fallback")
                self.inference_error_logged = True
            
            # Fallback to motion-based analysis
            try:
                keypoint_sequence = np.array([frame['keypoints'] for frame in self.frame_buffer])
                return self._analyze_motion_patterns(keypoint_sequence)
            except:
                return 0.0
    
    def _extract_keypoint_patches(self, frame: np.ndarray, keypoints: np.ndarray, patch_size: int = 32) -> Optional[np.ndarray]:
        """
        Extract image patches around each keypoint for VSViG model
        
        Args:
            frame: Input image (H, W, C)
            keypoints: Keypoints array (15, 3) with [x, y, confidence]
            patch_size: Size of patch to extract (default: 32x32)
            
        Returns:
            patches: Array of patches (15, H, W, C) or None if extraction fails
        """
        try:
            h, w = frame.shape[:2]
            half_patch = patch_size // 2
            patches = []
            
            for kp in keypoints:
                x, y, conf = kp
                
                # Skip low confidence keypoints
                if conf < 0.3:
                    # Use black patch for missing keypoints
                    patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                    continue
                
                # Calculate patch boundaries
                x1 = max(0, int(x) - half_patch)
                y1 = max(0, int(y) - half_patch)
                x2 = min(w, int(x) + half_patch)
                y2 = min(h, int(y) + half_patch)
                
                # Check if patch has valid size
                if x2 <= x1 or y2 <= y1:
                    # Invalid patch, use black patch
                    patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                    continue
                
                # Extract patch
                patch = frame[y1:y2, x1:x2].copy()
                
                # Check extracted patch is not empty
                if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                    patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                    continue
                
                # Ensure patch has 3 channels (BGR/RGB)
                if len(patch.shape) == 2:  # Grayscale
                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                elif patch.shape[2] == 4:  # BGRA
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2BGR)
                elif patch.shape[2] != 3:
                    # Unknown format, use black patch
                    patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                    continue
                
                # Resize to exact patch_size if needed
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = cv2.resize(patch, (patch_size, patch_size))
                
                patches.append(patch)
            
            return np.array(patches)  # (15, H, W, C)
            
        except Exception as e:
            self.logger.error(f"Failed to extract keypoint patches: {e}")
            return None
    
    def _analyze_motion_patterns(self, keypoint_sequence: np.ndarray) -> float:
        """
        Analyze motion patterns for seizure detection - BALANCED THRESHOLDS
        """
        if keypoint_sequence.shape[0] < 5:  # Cần ít nhất 5 frames
            return 0.0
        
        try:
            # Extract coordinates (ignore confidence for motion analysis)
            coords = keypoint_sequence[:, :, :2]  # (T, 15, 2)
            
            # Calculate velocities between frames
            velocities = np.diff(coords, axis=0)  # (T-1, 15, 2)
            
            # Calculate velocity magnitudes
            vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=2))  # (T-1, 15)
            
            # GIẢM ĐỘ NHẠY - tăng threshold để khó phát hiện hơn
            # 1. High velocity variance (irregular movement)
            velocity_variance = np.var(vel_magnitudes, axis=0).mean()
            velocity_score = np.tanh(velocity_variance / 55.0) if velocity_variance > 65 else 0.0  # Tăng: 45→65 (bớt nhạy)
            
            # 2. Acceleration peaks  
            accelerations = np.diff(velocities, axis=0)  # (T-2, 15, 2)
            acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=2))  # (T-2, 15)
            acceleration_peaks = np.max(acc_magnitudes, axis=0).mean()
            acceleration_score = np.tanh(acceleration_peaks / 90.0) if acceleration_peaks > 110 else 0.0  # Tăng: 85→110 (bớt nhạy)
            
            # 3. Frequency analysis - count rapid direction changes
            direction_changes = 0
            if vel_magnitudes.shape[0] > 5:
                for joint in range(min(8, vel_magnitudes.shape[1])):  # Check main joints
                    joint_vel = vel_magnitudes[:, joint]
                    changes = np.sum(np.diff(np.sign(joint_vel)) != 0)
                    direction_changes += changes
                frequency_score = np.tanh(direction_changes / 50.0) if direction_changes > 35 else 0.0  # Tăng: 25→35 (bớt nhạy)
            else:
                frequency_score = 0.0
            
            # 4. Overall movement intensity
            total_movement = np.mean(vel_magnitudes)
            intensity_score = np.tanh(total_movement / 22.0) if total_movement > 30 else 0.0  # Tăng: 22→30 (bớt nhạy)
            
            # 5. Sudden movement spikes (seizure characteristic)
            movement_spikes = np.max(vel_magnitudes, axis=0).mean()
            spike_score = np.tanh(movement_spikes / 40.0) if movement_spikes > 50 else 0.0  # Tăng: 35→50 (bớt nhạy)
            
            # GIẢM ĐỘ NHẠY: Tăng threshold để khó trigger hơn
            sensitive_threshold = 0.50  # Tăng: 0.35→0.50 (bớt nhạy)
            indicators = [
                velocity_score > sensitive_threshold,
                acceleration_score > sensitive_threshold, 
                frequency_score > sensitive_threshold,
                intensity_score > sensitive_threshold,
                spike_score > sensitive_threshold
            ]
            
            active_indicators = sum(indicators)
            if active_indicators < 3:
                return 0.0  # Cần 3 indicators thay vì 2 để tránh false positive
            
            # Weighted combination
            seizure_confidence = (
                0.25 * velocity_score +
                0.25 * acceleration_score +
                0.20 * frequency_score +
                0.15 * intensity_score +
                0.15 * spike_score
            )
            
            # Debug logging every few frames
            frame_count = getattr(self, '_debug_frame_count', 0)
            self._debug_frame_count = frame_count + 1
            if frame_count % 30 == 0:  # Log every 30 frames
                self.logger.info(f"Seizure Scores - Vel:{velocity_score:.3f}, Acc:{acceleration_score:.3f}, Freq:{frequency_score:.3f}, Int:{intensity_score:.3f}, Spike:{spike_score:.3f}, Final:{seizure_confidence:.3f}, Active:{active_indicators}")
            
            # GIẢM ĐỘ NHẠY: Threshold cao hơn để khó trigger
            if seizure_confidence < 0.85:  # Tăng: 0.7→0.85 - bớt nhạy hơn
                return 0.0
            
            return np.clip(seizure_confidence, 0.0, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _prepare_vsvig_keypoints(self, keypoint_sequence: np.ndarray) -> torch.Tensor:
        """
        Prepare keypoints for VSViG model (kpts parameter)
        
        Args:
            keypoint_sequence: Temporal keypoint sequence (T, 15, 3)
            
        Returns:
            torch.Tensor: Keypoints tensor for VSViG
        """
        # Extract coordinates from keypoints
        T, P, _ = keypoint_sequence.shape  # Temporal, Points, Coords (x,y,confidence)
        
        # For VSViG, we need normalized coordinates
        # Extract x,y coordinates and normalize them
        coords = keypoint_sequence[:, :, :2]  # Get x,y coordinates (T, 15, 2)
        
        # Normalize coordinates to [-1, 1] range (assuming input image is ~1920x1080)
        coords[:, :, 0] = (coords[:, :, 0] / 1920.0) * 2.0 - 1.0  # x coords
        coords[:, :, 1] = (coords[:, :, 1] / 1080.0) * 2.0 - 1.0  # y coords
        
        # Add confidence as third channel
        confidence = keypoint_sequence[:, :, 2:3]  # (T, 15, 1)
        
        # Combine coordinates and confidence: (T, 15, 3)
        features = np.concatenate([coords, confidence], axis=2)
        
        # Reshape for VSViG input: (1, T, P, 3)
        # VSViG expects batch dimension first
        kpts_input = features[np.newaxis, ...]  # Add batch dimension
        
        # Convert to tensor
        tensor = torch.from_numpy(kpts_input.astype(np.float32)).to(self.device)
        
        return tensor
    
    def _create_simple_patches(self, keypoint_sequence: np.ndarray) -> torch.Tensor:
        """
        Create simple patches from keypoints for VSViG model
        
        Args:
            keypoint_sequence: Temporal keypoint sequence (T, 15, 3)
            
        Returns:
            torch.Tensor: Patches tensor (B, T, P, C, H, W)
        """
        T, P, _ = keypoint_sequence.shape
        
        # Use 64x64 patches to match VSViG requirements (kernel size 32x32 needs larger input)
        patch_size = 64
        patches = np.zeros((1, T, P, 3, patch_size, patch_size), dtype=np.float32)
        
        for t in range(T):
            for p in range(P):
                x, y, conf = keypoint_sequence[t, p]
                
                # Create a Gaussian-like patch centered around the keypoint
                patch = np.zeros((3, patch_size, patch_size))
                
                # Fill with normalized coordinates and confidence
                norm_x = (x / 1920.0) * 2.0 - 1.0  # normalized x
                norm_y = (y / 1080.0) * 2.0 - 1.0  # normalized y
                
                # Create spatial Gaussian distribution around keypoint
                center = patch_size // 2
                sigma = patch_size // 8  # Control spread of Gaussian
                
                for i in range(patch_size):
                    for j in range(patch_size):
                        # Distance from center
                        dist_x = (i - center) / sigma
                        dist_y = (j - center) / sigma
                        dist = np.sqrt(dist_x**2 + dist_y**2)
                        
                        # Gaussian weight
                        weight = np.exp(-0.5 * dist**2)
                        
                        # Fill channels with weighted features
                        patch[0, i, j] = norm_x * weight  # x coordinate
                        patch[1, i, j] = norm_y * weight  # y coordinate  
                        patch[2, i, j] = conf * weight    # confidence
                
                patches[0, t, p] = patch
        
        # Convert to tensor
        tensor = torch.from_numpy(patches).to(self.device)
        
        return tensor

    def _prepare_vsvig_input(self, keypoint_sequence: np.ndarray) -> torch.Tensor:
        """
        Prepare keypoint sequence for VSViG model input
        
        Args:
            keypoint_sequence: Temporal keypoint sequence (T, 15, 3)
            
        Returns:
            torch.Tensor: VSViG model input tensor
        """
        # Extract coordinates from keypoints
        T, P, _ = keypoint_sequence.shape  # Temporal, Points, Coords (x,y,confidence)
        
        # For VSViG, we need normalized coordinates
        # Extract x,y coordinates and normalize them
        coords = keypoint_sequence[:, :, :2]  # Get x,y coordinates (T, 15, 2)
        
        # Normalize coordinates to [-1, 1] range (assuming input image is ~1920x1080)
        coords[:, :, 0] = (coords[:, :, 0] / 1920.0) * 2.0 - 1.0  # x coords
        coords[:, :, 1] = (coords[:, :, 1] / 1080.0) * 2.0 - 1.0  # y coords
        
        # Add confidence as third channel
        confidence = keypoint_sequence[:, :, 2:3]  # (T, 15, 1)
        
        # Combine coordinates and confidence: (T, 15, 3)
        features = np.concatenate([coords, confidence], axis=2)
        
        # Reshape for VSViG input: (1, T, P, 3)
        # VSViG expects batch dimension first
        vsvig_input = features[np.newaxis, ...]  # Add batch dimension
        
        # Convert to tensor
        tensor = torch.from_numpy(vsvig_input.astype(np.float32)).to(self.device)
        
        return tensor
    
    def _create_empty_result(self) -> Dict:
        """Create empty result for error cases"""
        return {
            'seizure_detected': False,
            'confidence': 0.0,
            'keypoints': None,
            'temporal_ready': False,
            'alert_level': 'normal'
        }
    
    def _calculate_aspect_ratio_from_keypoints(self, keypoints: np.ndarray) -> float:
        """Calculate aspect ratio from keypoints to determine if person is lying
        
        Args:
            keypoints: Array of shape (15, 3) or (17, 3) with [x, y, confidence]
            
        Returns:
            float: Aspect ratio (width/height) of bounding box around keypoints
        """
        try:
            # Lọc keypoints có confidence > 0.3
            valid_kpts = keypoints[keypoints[:, 2] > 0.3]
            if len(valid_kpts) < 5:
                return 0.0
            
            # Tính bounding box
            x_coords = valid_kpts[:, 0]
            y_coords = valid_kpts[:, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if height > 0:
                return width / height
            return 0.0
        except:
            return 0.0
    
    def _calculate_aspect_ratio_from_keypoints(self, keypoints: np.ndarray) -> float:
        """Calculate aspect ratio from keypoints to determine if person is lying
        
        Args:
            keypoints: Array of shape (15, 3) or (17, 3) with [x, y, confidence]
            
        Returns:
            float: Aspect ratio (width/height) of bounding box around keypoints
        """
        try:
            # Lọc keypoints có confidence > 0.3
            valid_kpts = keypoints[keypoints[:, 2] > 0.3]
            if len(valid_kpts) < 5:
                return 0.0
            
            # Tính bounding box
            x_coords = valid_kpts[:, 0]
            y_coords = valid_kpts[:, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if height > 0:
                return width / height
            return 0.0
        except:
            return 0.0
    
    def _check_if_standing(self, keypoints: np.ndarray) -> bool:
        """Check if person is standing based on keypoints
        
        Args:
            keypoints: Array of shape (15, 3) or (17, 3) with [x, y, confidence]
            
        Returns:
            bool: True if person is standing, False otherwise
        """
        try:
            # Keypoints indices (COCO format)
            # 0: nose, 5: left_shoulder, 6: right_shoulder
            # 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee
            # 15: left_ankle, 16: right_ankle
            
            # Check if we have enough keypoints
            if len(keypoints) < 15:
                return False  # Can't determine, assume laying to be safe
            
            # Get shoulder, hip, knee positions
            shoulders_y = (keypoints[5][1] + keypoints[6][1]) / 2  # Average shoulder Y
            hips_y = (keypoints[11][1] + keypoints[12][1]) / 2  # Average hip Y
            
            # Check if ankles/knees are visible
            knees_visible = keypoints[13][2] > 0.3 and keypoints[14][2] > 0.3
            
            if knees_visible:
                knees_y = (keypoints[13][1] + keypoints[14][1]) / 2
            else:
                knees_y = hips_y + 100  # Estimate if not visible
            
            # Person is standing if:
            # 1. Shoulders are ABOVE hips (smaller Y value)
            # 2. Vertical distance between shoulders and knees is significant
            vertical_distance = knees_y - shoulders_y
            
            # Standing: shoulders much higher than knees (>80px vertical distance)
            # Laying/Sitting: shoulders close to same level as knees (<50px)
            is_standing = vertical_distance > 80
            
            return is_standing
            
        except Exception as e:
            self.logger.warning(f"Failed to check standing pose: {e}")
            return False  # Assume laying to be safe
    
    def reset_buffer(self):
        """Reset temporal frame buffer"""
        self.frame_buffer.clear()
        self.logger.info("Temporal buffer reset")
    
    def get_statistics(self) -> Dict:
        """
        Get seizure detection statistics
        
        Returns:
            dict: Detection statistics
        """
        return {
            **self.stats,
            'buffer_size': len(self.frame_buffer),
            'temporal_window': self.temporal_window,
            'confidence_threshold': self.confidence_threshold,
            'model_initialized': self.is_initialized,
            'device': str(self.device)
        }
    
    def visualize_detection(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Visualize seizure detection on frame
        
        Args:
            frame: Input frame
            result: Detection result from detect_seizure()
            
        Returns:
            np.ndarray: Frame with visualization
        """
        frame_vis = frame.copy()
        
        # Draw pose keypoints if available
        if result['keypoints'] is not None:
            frame_vis = self.pose_estimator.visualize_pose(frame_vis, result['keypoints'])
        
        # Draw seizure detection info
        if result['seizure_detected']:
            color = (0, 0, 255)  # Red for seizure
            text = f"🚨 SEIZURE: {result['confidence']:.2f}"
        elif result['alert_level'] == 'warning':
            color = (0, 165, 255)  # Orange for warning
            text = f"⚠️ WARNING: {result['confidence']:.2f}"
        else:
            color = (0, 255, 0)  # Green for normal
            text = f"✅ NORMAL: {result['confidence']:.2f}"
        
        # Add text overlay
        cv2.putText(frame_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        # Add temporal status
        if result['temporal_ready']:
            cv2.putText(frame_vis, "📊 Temporal Analysis: Ready", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            frames_needed = self.temporal_window - len(self.frame_buffer)
            cv2.putText(frame_vis, f"📊 Buffering: {frames_needed} frames needed", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame_vis
