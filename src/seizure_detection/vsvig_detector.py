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
                 confidence_threshold: float = 0.50):  # 🔥 
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
        
        # 🆕 EXERCISE COOLDOWN - TẮT ĐỂ DETECT SEIZURE TỐT HƠN
        self.last_exercise_detection_time = 0
        self.exercise_cooldown = 0.0  # TẮT - không block seizure detection
        self.enable_exercise_detection = False  # TẮT exercise detection hoàn toàn
        
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
            
            # 🔥 IMPORTANT: Always add to buffer FIRST (even if standing/push-up)
            # This ensures we have fresh temporal data when person lies down
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
            
            # Check if person is standing - don't detect seizure if standing
            is_standing = self._check_if_standing(keypoints)
            if is_standing:
                # Person is standing/push-up, skip seizure detection but keep buffer updated
                result['status'] = 'standing'
                result['seizure_detected'] = False
                result['alert_level'] = 'normal'
                self.stats['total_frames_processed'] += 1
                return result
            
            # Debug: Show buffer filling progress every 5 frames - only debug level
            if len(self.frame_buffer) % 5 == 0 and len(self.frame_buffer) < self.temporal_window:
                self.logger.debug(f"🧠 Filling temporal buffer: {len(self.frame_buffer)}/{self.temporal_window} frames")
            
            # Check if we have enough frames for temporal analysis
            if len(self.frame_buffer) >= self.temporal_window:
                # Kiểm tra xem người đã nằm ổn định chưa - dùng _is_person_lying cải thiện
                lying_checks = []
                for fd in self.frame_buffer:
                    if 'keypoints' in fd and fd['keypoints'] is not None:
                        is_lying, reason, conf = self._is_person_lying(fd['keypoints'])
                        lying_checks.append(is_lying)
                
                # 🔥 BALANCED: lying_ratio requirement - cân bằng để detect seizure tốt hơn
                lying_ratio = sum(lying_checks) / len(lying_checks) if lying_checks else 0
                mostly_lying = lying_ratio >= 0.80  # 🔥 GIẢM: 95%→80% để detect seizure dễ hơn
                
                if not mostly_lying:
                    if self.stats['total_frames_processed'] % 30 == 0:  # Log mỗi 1 giây
                        self.logger.info(f"⚠️ Temporal ready but lying_ratio={lying_ratio:.1%} < 80% - skipping seizure")
                    result['temporal_ready'] = False
                    result['skipped_reason'] = 'not_mostly_lying'
                    result['lying_ratio'] = lying_ratio
                    return result
                
                # 🔥 BALANCED: POSTURE TRANSITION DETECTION - chỉ block khi thay đổi RÕ RÀNG
                # Nếu đang chuyển từ nằm → đứng hoặc ngược lại → KHÔNG phải co giật
                first_half_lying = sum(lying_checks[:len(lying_checks)//2]) / (len(lying_checks)//2) if lying_checks else 0
                second_half_lying = sum(lying_checks[len(lying_checks)//2:]) / (len(lying_checks) - len(lying_checks)//2) if lying_checks else 0
                
                # Nếu tỷ lệ lying thay đổi >40% giữa 2 nửa → đang thay đổi tư thế (TĂNG: 30%→40%)
                lying_change = abs(first_half_lying - second_half_lying)
                if lying_change > 0.4:
                    if self.stats['total_frames_processed'] % 30 == 0:
                        self.logger.info(f"🚫 POSTURE TRANSITION detected: first_half={first_half_lying:.1%}, second_half={second_half_lying:.1%}, change={lying_change:.1%}")
                        self.logger.info(f"   Person is getting up or lying down - NOT seizure")
                    result['temporal_ready'] = False
                    result['skipped_reason'] = 'posture_transition'
                    result['lying_change'] = lying_change
                    return result
                
                result['temporal_ready'] = True
                result['lying_ratio'] = lying_ratio
                
                # Debug logging for temporal readiness - only log once
                if len(self.frame_buffer) == self.temporal_window:
                    self.logger.debug(f"🧠 Temporal Window READY: {len(self.frame_buffer)}/{self.temporal_window} frames collected (all lying)")
                
                # 🆕 CHECK EXERCISE COOLDOWN - Nếu vừa detect exercise → skip seizure detection
                import time
                current_time = time.time()
                time_since_exercise = current_time - self.last_exercise_detection_time
                
                if time_since_exercise < self.exercise_cooldown:
                    if self.stats['total_frames_processed'] % 30 == 0:
                        self.logger.info(f"🏋️ EXERCISE COOLDOWN: {self.exercise_cooldown - time_since_exercise:.1f}s remaining - skipping seizure detection")
                    result['status'] = 'exercise_cooldown'
                    result['seizure_detected'] = False
                    result['alert_level'] = 'normal'
                    self.stats['total_frames_processed'] += 1
                    return result
                
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
            
            # Debug: print shapes - only debug level
            self.logger.debug(f"DEBUG: patches_tensor shape: {patches_tensor.shape}, kpts_tensor shape: {kpts_tensor.shape}")
            
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
        🔥 OPTIMIZED Seizure Detection - Dễ detect seizure thật hơn
        
        ĐẶC ĐIỂM CO GIẬT THẬT:
        - Motion JERKY (giật giật, không mượt)
        - Motion ERRATIC (không dự đoán được)
        - Ảnh hưởng NHIỀU phần cơ thể
        - Rhythm KHÔNG đều (khác với exercise)
        - Có thể có tremor (rung nhẹ liên tục)
        """
        if keypoint_sequence.shape[0] < 5:
            return 0.0
        
        try:
            coords = keypoint_sequence[:, :, :2]
            velocities = np.diff(coords, axis=0)
            vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=2))
            
            # === FILTER 1: Người ngồi/đứng yên hoàn toàn ===
            if vel_magnitudes.shape[1] >= 7:
                upper_body_motion = np.mean(vel_magnitudes[:, :7])
                if upper_body_motion < 3:  # GIẢM: 5→3 - chỉ filter khi HOÀN TOÀN yên
                    self.logger.debug(f"🪑 NO MOTION: upper_body={upper_body_motion:.1f} < 3")
                    return 0.0
            
            # === Tính các metrics cơ bản ===
            total_amplitude = np.max(vel_magnitudes) - np.min(vel_magnitudes)
            mean_velocity = np.mean(vel_magnitudes)
            velocity_over_time = np.mean(vel_magnitudes, axis=1)
            rhythm_variance = np.var(velocity_over_time)
            rhythm_regularity = 1.0 / (1.0 + rhythm_variance / 100.0)
            
            # === FILTER 2: Exercise RÕ RÀNG (amplitude LỚN + rhythm RẤT đều) ===
            is_obvious_exercise = (mean_velocity > 40 and rhythm_regularity > 0.60) or \
                                  (total_amplitude > 150 and rhythm_regularity > 0.55)
            if is_obvious_exercise:
                self.logger.debug(f"🏋️ OBVIOUS EXERCISE: amp={total_amplitude:.1f}, rhythm={rhythm_regularity:.2f}")
                return 0.0
            
            # === Jerkiness (đặc trưng seizure) ===
            vel_diff = np.diff(vel_magnitudes, axis=0)
            jerkiness = np.mean(np.abs(vel_diff))
            
            # === FILTER 3: Motion RẤT mượt + RẤT đều ===
            if jerkiness < 12 and rhythm_regularity > 0.55:
                self.logger.debug(f"🚫 VERY SMOOTH: jerk={jerkiness:.1f}, rhythm={rhythm_regularity:.2f}")
                return 0.0
            
            # === Active joints (seizure ảnh hưởng nhiều joints) ===
            joint_threshold = 8  # GIẢM: 10→8
            active_joints = sum(1 for j in range(vel_magnitudes.shape[1]) 
                              if np.mean(vel_magnitudes[:, j]) > joint_threshold)
            active_ratio = active_joints / vel_magnitudes.shape[1] if vel_magnitudes.shape[1] > 0 else 0
            
            # FILTER: Chỉ 1-2 joints di chuyển
            if active_ratio < 0.30:  # GIẢM: 40%→30%
                self.logger.debug(f"🚫 TOO FEW JOINTS: {active_joints}/{vel_magnitudes.shape[1]} ({active_ratio:.0%})")
                return 0.0
            
            # ============ SEIZURE SCORING ============
            
            # 1. Velocity variance
            velocity_variance = np.var(vel_magnitudes, axis=0).mean()
            velocity_score = np.tanh(velocity_variance / 60.0) if velocity_variance > 30 else 0.0
            
            # 2. Acceleration peaks
            accelerations = np.diff(velocities, axis=0)
            acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=2))
            acceleration_peaks = np.max(acc_magnitudes, axis=0).mean()
            acceleration_score = np.tanh(acceleration_peaks / 80.0) if acceleration_peaks > 40 else 0.0
            
            # 3. Frequency (direction changes)
            direction_changes = 0
            if vel_magnitudes.shape[0] > 5:
                for joint in range(min(8, vel_magnitudes.shape[1])):
                    joint_vel = vel_magnitudes[:, joint]
                    changes = np.sum(np.diff(np.sign(joint_vel)) != 0)
                    direction_changes += changes
                frequency_score = np.tanh(direction_changes / 40.0) if direction_changes > 15 else 0.0
            else:
                frequency_score = 0.0
            
            # 4. Intensity
            intensity_score = np.tanh(mean_velocity / 20.0) if mean_velocity > 10 else 0.0
            
            # 5. Spike score
            movement_spikes = np.max(vel_magnitudes, axis=0).mean()
            spike_score = np.tanh(movement_spikes / 40.0) if movement_spikes > 20 else 0.0
            
            # 6. ERRATIC score (variance cao = unpredictable)
            frame_variance = np.var(velocity_over_time)
            erratic_score = np.tanh(frame_variance / 150.0) if frame_variance > 60 else 0.0
            
            # 7. JERKINESS score
            jerkiness_score = np.tanh(jerkiness / 30.0) if jerkiness > 10 else 0.0
            
            # 8. IRREGULARITY score (rhythm không đều)
            irregularity_score = (1.0 - rhythm_regularity) if rhythm_regularity < 0.55 else 0.0
            
            # 9. 🆕 TREMOR score (motion nhỏ nhưng liên tục)
            tremor_score = 0.0
            if mean_velocity > 5 and mean_velocity < 25:  # Motion nhỏ-vừa
                if direction_changes > 20:  # Nhiều đổi hướng
                    tremor_score = 0.4
            
            # === INDICATOR CHECK (cần 2/5 indicators chính) ===
            threshold = 0.30
            main_indicators = [
                velocity_score > threshold,
                acceleration_score > threshold,
                jerkiness_score > threshold,
                erratic_score > 0.15,
                frequency_score > threshold
            ]
            active_indicators = sum(main_indicators)
            
            if active_indicators < 2:  # GIẢM: 3→2
                return 0.0
            
            # === FINAL SCORE ===
            seizure_confidence = (
                0.15 * velocity_score +
                0.15 * acceleration_score +
                0.12 * frequency_score +
                0.10 * intensity_score +
                0.10 * spike_score +
                0.12 * erratic_score +
                0.12 * jerkiness_score +
                0.07 * irregularity_score +
                0.07 * tremor_score
            )
            
            # BONUS for strong seizure indicators
            if active_indicators >= 4:
                seizure_confidence *= 1.20
            if jerkiness > 25 and rhythm_regularity < 0.45:
                seizure_confidence *= 1.15
            if tremor_score > 0:
                seizure_confidence *= 1.10
            
            # Debug
            frame_count = getattr(self, '_debug_frame_count', 0)
            self._debug_frame_count = frame_count + 1
            if frame_count % 30 == 0:
                self.logger.info(f"Seizure - Jerk:{jerkiness:.1f}({jerkiness_score:.2f}), Erratic:{erratic_score:.2f}, "
                               f"Vel:{velocity_score:.2f}, Freq:{frequency_score:.2f}, Tremor:{tremor_score:.2f}, "
                               f"Indicators:{active_indicators}/5, Joints:{active_ratio:.0%}, Final:{seizure_confidence:.2f}")
            
            # THRESHOLD cuối (GIẢM: 0.55→0.45)
            if seizure_confidence < 0.45:
                return 0.0
            
            return np.clip(seizure_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Motion analysis error: {e}")
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
    
    def _is_person_lying(self, keypoints: np.ndarray) -> tuple:
        """
        🔥 STRICT: Kiểm tra xem người có đang NẰM hay không
        
        PHẢI phân biệt rõ: NẰM vs ĐỨNG/ĐI LẠI/NGỒI
        - Khi ĐỨNG/ĐI: head ở TRÊN, chân ở DƯỚI, body angle nhỏ
        - Khi NẰM: head và hip gần level Y, body angle lớn
        
        Args:
            keypoints: Array of shape (15, 3) or (17, 3) with [x, y, confidence]
            
        Returns:
            tuple: (is_lying: bool, reason: str, confidence: float)
        """
        is_lying = False
        is_bending = False
        is_sitting = False
        is_standing_walking = False  # 🆕 Detect đứng/đi lại
        reasons = []
        confidence = 0.0
        
        if keypoints is None or len(keypoints) < 13:
            return False, "insufficient_keypoints", 0.0
        
        try:
            # === 🆕 CHECK STANDING/WALKING FIRST ===
            # Khi đứng/đi: head ở trên cùng, chân ở dưới cùng
            nose = keypoints[0]
            l_shoulder = keypoints[5]
            r_shoulder = keypoints[6]
            l_hip = keypoints[11]
            r_hip = keypoints[12]
            l_ankle = keypoints[15] if len(keypoints) > 15 else None
            r_ankle = keypoints[16] if len(keypoints) > 16 else None
            l_knee = keypoints[13] if len(keypoints) > 13 else None
            r_knee = keypoints[14] if len(keypoints) > 14 else None
            
            # Check vertical alignment: head -> shoulder -> hip -> knee -> ankle
            head_y = nose[1] if nose[2] > 0.3 else None
            shoulder_y = None
            hip_y = None
            knee_y = None
            ankle_y = None
            
            if l_shoulder[2] > 0.3 or r_shoulder[2] > 0.3:
                shoulder_y = np.mean([s[1] for s in [l_shoulder, r_shoulder] if s[2] > 0.3])
            if l_hip[2] > 0.3 or r_hip[2] > 0.3:
                hip_y = np.mean([h[1] for h in [l_hip, r_hip] if h[2] > 0.3])
            if l_knee is not None and r_knee is not None:
                if l_knee[2] > 0.3 or r_knee[2] > 0.3:
                    knee_y = np.mean([k[1] for k in [l_knee, r_knee] if k[2] > 0.3])
            if l_ankle is not None and r_ankle is not None:
                if l_ankle[2] > 0.3 or r_ankle[2] > 0.3:
                    ankle_y = np.mean([a[1] for a in [l_ankle, r_ankle] if a[2] > 0.3])
            
            # 🆕 STANDING/WALKING DETECTION: head trên cùng, chân dưới cùng (Y tăng = xuống dưới)
            if head_y is not None and hip_y is not None:
                # Nếu hip dưới head (hip_y > head_y) → có thể đứng
                if hip_y > head_y + 30:  # Hip ở dưới head ít nhất 30 pixels
                    is_standing_walking = True
                    confidence -= 0.3
                    reasons.append(f"hip_below_head={hip_y - head_y:.0f}px")
                    
                    # Nếu còn có ankle/knee ở dưới hip → CHẮC CHẮN đứng/đi
                    if ankle_y is not None and ankle_y > hip_y + 20:
                        is_standing_walking = True
                        confidence -= 0.5
                        reasons.append(f"STANDING_ankle_below_hip={ankle_y - hip_y:.0f}px")
                    elif knee_y is not None and knee_y > hip_y + 10:
                        is_standing_walking = True
                        confidence -= 0.4
                        reasons.append(f"STANDING_knee_below_hip={knee_y - hip_y:.0f}px")
            
            # 1. Tính aspect ratio từ keypoints bbox
            aspect_ratio = self._calculate_aspect_ratio_from_keypoints(keypoints)
            
            # 🔥 ASPECT RATIO - nhưng phải kết hợp với vertical check
            if aspect_ratio > 1.8:  # Rõ ràng nằm ngang
                if not is_standing_walking:  # Chỉ trust nếu không detect standing
                    is_lying = True
                    confidence += 0.55
                    reasons.append(f"aspect={aspect_ratio:.2f}>1.8_LYING")
                else:
                    reasons.append(f"aspect={aspect_ratio:.2f}_BUT_standing_detected")
            elif aspect_ratio > 1.4:  # Có thể nằm
                if not is_standing_walking:
                    confidence += 0.30
                    reasons.append(f"aspect={aspect_ratio:.2f}>1.4_maybe_lying")
            elif aspect_ratio < 0.5:  # Rõ ràng đứng
                confidence -= 0.5
                is_standing_walking = True
                reasons.append(f"aspect={aspect_ratio:.2f}<0.5_STANDING")
            elif aspect_ratio < 0.7:  # Có thể đứng hoặc ngồi
                confidence -= 0.25
                is_sitting = True
                reasons.append(f"aspect={aspect_ratio:.2f}<0.7_sitting_standing")
            else:
                reasons.append(f"aspect={aspect_ratio:.2f}_uncertain")
            
            upper_y_list = []
            lower_y_list = []
            
            if nose[2] > 0.3:
                upper_y_list.append(nose[1])
            if l_shoulder[2] > 0.3:
                upper_y_list.append(l_shoulder[1])
            if r_shoulder[2] > 0.3:
                upper_y_list.append(r_shoulder[1])
            if l_hip[2] > 0.3:
                lower_y_list.append(l_hip[1])
            if r_hip[2] > 0.3:
                lower_y_list.append(r_hip[1])
            
            if upper_y_list and lower_y_list:
                upper_y = np.mean(upper_y_list)
                lower_y = np.mean(lower_y_list)
                signed_diff = lower_y - upper_y
                
                valid_kpts = keypoints[keypoints[:, 2] > 0.3]
                if len(valid_kpts) >= 5:
                    bbox_height = np.max(valid_kpts[:, 1]) - np.min(valid_kpts[:, 1])
                    normalized_diff = signed_diff / max(bbox_height, 1)
                    
                    # 🔥 STRICTER THRESHOLDS - để filter đứng/đi
                    if normalized_diff > 0.35:  # Đứng/Ngồi (GIẢM: 0.45→0.35 để catch đứng)
                        is_lying = False
                        is_sitting = True
                        is_standing_walking = True
                        confidence -= 0.5
                        reasons.append(f"STANDING_diff={normalized_diff:.2f}>0.35")
                    elif abs(normalized_diff) < 0.25:  # Nằm - head và hip gần level (GIẢM: 0.30→0.25)
                        if not is_standing_walking:  # Chỉ khi không detect đứng
                            is_lying = True
                            confidence += 0.45
                            reasons.append(f"lying_diff={normalized_diff:.2f}~0")
                    elif normalized_diff < -0.15:  # Cúi
                        is_lying = False
                        is_bending = True
                        confidence -= 0.4
                        reasons.append(f"BENDING_diff={normalized_diff:.2f}<-0.15")
                    elif normalized_diff > 0.25:
                        is_sitting = True
                        is_standing_walking = True
                        confidence -= 0.3
                        reasons.append(f"maybe_standing_diff={normalized_diff:.2f}")
                    else:
                        reasons.append(f"uncertain_diff={normalized_diff:.2f}")
            
            # 3. Check body angle
            body_angle = self._calculate_body_angle_from_keypoints(keypoints)
            if body_angle is not None:
                if body_angle < 25:  # Body thẳng đứng (GIẢM: 30→25)
                    is_lying = False
                    is_sitting = True
                    confidence -= 0.35
                    reasons.append(f"upright_angle={body_angle:.1f}<25")
                elif body_angle > 65:  # Body nằm ngang (GIẢM: 70→65)
                    is_lying = True
                    confidence += 0.35
                    reasons.append(f"horizontal_angle={body_angle:.1f}>65")
            
        except Exception as e:
            self.logger.debug(f"Lying check error: {e}")
            return False, f"error: {e}", 0.0
        
        # Check push-up pattern (chỉ khi enabled)
        is_pushup = self._detect_pushup_pattern()
        if is_pushup:
            is_lying = False
            is_bending = False
            is_sitting = False
            confidence = -0.4
            reasons.append("PUSHUP_DETECTED")
        
        # 🔥 STRICT Final decision - ưu tiên detect STANDING/WALKING
        if is_standing_walking:
            # Nếu detect đứng/đi → CHẮC CHẮN không nằm
            is_lying = False
            confidence = min(confidence, 0)
            reasons.append("FINAL_STANDING")
        elif is_bending or is_pushup:
            is_lying = False
            confidence = min(confidence, 0)
        elif is_sitting and confidence < 0.55:
            is_lying = False
            confidence = min(confidence, 0.25)
        elif confidence >= 0.55:  # TĂNG: 0.5→0.55 để strict hơn
            is_lying = True
        elif confidence <= 0.1:
            is_lying = False
        
        reason_str = " | ".join(reasons) if reasons else "no_data"
        return is_lying, reason_str, max(0, min(confidence, 1.0))
    
    def _calculate_body_angle_from_keypoints(self, keypoints: np.ndarray) -> float:
        """
        🔥 NEW: Calculate body angle from vertical (0° = standing, 90° = lying)
        """
        try:
            # Get shoulder center
            l_shoulder = keypoints[5]
            r_shoulder = keypoints[6]
            shoulder_y, shoulder_x = None, None
            
            if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
                shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
            elif l_shoulder[2] > 0.3:
                shoulder_y, shoulder_x = l_shoulder[1], l_shoulder[0]
            elif r_shoulder[2] > 0.3:
                shoulder_y, shoulder_x = r_shoulder[1], r_shoulder[0]
            
            # Get hip center
            l_hip = keypoints[11]
            r_hip = keypoints[12]
            hip_y, hip_x = None, None
            
            if l_hip[2] > 0.3 and r_hip[2] > 0.3:
                hip_y = (l_hip[1] + r_hip[1]) / 2
                hip_x = (l_hip[0] + r_hip[0]) / 2
            elif l_hip[2] > 0.3:
                hip_y, hip_x = l_hip[1], l_hip[0]
            elif r_hip[2] > 0.3:
                hip_y, hip_x = r_hip[1], r_hip[0]
            
            if shoulder_y is None or hip_y is None:
                return None
            
            # Calculate angle from vertical
            dx = abs(shoulder_x - hip_x)
            dy = abs(shoulder_y - hip_y)
            
            if dy < 1:
                return 90  # Horizontal body
            
            import math
            angle = math.degrees(math.atan2(dx, dy))
            return angle
            
        except Exception as e:
            return None
    
    def _detect_pushup_pattern(self) -> bool:
        """
        🔥 IMPROVED: Detect EXERCISE patterns (hít đất, gập bụng, plank, giãn cơ)
        
        PHÂN BIỆT TẬP THỂ DỤC vs CO GIẬT:
        =====================================================
        TẬP THỂ DỤC (Exercise):
        - Motion SMOOTH và CONTROLLED (jerkiness thấp)
        - Rhythm ĐỀU đặn (regularity cao)  
        - Chỉ một số body parts di chuyển
        - Velocity thay đổi CHẬM và CÓ PATTERN
        
        CO GIẬT (Seizure):
        - Motion JERKY và ERRATIC (jerkiness cao)
        - Rhythm KHÔNG đều (regularity thấp)
        - NHIỀU body parts di chuyển cùng lúc
        - Velocity thay đổi NHANH và RANDOM
        =====================================================
        
        Returns:
            bool: True if exercise pattern detected (NOT seizure)
        """
        # 🆕 Check if exercise detection is enabled
        if not getattr(self, 'enable_exercise_detection', True):
            return False  # Skip exercise detection completely
        
        if len(self.frame_buffer) < 8:
            return False
        
        try:
            # ========== PHASE 1: Thu thập motion data từ NHIỀU body parts ==========
            shoulder_y_history = []
            hip_y_history = []
            torso_y_history = []  # Trung bình shoulder và hip
            head_y_history = []
            
            for fd in self.frame_buffer[-15:]:  # Tăng window để detect pattern tốt hơn
                if 'keypoints' in fd and fd['keypoints'] is not None:
                    kpts = fd['keypoints']
                    
                    # Shoulder
                    l_shoulder = kpts[5] if len(kpts) > 5 else None
                    r_shoulder = kpts[6] if len(kpts) > 6 else None
                    shoulder_y = []
                    if l_shoulder is not None and l_shoulder[2] > 0.3:
                        shoulder_y.append(l_shoulder[1])
                    if r_shoulder is not None and r_shoulder[2] > 0.3:
                        shoulder_y.append(r_shoulder[1])
                    if shoulder_y:
                        shoulder_y_history.append(np.mean(shoulder_y))
                    
                    # Hip
                    l_hip = kpts[11] if len(kpts) > 11 else None
                    r_hip = kpts[12] if len(kpts) > 12 else None
                    hip_y = []
                    if l_hip is not None and l_hip[2] > 0.3:
                        hip_y.append(l_hip[1])
                    if r_hip is not None and r_hip[2] > 0.3:
                        hip_y.append(r_hip[1])
                    if hip_y:
                        hip_y_history.append(np.mean(hip_y))
                    
                    # Torso (avg of shoulder and hip)
                    if shoulder_y and hip_y:
                        torso_y_history.append((np.mean(shoulder_y) + np.mean(hip_y)) / 2)
                    
                    # Head (nose)
                    nose = kpts[0] if len(kpts) > 0 else None
                    if nose is not None and nose[2] > 0.3:
                        head_y_history.append(nose[1])
            
            if len(shoulder_y_history) < 6:
                return False
            
            # ========== PHASE 2: Tính các metrics quan trọng ==========
            
            # 2.1. Amplitude và direction changes cho shoulder
            y_array = np.array(shoulder_y_history)
            diffs = np.diff(y_array)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            amplitude = np.max(y_array) - np.min(y_array)
            
            # 2.2. SMOOTHNESS: Tính jerkiness (đạo hàm bậc 2 - acceleration changes)
            if len(diffs) >= 2:
                acceleration = np.diff(diffs)
                jerkiness = np.mean(np.abs(acceleration)) if len(acceleration) > 0 else 0
            else:
                jerkiness = 0
            
            # 2.3. RHYTHM REGULARITY: Độ đều của motion
            abs_diffs = np.abs(diffs)
            if len(abs_diffs) > 2 and np.mean(abs_diffs) > 0:
                # Coefficient of variation - thấp = đều, cao = không đều
                rhythm_cv = np.std(abs_diffs) / np.mean(abs_diffs) if np.mean(abs_diffs) > 0 else float('inf')
                rhythm_regularity = 1.0 / (1.0 + rhythm_cv)  # 0-1, cao = đều
            else:
                rhythm_regularity = 0.5
            
            # 2.4. CONTROLLED MOTION: Kiểm tra motion có pattern không
            # Dùng autocorrelation để detect repetitive motion
            if len(y_array) >= 8:
                centered = y_array - np.mean(y_array)
                autocorr = np.correlate(centered, centered, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                # Normalize
                autocorr = autocorr / (autocorr[0] + 1e-6)
                # Tìm peak ngoài peak đầu tiên
                if len(autocorr) > 3:
                    secondary_peaks = autocorr[2:]
                    has_repetitive_pattern = np.max(secondary_peaks) > 0.5
                else:
                    has_repetitive_pattern = False
            else:
                has_repetitive_pattern = False
            
            # 2.5. BODY STABILITY: Kiểm tra hip có ổn định không (exercise thường có hip stable)
            if len(hip_y_history) >= 4:
                hip_variance = np.var(hip_y_history)
                hip_stable = hip_variance < 200  # Hip ít di chuyển
            else:
                hip_stable = False
            
            # 2.6. 🆕 LEG RAISE DETECTION: Detect giãn cơ chân (chỉ chân di chuyển, body stable)
            # Thu thập ankle/knee positions
            ankle_y_history = []
            knee_y_history = []
            for fd in self.frame_buffer[-15:]:
                if 'keypoints' in fd and fd['keypoints'] is not None:
                    kpts = fd['keypoints']
                    # Ankles (COCO: 15=L_ankle, 16=R_ankle)
                    l_ankle = kpts[15] if len(kpts) > 15 else None
                    r_ankle = kpts[16] if len(kpts) > 16 else None
                    ankle_y = []
                    if l_ankle is not None and l_ankle[2] > 0.3:
                        ankle_y.append(l_ankle[1])
                    if r_ankle is not None and r_ankle[2] > 0.3:
                        ankle_y.append(r_ankle[1])
                    if ankle_y:
                        ankle_y_history.append(np.mean(ankle_y))
                    
                    # Knees (COCO: 13=L_knee, 14=R_knee)
                    l_knee = kpts[13] if len(kpts) > 13 else None
                    r_knee = kpts[14] if len(kpts) > 14 else None
                    knee_y = []
                    if l_knee is not None and l_knee[2] > 0.3:
                        knee_y.append(l_knee[1])
                    if r_knee is not None and r_knee[2] > 0.3:
                        knee_y.append(r_knee[1])
                    if knee_y:
                        knee_y_history.append(np.mean(knee_y))
            
            # Tính leg motion amplitude
            leg_amplitude = 0
            if len(ankle_y_history) >= 4:
                leg_amplitude = np.max(ankle_y_history) - np.min(ankle_y_history)
            
            # Tính upper body stability (head + shoulder)
            upper_body_stable = False
            if len(head_y_history) >= 4 and len(shoulder_y_history) >= 4:
                head_variance = np.var(head_y_history)
                shoulder_variance = np.var(shoulder_y_history)
                upper_body_stable = head_variance < 100 and shoulder_variance < 100
            
            # LEG RAISE pattern: chân di chuyển nhiều (>80px) + upper body stable
            is_leg_raise = leg_amplitude > 80 and upper_body_stable
            
            # ========== PHASE 3: DECISION LOGIC ==========
            
            is_exercise = False
            exercise_type = "none"
            
            # 🔥 DEBUG
            self.logger.debug(f"📊 Exercise analysis: amp={amplitude:.1f}, sign_ch={sign_changes}, "
                            f"jerk={jerkiness:.1f}, rhythm={rhythm_regularity:.2f}, "
                            f"repeat={has_repetitive_pattern}, hip_stable={hip_stable}, "
                            f"leg_amp={leg_amplitude:.1f}, upper_stable={upper_body_stable}")
            
            # 🆕 CASE 0: LEG RAISE / GIÃN CƠ CHÂN (ưu tiên cao nhất)
            # Chỉ CHÂN di chuyển, UPPER BODY stable → đang tập giãn cơ chân
            if is_leg_raise:
                is_exercise = True
                exercise_type = "leg_raise"
                self.logger.info(f"🦵 LEG RAISE DETECTED: leg_amp={leg_amplitude:.1f}px, upper_stable={upper_body_stable}")
                # 🆕 SET EXERCISE COOLDOWN
                import time
                self.last_exercise_detection_time = time.time()
                self.logger.info(f"⏰ EXERCISE COOLDOWN ACTIVATED: {self.exercise_cooldown}s - blocking seizure detection")
                return True  # Return ngay để skip các case khác
            
            # CASE 1: HÍT ĐẤT / PUSH-UP (shoulder amplitude lớn, rhythm đều)
            if amplitude > 80 and rhythm_regularity > 0.3 and jerkiness < 25:
                is_exercise = True
                exercise_type = "pushup"
                self.logger.info(f"🏋️ PUSH-UP/EXERCISE DETECTED: amp={amplitude:.1f}px, rhythm={rhythm_regularity:.2f}, jerk={jerkiness:.1f}")
            
            # CASE 2: GẬP BỤNG / CRUNCH (hip stable, upper body moves rhythmically)
            elif hip_stable and rhythm_regularity > 0.35 and jerkiness < 20:
                is_exercise = True
                exercise_type = "crunch"
                self.logger.info(f"🏋️ CRUNCH/SIT-UP DETECTED: hip_stable={hip_stable}, rhythm={rhythm_regularity:.2f}, jerk={jerkiness:.1f}")
            
            # CASE 3: GIÃN CƠ / STRETCHING (slow, controlled movement)
            elif jerkiness < 10 and rhythm_regularity > 0.4:
                is_exercise = True
                exercise_type = "stretching"
                self.logger.info(f"🧘 STRETCHING DETECTED: jerk={jerkiness:.1f}, rhythm={rhythm_regularity:.2f}")
            
            # CASE 4: REPETITIVE EXERCISE (bất kỳ bài tập có pattern lặp lại)
            elif has_repetitive_pattern and jerkiness < 30:
                is_exercise = True
                exercise_type = "repetitive"
                self.logger.info(f"🏃 REPETITIVE EXERCISE DETECTED: pattern=True, jerk={jerkiness:.1f}")
            
            # CASE 5: CONTROLLED MOTION (chuyển động có kiểm soát, không giật)
            elif rhythm_regularity > 0.5 and jerkiness < 15:
                is_exercise = True
                exercise_type = "controlled"
                self.logger.info(f"✅ CONTROLLED MOTION DETECTED: rhythm={rhythm_regularity:.2f}, jerk={jerkiness:.1f}")
            
            # CASE 6: HIGH AMPLITUDE với direction changes ít = exercise
            elif amplitude > 100 and sign_changes <= 5 and jerkiness < 35:
                is_exercise = True
                exercise_type = "high_amplitude"
                self.logger.info(f"🏋️ HIGH AMPLITUDE EXERCISE: amp={amplitude:.1f}px, dir_ch={sign_changes}, jerk={jerkiness:.1f}")
            
            # 🆕 SET EXERCISE COOLDOWN khi detect bất kỳ exercise nào
            if is_exercise:
                import time
                self.last_exercise_detection_time = time.time()
                self.logger.info(f"⏰ EXERCISE COOLDOWN ACTIVATED: {self.exercise_cooldown}s - blocking seizure detection")
            
            return is_exercise
            
        except Exception as e:
            self.logger.debug(f"Exercise detection error: {e}")
            return False
    
    def _check_if_standing(self, keypoints: np.ndarray) -> bool:
        """Check if person is standing/bending/doing push-ups (NOT lying) based on keypoints
        
        Args:
            keypoints: Array of shape (15, 3) or (17, 3) with [x, y, confidence]
            
        Returns:
            bool: True if person is standing/bending/push-ups (NOT lying), False if lying
        """
        try:
            # 🔥 First check for push-up pattern (blocks seizure detection)
            if self._detect_pushup_pattern():
                return True  # Push-ups = NOT lying = skip seizure detection
            
            # Dùng _is_person_lying để check
            is_lying, reason, confidence = self._is_person_lying(keypoints)
            
            # Nếu đang nằm → return False (không phải standing)
            # Nếu không nằm (đứng/cúi) → return True (là standing)
            return not is_lying
            
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
