#!/usr/bin/env python3
"""
PRODUCTION SYSTEM TEST WITH VIDEO
Test toàn bộ hệ thống (Fall + Seizure Detection) sử dụng video thay vì camera
Giữ nguyên 100% logic production, chỉ thay thế nguồn input

Usage:
    python test_production_with_video.py --video 1
    python test_production_with_video.py --video 7 --show
    python test_production_with_video.py --video 1 --speed 0.5  # Slow motion
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import argparse

# Add src to path - CRITICAL for importing production modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import video camera service (local)
from video_camera_service import VideoCameraService

# ============================================================================
# IMPORT PRODUCTION MODULES - EXACT SAME AS MAIN.PY
# ============================================================================

try:
    from fall_detection.simple_fall_detector import SimpleFallDetector
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor
    from ultralytics import YOLO
    PRODUCTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Production modules not available: {e}")
    PRODUCTION_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionVideoTest:
    """
    Test production system với video input
    Sử dụng ĐÚNG logic như main.py nhưng thay camera bằng video
    """
    
    def __init__(self, video_input: str = "1", show_display: bool = True, speed: float = 1.0,
                 no_cooldown: bool = False, skip_frames: int = 1, fast_mode: bool = False):
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        
        # Find video (support both number and name)
        self.video_path = self.find_video(video_input)
        self.video_name = self.video_path.stem
        self.video_input = video_input
        
        # Settings
        self.show_display = show_display
        self.speed = speed  # 1.0 = normal, 0.5 = slow, 2.0 = fast
        self.no_cooldown = no_cooldown  # Disable all cooldowns for testing
        self.skip_frames = max(1, skip_frames)  # Process every Nth frame (1=all, 2=every 2nd, etc)
        self.fast_mode = fast_mode  # 🆕 Disable keypoints visualization for SPEED
        self.preview_mode = False  # 🆕 PREVIEW MODE: Just watch video, no processing
        
        # Output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.script_dir / "test_results" / f"production_video_{self.video_name}_{timestamp}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            # Fall detection
            'fall_detections': 0,
            'fall_alerts': [],
            'max_fall_confidence': 0.0,
            # Seizure detection
            'seizure_detections': 0,
            'seizure_alerts': [],
            'max_seizure_confidence': 0.0,
            # Normal
            'normal_frames': 0,
        }
        
        # Production components (will be initialized)
        self.person_detector = None
        self.fall_detector = None
        self.seizure_detector = None
        self.seizure_predictor = None
        self.pose_estimator = None  # 🆕 For keypoint visualization
        
        # Detection state (same as production)
        self.last_fall_time = None
        self.last_seizure_time = None
        
        # Cooldowns - có thể tắt để test
        if self.no_cooldown:
            self.fall_cooldown = 0.0
            self.seizure_cooldown = 0.0
            self.global_cooldown = 0.0
            print("⚠️ ALL COOLDOWNS DISABLED FOR TESTING")
        else:
            self.fall_cooldown = 10.0  # Same as production
            self.seizure_cooldown = 30.0  # Same as production
            self.global_cooldown = 45.0  # Same as production
        
        self.last_any_event_time = 0
        self.active_event_type = None
        
        # Frame buffer for temporal analysis
        self.frame_buffer = []
        self.buffer_size = 3  # Store last 3 frames
    
    def find_video(self, video_input: str) -> Path:
        """Tìm video theo số hoặc tên"""
        # Try as number first
        video_path_lower = self.resource_folder / f"{video_input}.mp4"
        video_path_upper = self.resource_folder / f"{video_input}.MP4"
        
        if video_path_lower.exists():
            return video_path_lower
        elif video_path_upper.exists():
            return video_path_upper
        else:
            # List available videos
            available = list(self.resource_folder.glob("*.mp4")) + list(self.resource_folder.glob("*.MP4"))
            available_names = [p.stem for p in available]
            raise FileNotFoundError(
                f"Video '{video_input}' not found in {self.resource_folder}\n"
                f"Available videos: {sorted(available_names)}"
            )
    
    def initialize_production_components(self) -> bool:
        """Initialize production components - SAME AS MAIN.PY"""
        try:
            print("\n" + "="*80)
            print("🔧 INITIALIZING PRODUCTION COMPONENTS...")
            print("="*80)
            
            if not PRODUCTION_AVAILABLE:
                print("❌ Production modules not available")
                return False
            
            # 1. YOLO Person Detector - Use yolov8s for better detection
            # yolov8n misses some frames which causes gaps in fall detection buffer
            print("📦 Loading YOLO person detector...")
            yolo_path = self.script_dir / "yolov8s.pt"  # Use 's' model for better detection
            if not yolo_path.exists():
                yolo_path = PROJECT_ROOT / "yolov8s.pt"
            if not yolo_path.exists():
                yolo_path = self.script_dir / "yolov8n.pt"  # Fallback to 'n' if 's' not available
            if not yolo_path.exists():
                yolo_path = PROJECT_ROOT / "yolov8n.pt"
            if not yolo_path.exists():
                yolo_path = Path("yolov8n.pt")
            
            self.person_detector = YOLO(str(yolo_path))
            print(f"✅ YOLO loaded from {yolo_path}")
            
            # 2. Fall Detector - PRODUCTION VERSION
            print("🚨 Loading fall detector...")
            self.fall_detector = SimpleFallDetector(
                confidence_threshold=0.28  # Same as production
            )
            print("✅ Fall detector loaded")
            
            # 3. Seizure Detector - PRODUCTION VERSION
            print("🧠 Loading VSViG seizure detector...")
            self.seizure_detector = VSViGSeizureDetector(
                confidence_threshold=0.65,  # Same as production
                device='auto'
            )
            if not self.seizure_detector.load_models():
                print("⚠️ VSViG models not loaded - fallback mode")
            else:
                print("✅ VSViG loaded")
            
            # 4. Seizure Predictor
            print("📊 Initializing seizure predictor...")
            self.seizure_predictor = SeizurePredictor(
                temporal_window=15,
                alert_threshold=0.70,
                warning_threshold=0.55
            )
            print("✅ Seizure predictor initialized")
            
            # 5. 🆕 Pose Estimator for keypoint visualization
            print("🦴 Loading pose estimator for keypoints...")
            pose_path = PROJECT_ROOT / "yolov8n-pose.pt"
            if not pose_path.exists():
                pose_path = Path("yolov8n-pose.pt")
            self.pose_estimator = YOLO(str(pose_path))
            print("✅ Pose estimator loaded")
            
            print("="*80 + "\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    def detect_persons(self, frame: np.ndarray) -> list:
        """Detect persons - SAME AS PRODUCTION"""
        results = self.person_detector(frame, conf=0.15, classes=[0], verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Calculate bbox in production format [x, y, w, h]
                w = x2 - x1
                h = y2 - y1
                
                persons.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        return persons
    
    def calculate_motion_level(self, frame: np.ndarray) -> float:
        """Calculate motion level - SAME AS PRODUCTION"""
        if len(self.frame_buffer) < 2:
            self.frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            return 0.0
        
        # Add current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compare with previous frame
        prev_gray = self.frame_buffer[-1]
        diff = cv2.absdiff(gray, prev_gray)
        motion_level = np.sum(diff > 25) / diff.size
        
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        return motion_level
    
    def check_cooldowns(self, current_time: float) -> dict:
        """Check all cooldowns - SAME AS PRODUCTION"""
        result = {
            'can_detect_fall': True,
            'can_detect_seizure': True,
            'global_blocked': False,
            'fall_cooldown_remaining': 0,
            'seizure_cooldown_remaining': 0,
            'global_cooldown_remaining': 0
        }
        
        # Global cooldown
        time_since_last_event = current_time - self.last_any_event_time
        if time_since_last_event < self.global_cooldown and self.active_event_type:
            result['global_blocked'] = True
            result['global_cooldown_remaining'] = self.global_cooldown - time_since_last_event
            result['can_detect_fall'] = False
            result['can_detect_seizure'] = False
        
        # Fall cooldown
        if self.last_fall_time:
            time_since_fall = current_time - self.last_fall_time
            if time_since_fall < self.fall_cooldown:
                result['can_detect_fall'] = False
                result['fall_cooldown_remaining'] = self.fall_cooldown - time_since_fall
        
        # Seizure cooldown
        if self.last_seizure_time:
            time_since_seizure = current_time - self.last_seizure_time
            if time_since_seizure < self.seizure_cooldown:
                result['can_detect_seizure'] = False
                result['seizure_cooldown_remaining'] = self.seizure_cooldown - time_since_seizure
        
        return result
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> dict:
        """
        Process single frame - USING PRODUCTION LOGIC
        Returns detection results
        """
        current_time = time.time()
        result = {
            'frame_number': frame_number,
            'persons': [],
            'fall_detected': False,
            'fall_confidence': 0.0,
            'seizure_detected': False,
            'seizure_confidence': 0.0,
            'event_type': 'normal',
            'motion_level': 0.0,
            'cooldown_status': {}
        }
        
        # Calculate motion
        motion_level = self.calculate_motion_level(frame)
        result['motion_level'] = motion_level
        
        # Detect persons
        persons = self.detect_persons(frame)
        result['persons'] = persons
        
        if len(persons) == 0:
            return result
        
        self.stats['frames_with_person'] += 1
        
        # Get primary person (largest)
        primary_person = max(persons, key=lambda x: x['bbox'][2] * x['bbox'][3])
        person_bbox = primary_person['bbox_xyxy']
        
        # Check cooldowns
        cooldown_status = self.check_cooldowns(current_time)
        result['cooldown_status'] = cooldown_status
        
        # ==================== FALL DETECTION (PRODUCTION LOGIC) ====================
        if cooldown_status['can_detect_fall']:
            try:
                # Debug: Log fall detection input
                if frame_number % 100 == 0:
                    print(f"🔍 Fall Debug: bbox={person_bbox}, motion={motion_level:.3f}")
                
                fall_result = self.fall_detector.detect_fall(
                    frame=frame,
                    timestamp=current_time,
                    person_bbox=person_bbox,
                    motion_level=motion_level
                )
                
                fall_confidence = fall_result.get('confidence', 0.0)
                fall_method = fall_result.get('method', 'unknown')
                
                # Debug: Log fall detection result
                if fall_confidence > 0.1 or (fall_method and fall_method != 'unknown' and fall_method != 'simplified'):
                    print(f"🔍 Fall Result: conf={fall_confidence:.3f}, method={fall_method}, motion={motion_level:.3f}")
                
                # Log why fall was rejected
                if fall_confidence > 0 and fall_confidence < 0.28:
                    print(f"⚠️ Fall REJECTED (low conf): conf={fall_confidence:.3f} < 0.28")
                elif fall_confidence >= 0.28 and motion_level <= 0.015 and fall_method not in ['rapid_downward', 'sideways_fall']:
                    print(f"⚠️ Fall REJECTED (low motion): conf={fall_confidence:.3f}, motion={motion_level:.3f} < 0.015")
                
                # PRODUCTION THRESHOLDS
                has_real_motion = motion_level > 0.015
                is_rapid_fall = fall_method == 'rapid_downward'
                is_sideways_fall = fall_method == 'sideways_fall'  # 🆕 Té ngang bypass motion filter
                
                if fall_confidence >= 0.28 and (has_real_motion or is_rapid_fall or is_sideways_fall):
                    result['fall_detected'] = True
                    result['fall_confidence'] = fall_confidence
                    result['fall_method'] = fall_method
                    result['event_type'] = 'fall'
                    
                    # Update cooldowns
                    self.last_fall_time = current_time
                    self.last_any_event_time = current_time
                    self.active_event_type = 'fall'
                    
                    self.stats['fall_detections'] += 1
                    self.stats['fall_alerts'].append({
                        'frame': frame_number,
                        'confidence': fall_confidence,
                        'method': fall_method,
                        'timestamp': current_time
                    })
                    
                    if fall_confidence > self.stats['max_fall_confidence']:
                        self.stats['max_fall_confidence'] = fall_confidence
                    
                    # Save alert frame
                    self.save_alert_frame(frame, frame_number, 'fall', fall_confidence)
                    
            except Exception as e:
                logger.debug(f"Fall detection error: {e}")
        
        # ==================== SEIZURE DETECTION (PRODUCTION LOGIC) ====================
        if cooldown_status['can_detect_seizure'] and not result['fall_detected']:
            try:
                seizure_result = self.seizure_detector.detect_seizure(frame, person_bbox)
                
                if seizure_result.get('temporal_ready', False):
                    raw_confidence = seizure_result.get('confidence', 0.0)
                    
                    # Use predictor for smoothing
                    prediction = self.seizure_predictor.update_prediction(raw_confidence)
                    smoothed_confidence = prediction['smoothed_confidence']
                    
                    result['seizure_raw_confidence'] = raw_confidence
                    result['seizure_confidence'] = smoothed_confidence
                    
                    # PRODUCTION THRESHOLDS
                    seizure_threshold = 0.60
                    min_motion = 0.008
                    
                    if smoothed_confidence >= seizure_threshold and motion_level >= min_motion:
                        result['seizure_detected'] = True
                        result['event_type'] = 'seizure'
                        
                        # Update cooldowns
                        self.last_seizure_time = current_time
                        self.last_any_event_time = current_time
                        self.active_event_type = 'seizure'
                        
                        self.stats['seizure_detections'] += 1
                        self.stats['seizure_alerts'].append({
                            'frame': frame_number,
                            'confidence': smoothed_confidence,
                            'raw_confidence': raw_confidence,
                            'timestamp': current_time
                        })
                        
                        if smoothed_confidence > self.stats['max_seizure_confidence']:
                            self.stats['max_seizure_confidence'] = smoothed_confidence
                        
                        # Save alert frame
                        self.save_alert_frame(frame, frame_number, 'seizure', smoothed_confidence)
                
            except Exception as e:
                logger.debug(f"Seizure detection error: {e}")
        
        # Update normal frames count
        if result['event_type'] == 'normal':
            self.stats['normal_frames'] += 1
        
        return result
    
    def save_alert_frame(self, frame: np.ndarray, frame_number: int, 
                        event_type: str, confidence: float):
        """Save alert frame to output folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{event_type}_frame_{frame_number}_{timestamp}_conf_{confidence:.2f}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), frame)
        logger.info(f"🚨 Alert saved: {filename}")
    
    def get_keypoints(self, frame: np.ndarray) -> list:
        """🆕 Get keypoints from pose estimator"""
        if self.pose_estimator is None:
            return []
        
        try:
            results = self.pose_estimator(frame, conf=0.3, verbose=False)
            keypoints_list = []
            
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints.data.cpu().numpy()
                    for person_kpts in kpts:
                        keypoints_list.append(person_kpts)
            
            return keypoints_list
        except Exception as e:
            logger.debug(f"Keypoint extraction error: {e}")
            return []
    
    def draw_keypoints(self, frame: np.ndarray, keypoints_list: list) -> np.ndarray:
        """🆕 Draw keypoints skeleton on frame"""
        display = frame.copy()
        
        # COCO keypoint connections (skeleton)
        SKELETON = [
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
        
        # Colors for different body parts
        COLORS = {
            'head': (255, 255, 0),      # Cyan - head
            'torso': (0, 255, 0),       # Green - torso
            'left_arm': (255, 0, 255),  # Magenta - left arm
            'right_arm': (0, 255, 255), # Yellow - right arm  
            'left_leg': (255, 128, 0),  # Orange - left leg
            'right_leg': (0, 128, 255), # Light blue - right leg
        }
        
        KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for keypoints in keypoints_list:
            if len(keypoints) < 17:
                continue
            
            # Draw skeleton connections
            for i, (start_idx, end_idx) in enumerate(SKELETON):
                if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                    continue
                    
                start_kpt = keypoints[start_idx]
                end_kpt = keypoints[end_idx]
                
                # Check confidence
                if start_kpt[2] < 0.3 or end_kpt[2] < 0.3:
                    continue
                
                start_pos = (int(start_kpt[0]), int(start_kpt[1]))
                end_pos = (int(end_kpt[0]), int(end_kpt[1]))
                
                # Color based on body part
                if i < 4:
                    color = COLORS['head']
                elif i == 4:
                    color = COLORS['torso']
                elif i < 7:
                    color = COLORS['left_arm']
                elif i < 9:
                    color = COLORS['right_arm']
                elif i < 12:
                    color = COLORS['torso']
                elif i < 14:
                    color = COLORS['left_leg']
                else:
                    color = COLORS['right_leg']
                
                cv2.line(display, start_pos, end_pos, color, 2)
            
            # Draw keypoints as circles
            for idx, kpt in enumerate(keypoints):
                if kpt[2] < 0.3:
                    continue
                
                x, y = int(kpt[0]), int(kpt[1])
                conf = kpt[2]
                
                # Color based on body part
                if idx < 5:
                    color = COLORS['head']
                elif idx < 11:
                    color = COLORS['torso']
                else:
                    color = COLORS['left_leg'] if idx % 2 == 1 else COLORS['right_leg']
                
                # Size based on confidence
                radius = int(4 + conf * 4)
                cv2.circle(display, (x, y), radius, color, -1)
                cv2.circle(display, (x, y), radius, (0, 0, 0), 1)
                
                # Draw keypoint name for important points
                if idx in [0, 5, 6, 11, 12, 15, 16]:  # nose, shoulders, hips, ankles
                    name = KEYPOINT_NAMES[idx][:3].upper()
                    cv2.putText(display, name, (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # 🆕 Draw aspect ratio info
            valid_kpts = keypoints[keypoints[:, 2] > 0.3]
            if len(valid_kpts) >= 5:
                min_x, min_y = np.min(valid_kpts[:, 0]), np.min(valid_kpts[:, 1])
                max_x, max_y = np.max(valid_kpts[:, 0]), np.max(valid_kpts[:, 1])
                w = max_x - min_x
                h = max_y - min_y
                aspect_ratio = w / h if h > 0 else 0
                
                # Draw bbox around keypoints
                cv2.rectangle(display, (int(min_x), int(min_y)), (int(max_x), int(max_y)), 
                             (255, 255, 255), 1)
                
                # Aspect ratio text
                aspect_text = f"AR:{aspect_ratio:.2f}"
                if aspect_ratio > 1.3:
                    aspect_color = (0, 0, 255)  # Red = lying
                    aspect_text += " LYING"
                elif aspect_ratio < 0.7:
                    aspect_color = (0, 255, 0)  # Green = standing
                    aspect_text += " STAND"
                else:
                    aspect_color = (255, 255, 0)  # Cyan = transition
                    aspect_text += " TRANS"
                
                cv2.putText(display, aspect_text, (int(min_x), int(min_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, aspect_color, 2)
        
        return display
    
    def draw_results(self, frame: np.ndarray, result: dict, keypoints_list: list = None) -> np.ndarray:
        """Draw detection results on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 🆕 Draw keypoints first (background layer)
        if keypoints_list:
            display_frame = self.draw_keypoints(display_frame, keypoints_list)
        
        # Draw person bounding boxes
        for person in result['persons']:
            x1, y1, x2, y2 = person['bbox_xyxy']
            
            # Color based on event type
            if result['event_type'] == 'fall':
                color = (0, 0, 255)  # Red
                thickness = 3
            elif result['event_type'] == 'seizure':
                color = (0, 165, 255)  # Orange
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Info panel (top-left) - TĂNG SIZE cho dễ đọc
        panel_width = 450
        panel_height = 500
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
        
        y = 40
        cv2.putText(display_frame, "PRODUCTION TEST", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        y += 45
        
        cv2.putText(display_frame, f"Frame: {result['frame_number']}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35
        cv2.putText(display_frame, f"Persons: {len(result['persons'])}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35
        cv2.putText(display_frame, f"Motion: {result['motion_level']:.3f}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 45
        
        # Fall detection status
        cv2.putText(display_frame, "FALL:", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        fall_conf = result.get('fall_confidence', 0.0)
        fall_color = (0, 0, 255) if result['fall_detected'] else (150, 150, 150)
        cv2.putText(display_frame, f"{fall_conf:.2f}", (120, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, fall_color, 2)
        y += 40
        
        # Seizure detection status
        cv2.putText(display_frame, "SEIZURE:", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        seizure_conf = result.get('seizure_confidence', 0.0)
        seizure_color = (0, 165, 255) if result['seizure_detected'] else (150, 150, 150)
        cv2.putText(display_frame, f"{seizure_conf:.2f}", (180, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, seizure_color, 2)
        y += 45
        
        # Cooldown status
        cooldown = result.get('cooldown_status', {})
        if cooldown.get('global_blocked'):
            cv2.putText(display_frame, f"COOLDOWN: {cooldown['global_cooldown_remaining']:.0f}s", 
                       (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            y += 35
        
        # 🆕 DEBUG INFO - Thêm thông tin chi tiết (FONT LỚN HƠN)
        cv2.putText(display_frame, "BUFFERS:", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        y += 35
        
        # Fall buffer info
        if self.fall_detector:
            buffer_size = len(self.fall_detector.frame_buffer)
            cv2.putText(display_frame, f"Fall: {buffer_size}/5", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y += 30
        
        # Seizure buffer info  
        if self.seizure_detector:
            seizure_buffer = len(self.seizure_detector.frame_buffer)
            cv2.putText(display_frame, f"Seizure: {seizure_buffer}/10", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y += 35
        
        # Aspect ratio from keypoints - HIỂN THỊ LỚN
        if keypoints_list and len(keypoints_list) > 0:
            kpts = keypoints_list[0]
            valid_kpts = kpts[kpts[:, 2] > 0.3]
            if len(valid_kpts) >= 5:
                min_x, max_x = np.min(valid_kpts[:, 0]), np.max(valid_kpts[:, 0])
                min_y, max_y = np.min(valid_kpts[:, 1]), np.max(valid_kpts[:, 1])
                ar = (max_x - min_x) / (max_y - min_y) if (max_y - min_y) > 0 else 0
                ar_status = "LYING" if ar > 1.3 else ("STAND" if ar < 0.7 else "TRANS")
                ar_color = (0, 0, 255) if ar > 1.3 else ((0, 255, 0) if ar < 0.7 else (255, 255, 0))
                cv2.putText(display_frame, f"POSE: {ar_status}", (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ar_color, 2)
                y += 30
                cv2.putText(display_frame, f"AR: {ar:.2f}", (15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, ar_color, 2)
        
        # Event type banner - FONT LỚN HƠN
        event_type = result['event_type'].upper()
        if event_type == 'FALL':
            banner_color = (0, 0, 255)
            banner_text = "!!! FALL DETECTED !!!"
        elif event_type == 'SEIZURE':
            banner_color = (0, 165, 255)
            banner_text = "!!! SEIZURE DETECTED !!!"
        else:
            banner_color = (0, 100, 0)
            banner_text = "NORMAL"
        
        # Draw banner at bottom - LỚN HƠN
        cv2.rectangle(display_frame, (0, h-60), (w, h), banner_color, -1)
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display_frame, banner_text, (text_x, h-18),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Statistics (bottom-right) - FONT LỚN HƠN
        stats_x = w - 300
        stats_y = h - 120
        cv2.putText(display_frame, f"Falls: {self.stats['fall_detections']}", 
                   (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Seizures: {self.stats['seizure_detections']}", 
                   (stats_x, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display_frame
    
    def run(self):
        """Run production test with video"""
        print("\n" + "="*80)
        print(f"🎬 PRODUCTION SYSTEM TEST WITH VIDEO")
        print("="*80)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print(f"💾 Output: {self.output_folder}")
        print(f"🎮 Display: {'ON' if self.show_display else 'OFF'}")
        print(f"⏩ Speed: {self.speed}x")
        print(f"🔄 Skip frames: {self.skip_frames} (process 1/{self.skip_frames} frames)")
        print(f"⏸️ Cooldown: {'OFF' if self.no_cooldown else 'ON'}")
        print(f"🚀 Fast mode: {'ON (no keypoints)' if self.fast_mode else 'OFF (with keypoints)'}")
        print(f"👁️ Preview mode: {'ON (no processing)' if self.preview_mode else 'OFF (full detection)'}")
        print("="*80)
        
        if self.show_display:
            print("\n⌨️ Controls: SPACE=Pause | Q=Quit | S=Save | +/-=Speed")
        print("="*80 + "\n")
        
        # Initialize production components
        if not self.initialize_production_components():
            print("❌ Failed to initialize production components")
            return
        
        # Setup video camera
        camera_config = {
            'video_path': str(self.video_path),
            'camera_id': f'test_video_{self.video_name}',
            'loop': False
        }
        
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print(f"❌ Failed to open video")
            return
        
        total_frames = camera.total_frames
        video_fps = camera.video_fps
        video_width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"▶️ Video: {total_frames} frames, {video_fps:.2f} FPS, {video_width}x{video_height}")
        print("")
        
        # Create display window
        if self.show_display:
            cv2.namedWindow('Production Test', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Production Test', video_width, video_height)
        
        # Processing loop
        frame_number = 0
        paused = False
        start_time = time.time()
        
        try:
            while True:
                if not paused:
                    frame = camera.get_frame()
                    if frame is None:
                        print("\n✅ Video finished!")
                        break
                    
                    frame_number += 1
                    self.stats['total_frames'] = frame_number
                    
                    # Skip frames for faster processing
                    if frame_number % self.skip_frames != 0:
                        # Still show frame but don't process
                        if self.show_display:
                            cv2.imshow("Production Test", frame)
                        # Faster wait time for skipped frames
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("\n⏹️ Stopped by user")
                            break
                        continue
                    
                    # 🆕 PREVIEW MODE: Skip all processing, just show video
                    if self.preview_mode:
                        if self.show_display:
                            # Just show raw frame with simple overlay
                            cv2.putText(frame, f"PREVIEW - Frame {frame_number}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow("Production Test", frame)
                        continue
                    
                    # Process frame using PRODUCTION LOGIC
                    result = self.process_frame(frame, frame_number)
                    
                    # 🆕 Get keypoints for visualization (ONLY if NOT fast mode)
                    keypoints_list = []
                    if not self.fast_mode and frame_number % 5 == 0:  # Every 5 frames if not fast
                        self._cached_keypoints = self.get_keypoints(frame)
                    if not self.fast_mode:
                        keypoints_list = getattr(self, '_cached_keypoints', [])
                    
                    # Display
                    if self.show_display:
                        display_frame = self.draw_results(frame, result, keypoints_list)
                        cv2.imshow("Production Test", display_frame)
                    
                    # Progress log
                    if frame_number % 100 == 0:
                        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                        print(f"⏳ {frame_number}/{total_frames} ({progress:.1f}%) | "
                              f"Falls: {self.stats['fall_detections']} | "
                              f"Seizures: {self.stats['seizure_detections']} | "
                              f"Motion: {result['motion_level']:.3f}")
                
                # Keyboard control - FIXED: use waitKey(1) for speed, timing handled by processing
                wait_time = 1 if self.fast_mode or self.preview_mode else int(1000 / video_fps / self.speed)
                wait_time = max(1, wait_time) if not paused else 100
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('s'):
                    save_path = self.output_folder / f"manual_save_{frame_number}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    print(f"💾 Saved: {save_path.name}")
                elif key == ord('+') or key == ord('='):
                    self.speed = min(4.0, self.speed + 0.25)
                    print(f"⏩ Speed: {self.speed}x")
                elif key == ord('-'):
                    self.speed = max(0.25, self.speed - 0.25)
                    print(f"⏩ Speed: {self.speed}x")
        
        finally:
            cv2.destroyAllWindows()
            camera.disconnect()
        
        # Final statistics
        processing_time = time.time() - start_time
        self.print_final_stats(processing_time)
        self.save_report()
    
    def print_final_stats(self, processing_time: float):
        """Print final statistics"""
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS")
        print("="*80)
        print(f"Video: {self.video_name}")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Frames with Person: {self.stats['frames_with_person']}")
        print("-"*40)
        print(f"🚨 FALL DETECTIONS: {self.stats['fall_detections']}")
        print(f"   Max Confidence: {self.stats['max_fall_confidence']:.3f}")
        for i, alert in enumerate(self.stats['fall_alerts'][:5]):  # Show first 5
            print(f"   [{i+1}] Frame {alert['frame']}: conf={alert['confidence']:.3f}, method={alert['method']}")
        print("-"*40)
        print(f"🧠 SEIZURE DETECTIONS: {self.stats['seizure_detections']}")
        print(f"   Max Confidence: {self.stats['max_seizure_confidence']:.3f}")
        for i, alert in enumerate(self.stats['seizure_alerts'][:5]):  # Show first 5
            print(f"   [{i+1}] Frame {alert['frame']}: conf={alert['confidence']:.3f}")
        print("-"*40)
        print(f"✅ Normal Frames: {self.stats['normal_frames']}")
        print(f"⏱️ Processing Time: {processing_time:.2f}s")
        print(f"📈 Processing FPS: {self.stats['total_frames'] / processing_time:.2f}")
        print(f"📁 Output: {self.output_folder}")
        print("="*80 + "\n")
    
    def save_report(self):
        """Save test report to JSON"""
        import json
        
        report = {
            'video': str(self.video_path),
            'video_name': self.video_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'settings': {
                'fall_cooldown': self.fall_cooldown,
                'seizure_cooldown': self.seizure_cooldown,
                'global_cooldown': self.global_cooldown
            }
        }
        
        report_path = self.output_folder / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📝 Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Test production system với video')
    parser.add_argument('--video', type=str, default="1", help='Video number or name (e.g., 1, 7, test)')
    parser.add_argument('--show', action='store_true', default=True, help='Show display window')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Disable display')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (0.25-4.0)')
    parser.add_argument('--no-cooldown', action='store_true', help='Disable all cooldowns for testing')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame (1=all, 2=skip 1, 3=skip 2, etc)')
    parser.add_argument('--fast', action='store_true', help='🆕 FAST MODE: Disable keypoints for normal speed playback')
    parser.add_argument('--preview', action='store_true', help='🆕 PREVIEW: Just watch video at normal speed, NO detection')
    args = parser.parse_args()
    
    tester = ProductionVideoTest(
        video_input=args.video,
        show_display=args.show,
        speed=args.speed,
        no_cooldown=args.no_cooldown,
        skip_frames=args.skip,
        fast_mode=args.fast
    )
    tester.preview_mode = args.preview
    if args.preview:
        tester.fast_mode = True  # Preview implies fast
    tester.run()


if __name__ == "__main__":
    main()
