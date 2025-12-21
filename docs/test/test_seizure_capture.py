#!/usr/bin/env python3
"""
SEIZURE DETECTION TEST WITH ENHANCED CAPTURE
Test co giật với khả năng capture chi tiết nhân vật bị co giật
Capture: Full frame + Cropped person + Keypoints + Motion analysis
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import video camera
from video_camera_service import VideoCameraService

# Import PRODUCTION seizure detection
try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor
    SEIZURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Seizure detection not available: {e}")
    SEIZURE_AVAILABLE = False

# Import YOLO for person detection
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeizureCaptureTest:
    """Test co giật với capture chi tiết"""
    
    def __init__(self, video_number: int = 1, 
                 capture_threshold: float = 0.30,
                 alert_threshold: float = 0.70,
                 save_all_detections: bool = False):
        """
        Args:
            video_number: Số video trong folder resource
            capture_threshold: Ngưỡng confidence để capture (mặc định 0.30)
            alert_threshold: Ngưỡng alert (mặc định 0.70)
            save_all_detections: Lưu tất cả detections (không chỉ alerts)
        """
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        
        # Find video
        self.video_path = self.find_video(video_number)
        self.video_name = self.video_path.stem
        self.video_number = video_number
        
        # Thresholds
        self.capture_threshold = capture_threshold
        self.alert_threshold = alert_threshold
        self.save_all_detections = save_all_detections
        
        # Output folder với structure chi tiết
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.script_dir / "test_results" / f"seizure_capture_v{video_number}_{timestamp}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Sub-folders
        self.full_frames_folder = self.output_folder / "full_frames"
        self.cropped_persons_folder = self.output_folder / "cropped_persons"
        self.keypoints_folder = self.output_folder / "keypoints"
        self.alerts_folder = self.output_folder / "alerts"
        self.motion_folder = self.output_folder / "motion_analysis"
        
        for folder in [self.full_frames_folder, self.cropped_persons_folder, 
                       self.keypoints_folder, self.alerts_folder, self.motion_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Motion analysis buffers
        self.prev_keypoints = None
        self.motion_history = deque(maxlen=30)
        self.keypoint_history = deque(maxlen=30)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'captures_saved': 0,
            'alerts_saved': 0,
            'max_confidence': 0.0,
            'confidence_timeline': [],
            'motion_timeline': [],
            'alert_frames': [],
            'capture_frames': []
        }
        
        # Models
        self.person_detector = None
        self.pose_detector = None
        self.seizure_detector = None
        self.seizure_predictor = None
        
        print(f"📁 Output folder: {self.output_folder}")
    
    def find_video(self, video_number: int) -> Path:
        """Tìm video theo số"""
        video_path_lower = self.resource_folder / f"{video_number}.mp4"
        video_path_upper = self.resource_folder / f"{video_number}.MP4"
        
        if video_path_lower.exists():
            return video_path_lower
        elif video_path_upper.exists():
            return video_path_upper
        else:
            raise FileNotFoundError(f"Video {video_number} not found in {self.resource_folder}")
    
    def initialize_models(self) -> bool:
        """Initialize detection models"""
        try:
            print("\n" + "="*80)
            print("🔧 INITIALIZING MODELS...")
            print("="*80)
            
            # 1. YOLO Person Detector
            print("📦 Loading YOLO person detector...")
            yolo_path = self.script_dir / "yolov8n.pt"
            if not yolo_path.exists():
                yolo_path = Path("yolov8n.pt")
            self.person_detector = YOLO(str(yolo_path))
            print(f"✅ YOLO person detector loaded")
            
            # 2. YOLO Pose Detector
            print("📦 Loading YOLO pose detector...")
            pose_path = self.script_dir / "yolov8n-pose.pt"
            if not pose_path.exists():
                pose_path = Path("yolov8n-pose.pt")
            self.pose_detector = YOLO(str(pose_path))
            print(f"✅ YOLO pose detector loaded")
            
            if not SEIZURE_AVAILABLE:
                print("⚠️ VSViG Seizure detection not available - using motion analysis only")
                return True
            
            # 3. VSViG Seizure Detector
            print("🧠 Loading VSViG seizure detector...")
            self.seizure_detector = VSViGSeizureDetector(
                confidence_threshold=0.50,
                device='auto'
            )
            
            if not self.seizure_detector.load_models():
                print("⚠️ VSViG models not loaded - using motion analysis only")
            else:
                print("✅ VSViG loaded")
            
            # 4. Seizure Predictor
            print("📊 Initializing seizure predictor...")
            self.seizure_predictor = SeizurePredictor(
                temporal_window=15,
                alert_threshold=self.alert_threshold,
                warning_threshold=self.alert_threshold * 0.7
            )
            print("✅ Seizure predictor initialized")
            
            print("="*80 + "\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
            return False
    
    def detect_persons(self, frame: np.ndarray, prefer_lying: bool = True):
        """Detect persons and return bboxes - có option ưu tiên người nằm"""
        results = self.person_detector(frame, conf=0.15, classes=[0], verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                # Tính lying score dựa trên bbox aspect ratio
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / max(height, 1)
                
                # Lying score từ bbox
                bbox_lying_score = 0
                if aspect_ratio > 1.5:
                    bbox_lying_score = 1.0
                elif aspect_ratio > 1.2:
                    bbox_lying_score = 0.7
                elif aspect_ratio > 0.9:
                    bbox_lying_score = 0.4
                else:
                    bbox_lying_score = 0.1  # Đứng thẳng
                
                persons.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'keypoints': None,
                    'bbox_lying_score': bbox_lying_score,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sắp xếp theo lying_score (ưu tiên người nằm) rồi confidence
        if prefer_lying and len(persons) > 1:
            persons.sort(key=lambda p: (p['bbox_lying_score'], p['confidence']), reverse=True)
        
        return persons
    
    def detect_pose(self, frame: np.ndarray, person_bboxes: list = None):
        """Detect pose keypoints và match với person bboxes - ƯU TIÊN NGƯỜI NẰM"""
        results = self.pose_detector(frame, conf=0.3, verbose=False)
        
        all_keypoints = []
        all_bboxes = []  # Bbox từ pose detector
        all_lying_scores = []  # Score để ưu tiên người nằm
        
        for result in results:
            if result.keypoints is not None and result.boxes is not None:
                for i, kp in enumerate(result.keypoints.data):
                    keypoints = kp.cpu().numpy()  # Shape: (17, 3) - x, y, confidence
                    all_keypoints.append(keypoints)
                    
                    # Lấy bbox từ pose detector
                    if i < len(result.boxes):
                        box = result.boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        all_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                        
                        # Tính lying score cho keypoints này
                        lying_score = self._calculate_lying_score(keypoints, [int(x1), int(y1), int(x2), int(y2)])
                        all_lying_scores.append(lying_score)
                    else:
                        all_bboxes.append(None)
                        all_lying_scores.append(0)
        
        # Match keypoints với person bboxes - ƯU TIÊN NGƯỜI NẰM
        if person_bboxes and all_keypoints:
            matched_keypoints = []
            for person_bbox in person_bboxes:
                best_kp = None
                best_score = -1  # Combined score = IoU + lying_bonus
                
                for i, kp in enumerate(all_keypoints):
                    if i < len(all_bboxes) and all_bboxes[i] is not None:
                        # 1. Calculate IoU
                        iou = self._calculate_iou(person_bbox, all_bboxes[i])
                        
                        # 2. Check if keypoints center is inside person bbox
                        kp_center = self._get_keypoints_center(kp)
                        inside_bonus = 0
                        if kp_center:
                            cx, cy = kp_center
                            if (person_bbox[0] <= cx <= person_bbox[2] and 
                                person_bbox[1] <= cy <= person_bbox[3]):
                                inside_bonus = 0.3  # Bonus for keypoints inside bbox
                        
                        # 3. Lying bonus - ưu tiên người nằm
                        lying_bonus = all_lying_scores[i] * 0.5 if i < len(all_lying_scores) else 0
                        
                        # Combined score
                        combined_score = iou + inside_bonus + lying_bonus
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_kp = kp
                
                # Nếu không có match tốt, tìm keypoints NẰM nhất
                if best_kp is None and all_keypoints:
                    # Ưu tiên keypoints có lying score cao nhất
                    if all_lying_scores:
                        best_lying_idx = max(range(len(all_lying_scores)), key=lambda i: all_lying_scores[i])
                        best_kp = all_keypoints[best_lying_idx]
                    else:
                        best_kp = all_keypoints[0]
                
                matched_keypoints.append(best_kp)
            
            return matched_keypoints
        
        return all_keypoints
    
    def _calculate_lying_score(self, keypoints: np.ndarray, bbox: list) -> float:
        """
        Tính điểm 'nằm' của một bộ keypoints
        Score cao = người đang nằm (horizontal)
        Score thấp/âm = người đang đứng/ngồi/cúi
        
        QUAN TRỌNG: Phân biệt:
        - NẰM: body gần như horizontal, head và hip cùng level Y
        - CÚI: head thấp hơn hip (nhìn xuống), body nghiêng
        - ĐỨNG: head cao hơn hip nhiều
        """
        score = 0.0
        
        if keypoints is None or len(keypoints) < 13:
            return score
        
        try:
            # 1. Check bbox aspect ratio (width/height)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            
            if aspect_ratio > 1.8:  # Rõ ràng nằm ngang
                score += 0.6
            elif aspect_ratio > 1.4:
                score += 0.4
            elif aspect_ratio > 1.1:
                score += 0.2
            elif aspect_ratio < 0.6:  # Rõ ràng đứng thẳng
                score -= 0.4
            
            # 2. Check head-hip vertical distance VÀ hướng
            # COCO: 0=nose, 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip
            nose = keypoints[0]
            l_shoulder = keypoints[5]
            r_shoulder = keypoints[6]
            l_hip = keypoints[11]
            r_hip = keypoints[12]
            
            # Tính Y trung bình của upper và lower body
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
                
                # Signed difference: positive = head ABOVE hip (normal), negative = head BELOW hip (bending)
                # Trong hệ tọa độ image: Y tăng từ trên xuống
                # upper_y < lower_y → head cao hơn hip (đứng/ngồi thường)
                # upper_y > lower_y → head thấp hơn hip (cúi xuống)
                signed_diff = lower_y - upper_y  # positive = head above hip
                
                # Normalize theo bbox height
                normalized_diff = signed_diff / max(height, 1)
                
                # CASE 1: Head cao hơn hip nhiều → ĐỨNG
                if normalized_diff > 0.5:
                    score -= 0.4  # Penalize - đang đứng
                    
                # CASE 2: Head và hip gần level → có thể NẰM
                elif abs(normalized_diff) < 0.25:
                    score += 0.5  # Bonus - có thể nằm
                    
                # CASE 3: Head THẤP hơn hip → CÚI XUỐNG (bending forward)
                elif normalized_diff < -0.1:
                    score -= 0.5  # Penalize mạnh - đang cúi
                    
            # 3. Thêm check: Nếu head ở PHÍA TRÊN frame (y nhỏ) mà hip ở giữa → đứng/cúi
            if upper_y_list and lower_y_list:
                head_y = upper_y_list[0] if len(upper_y_list) > 0 else 0
                hip_y = np.mean(lower_y_list)
                
                # Nếu head ở 1/3 trên của bbox và hip ở 2/3 dưới → đứng/cúi
                bbox_third = height / 3
                head_relative = head_y - y1
                hip_relative = hip_y - y1
                
                if head_relative < bbox_third and hip_relative > bbox_third * 2:
                    score -= 0.3  # Penalize - tư thế đứng/cúi
                    
        except Exception as e:
            logger.debug(f"Lying score calculation error: {e}")
        
        return max(-1, min(score, 1.0))  # Clamp to [-1, 1]
    
    def _get_keypoints_center(self, keypoints: np.ndarray) -> tuple:
        """Tính tâm của keypoints (chỉ dùng các điểm có confidence cao)"""
        if keypoints is None:
            return None
        
        valid_x = []
        valid_y = []
        
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.3:
                valid_x.append(kp[0])
                valid_y.append(kp[1])
        
        if valid_x and valid_y:
            return (np.mean(valid_x), np.mean(valid_y))
        return None
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    def is_person_lying(self, keypoints: np.ndarray = None, bbox: list = None) -> tuple:
        """
        Kiểm tra xem người có đang NẰM hay không
        
        PHÂN BIỆT:
        - NẰM: body horizontal, head và hip gần cùng level Y, bbox width > height
        - CÚI: head THẤP hơn hip (Y lớn hơn), đang cúi xuống giúp người khác
        - ĐỨNG/NGỒI: head CAO hơn hip nhiều
        
        Returns:
            tuple: (is_lying: bool, reason: str, confidence: float)
        """
        is_lying = False
        is_bending = False  # Đang cúi xuống
        reasons = []
        confidence = 0.0
        
        # Method 1: Check bbox aspect ratio
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            
            # Nếu width > height * 1.4 → rõ ràng đang nằm ngang
            if aspect_ratio > 1.4:
                is_lying = True
                confidence += 0.5
                reasons.append(f"bbox_aspect={aspect_ratio:.2f}>1.4")
            elif aspect_ratio > 1.1:
                confidence += 0.25
                reasons.append(f"bbox_aspect={aspect_ratio:.2f}>1.1")
            elif aspect_ratio < 0.7:
                # Bbox dọc → đứng thẳng
                confidence -= 0.3
                reasons.append(f"bbox_aspect={aspect_ratio:.2f}<0.7_standing")
        
        # Method 2: Check keypoints - head vs hip position
        if keypoints is not None and len(keypoints) >= 13:
            try:
                # COCO keypoints: 0=nose, 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip
                nose = keypoints[0]
                l_shoulder = keypoints[5]
                r_shoulder = keypoints[6]
                l_hip = keypoints[11]
                r_hip = keypoints[12]
                
                # Tính trung bình vị trí Y của upper và lower body
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
                    
                    # QUAN TRỌNG: Trong hệ tọa độ image, Y tăng từ TRÊN xuống DƯỚI
                    # - upper_y < lower_y → head CAO hơn hip (đứng/ngồi thường)
                    # - upper_y > lower_y → head THẤP hơn hip (CÚI xuống!)
                    # - upper_y ≈ lower_y → head và hip cùng level (NẰM)
                    
                    signed_diff = lower_y - upper_y  # positive = head above hip
                    
                    if bbox is not None:
                        bbox_height = bbox[3] - bbox[1]
                        normalized_diff = signed_diff / max(bbox_height, 1)
                        
                        # CASE 1: Head CAO hơn hip nhiều → ĐỨNG
                        if normalized_diff > 0.5:
                            is_lying = False
                            confidence -= 0.4
                            reasons.append(f"kp_standing_diff={normalized_diff:.2f}>0.5")
                        
                        # CASE 2: Head và hip GẦN CÙNG LEVEL → NẰM
                        elif abs(normalized_diff) < 0.3:
                            is_lying = True
                            confidence += 0.5
                            reasons.append(f"kp_lying_diff={normalized_diff:.2f}~0")
                        
                        # CASE 3: Head THẤP hơn hip → CÚI XUỐNG (bending)
                        elif normalized_diff < -0.15:
                            is_lying = False
                            is_bending = True
                            confidence -= 0.5
                            reasons.append(f"kp_BENDING_diff={normalized_diff:.2f}<-0.15")
                        
                        # CASE 4: Vùng giữa - có thể ngồi hoặc nằm nghiêng
                        else:
                            confidence += 0.1
                            reasons.append(f"kp_uncertain_diff={normalized_diff:.2f}")
                    else:
                        # Không có bbox, dùng absolute
                        abs_diff = abs(signed_diff)
                        if abs_diff < 80:
                            is_lying = True
                            confidence += 0.3
                            reasons.append(f"kp_abs_diff={abs_diff:.0f}<80")
                        elif signed_diff < -50:
                            is_bending = True
                            confidence -= 0.4
                            reasons.append(f"kp_BENDING_abs={signed_diff:.0f}")
                            
            except Exception as e:
                logger.debug(f"Keypoint lying check error: {e}")
        
        # Final decision
        # Nếu đang cúi → KHÔNG được xem là nằm
        if is_bending:
            is_lying = False
            confidence = min(confidence, 0)  # Ensure negative/zero
        elif confidence >= 0.5:
            is_lying = True
        elif confidence <= 0:
            is_lying = False
        
        reason_str = " | ".join(reasons) if reasons else "no_data"
        
        return is_lying, reason_str, max(0, min(confidence, 1.0))

    def calculate_motion(self, current_keypoints: np.ndarray) -> dict:
        """Tính toán motion từ keypoints"""
        if self.prev_keypoints is None or current_keypoints is None:
            self.prev_keypoints = current_keypoints
            return {'motion_score': 0, 'body_parts_motion': {}}
        
        try:
            # Body part indices
            body_parts = {
                'head': [0, 1, 2, 3, 4],
                'arms': [5, 6, 7, 8, 9, 10],
                'torso': [5, 6, 11, 12],
                'legs': [11, 12, 13, 14, 15, 16]
            }
            
            motion_scores = {}
            total_motion = 0
            valid_points = 0
            
            for part_name, indices in body_parts.items():
                part_motion = 0
                part_valid = 0
                
                for idx in indices:
                    if idx < len(current_keypoints) and idx < len(self.prev_keypoints):
                        curr_pt = current_keypoints[idx]
                        prev_pt = self.prev_keypoints[idx]
                        
                        # Check confidence
                        if len(curr_pt) >= 3 and len(prev_pt) >= 3:
                            if curr_pt[2] > 0.3 and prev_pt[2] > 0.3:
                                dx = curr_pt[0] - prev_pt[0]
                                dy = curr_pt[1] - prev_pt[1]
                                motion = np.sqrt(dx*dx + dy*dy)
                                part_motion += motion
                                part_valid += 1
                                total_motion += motion
                                valid_points += 1
                
                motion_scores[part_name] = part_motion / max(part_valid, 1)
            
            avg_motion = total_motion / max(valid_points, 1)
            
            # Normalize motion score (0-1)
            # Giả sử max motion là 100 pixels per frame
            normalized_motion = min(avg_motion / 100.0, 1.0)
            
            self.prev_keypoints = current_keypoints
            
            return {
                'motion_score': normalized_motion,
                'raw_motion': avg_motion,
                'body_parts_motion': motion_scores
            }
            
        except Exception as e:
            logger.warning(f"Motion calculation error: {e}")
            return {'motion_score': 0, 'body_parts_motion': {}}
    
    def detect_seizure_by_motion(self, motion_history: list) -> dict:
        """Phát hiện co giật dựa trên motion patterns"""
        if len(motion_history) < 5:
            return {'seizure_confidence': 0, 'pattern': 'insufficient_data'}
        
        motions = [m['motion_score'] for m in list(motion_history)[-15:]]
        
        # Tính các đặc trưng
        avg_motion = np.mean(motions)
        motion_std = np.std(motions)
        max_motion = np.max(motions)
        
        # Đếm oscillations (chuyển động giật)
        oscillations = 0
        for i in range(1, len(motions)):
            if abs(motions[i] - motions[i-1]) > 0.1:
                oscillations += 1
        oscillation_rate = oscillations / len(motions)
        
        # Seizure indicators
        high_motion = avg_motion > 0.3
        high_variability = motion_std > 0.15
        rapid_oscillation = oscillation_rate > 0.5
        
        # Calculate confidence
        confidence = 0
        pattern = 'normal'
        
        if high_motion:
            confidence += 0.3
        if high_variability:
            confidence += 0.3
        if rapid_oscillation:
            confidence += 0.4
            pattern = 'rapid_oscillation'
        
        if confidence > 0.5:
            pattern = 'seizure_like'
        
        return {
            'seizure_confidence': confidence,
            'pattern': pattern,
            'avg_motion': avg_motion,
            'motion_std': motion_std,
            'oscillation_rate': oscillation_rate
        }
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color=(0, 255, 255)):
        """Draw skeleton keypoints"""
        if keypoints is None or len(keypoints) == 0:
            return
        
        # COCO connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                if len(start_point) >= 3 and len(end_point) >= 3:
                    if start_point[2] > 0.3 and end_point[2] > 0.3:
                        cv2.line(frame, 
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                color, 2)
        
        # Draw keypoints
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.3:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 0, 255), -1)
    
    def save_capture(self, frame: np.ndarray, person_bbox: list, keypoints: np.ndarray,
                     frame_number: int, confidence: float, motion_data: dict,
                     is_alert: bool = False):
        """Lưu capture chi tiết"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"frame_{frame_number:06d}_{timestamp}_conf_{confidence:.3f}"
        
        try:
            # 1. Save full frame với annotations
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = person_bbox
            
            # Draw bbox
            color = (0, 0, 255) if is_alert else (0, 165, 255) if confidence > 0.5 else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw keypoints
            if keypoints is not None:
                self.draw_skeleton(annotated_frame, keypoints, color)
            
            # Draw info
            info_text = f"Conf: {confidence:.3f} | Motion: {motion_data.get('motion_score', 0):.3f}"
            cv2.putText(annotated_frame, info_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_alert:
                cv2.putText(annotated_frame, "!!! SEIZURE ALERT !!!", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            full_frame_path = self.full_frames_folder / f"{base_name}_full.jpg"
            cv2.imwrite(str(full_frame_path), annotated_frame)
            
            # 2. Save cropped person
            h, w = frame.shape[:2]
            # Expand bbox slightly
            margin = 30
            crop_x1 = max(0, x1 - margin)
            crop_y1 = max(0, y1 - margin)
            crop_x2 = min(w, x2 + margin)
            crop_y2 = min(h, y2 + margin)
            
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Draw keypoints on cropped (adjust coordinates)
            if keypoints is not None:
                adjusted_kps = keypoints.copy()
                adjusted_kps[:, 0] -= crop_x1
                adjusted_kps[:, 1] -= crop_y1
                self.draw_skeleton(cropped, adjusted_kps, (0, 255, 255))
            
            cropped_path = self.cropped_persons_folder / f"{base_name}_person.jpg"
            cv2.imwrite(str(cropped_path), cropped)
            
            # 3. Save keypoints visualization - FIX: vẽ đúng tỷ lệ
            if keypoints is not None:
                # Tạo canvas lớn hơn để vẽ rõ
                kp_canvas = np.zeros((500, 400, 3), dtype=np.uint8)
                
                # Lọc keypoints có confidence > 0.3
                valid_mask = keypoints[:, 2] > 0.3
                if np.any(valid_mask):
                    valid_kps = keypoints[valid_mask]
                    
                    min_x = np.min(valid_kps[:, 0])
                    max_x = np.max(valid_kps[:, 0])
                    min_y = np.min(valid_kps[:, 1])
                    max_y = np.max(valid_kps[:, 1])
                    
                    # Tính scale để fit vào canvas với margin
                    margin = 50
                    canvas_w = 400 - 2 * margin
                    canvas_h = 500 - 2 * margin
                    
                    range_x = max(max_x - min_x, 1)
                    range_y = max(max_y - min_y, 1)
                    
                    scale = min(canvas_w / range_x, canvas_h / range_y)
                    
                    # Center trong canvas
                    offset_x = margin + (canvas_w - range_x * scale) / 2
                    offset_y = margin + (canvas_h - range_y * scale) / 2
                    
                    # Normalize keypoints
                    normalized_kps = keypoints.copy()
                    normalized_kps[:, 0] = (keypoints[:, 0] - min_x) * scale + offset_x
                    normalized_kps[:, 1] = (keypoints[:, 1] - min_y) * scale + offset_y
                    
                    # Vẽ skeleton
                    self.draw_skeleton(kp_canvas, normalized_kps, (0, 255, 0))
                    
                    # Thêm label cho từng keypoint
                    kp_names = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
                               'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
                               'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
                               'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
                    
                    for i, (kp, name) in enumerate(zip(normalized_kps, kp_names)):
                        if kp[2] > 0.3:
                            x, y = int(kp[0]), int(kp[1])
                            cv2.putText(kp_canvas, name, (x+5, y-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # Thêm info
                    cv2.putText(kp_canvas, f"Frame: {frame_number}", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(kp_canvas, f"Conf: {confidence:.3f}", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                kp_path = self.keypoints_folder / f"{base_name}_keypoints.jpg"
                cv2.imwrite(str(kp_path), kp_canvas)
            
            # 4. Save alert frame if applicable
            if is_alert:
                alert_path = self.alerts_folder / f"ALERT_{base_name}.jpg"
                cv2.imwrite(str(alert_path), annotated_frame)
                self.stats['alerts_saved'] += 1
            
            # 5. Save metadata JSON - FIX: convert numpy to native Python types
            def convert_to_native(obj):
                """Convert numpy types to native Python types for JSON"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(i) for i in obj]
                return obj
            
            metadata = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'video_name': self.video_name,
                'confidence': float(confidence),
                'is_alert': is_alert,
                'bbox': [int(x) for x in person_bbox],
                'motion_data': convert_to_native(motion_data),
                'keypoints': keypoints.tolist() if keypoints is not None else None
            }
            
            meta_path = self.full_frames_folder / f"{base_name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.stats['captures_saved'] += 1
            self.stats['capture_frames'].append(frame_number)
            
            if is_alert:
                self.stats['alert_frames'].append(frame_number)
            
            logger.info(f"{'🚨 ALERT' if is_alert else '📸 CAPTURE'}: Frame {frame_number}, Conf: {confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Error saving capture: {e}")
    
    def draw_results(self, frame: np.ndarray, persons: list, all_keypoints: list,
                    seizure_result: dict, motion_data: dict, frame_number: int) -> np.ndarray:
        """Draw detection results on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw person bounding boxes
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            
            # Color based on seizure
            confidence = seizure_result.get('smoothed_confidence', 0)
            if seizure_result.get('alert_triggered', False):
                color = (0, 0, 255)  # Red
                thickness = 3
            elif confidence > self.alert_threshold * 0.7:
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw skeleton
            if i < len(all_keypoints):
                self.draw_skeleton(display_frame, all_keypoints[i], color)
        
        # Info panel (left side)
        panel_width = min(400, w // 3)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        y = 25
        cv2.putText(display_frame, "SEIZURE CAPTURE TEST", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 35
        
        # Frame info
        cv2.putText(display_frame, f"Frame: {frame_number}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(display_frame, f"Persons: {len(persons)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        
        # Lying status - IMPORTANT!
        is_lying = seizure_result.get('is_lying', False)
        lying_reason = seizure_result.get('lying_reason', 'unknown')
        if is_lying:
            cv2.putText(display_frame, "Status: LYING [DETECT ON]", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Status: STANDING [SKIP]", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        y += 35
        
        # Seizure metrics
        cv2.putText(display_frame, "SEIZURE METRICS:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        smooth_conf = seizure_result.get('smoothed_confidence', 0.0)
        raw_conf = seizure_result.get('raw_confidence', 0.0)
        
        conf_color = (0, 255, 0) if smooth_conf < self.alert_threshold*0.7 else \
                     (0, 165, 255) if smooth_conf < self.alert_threshold else (0, 0, 255)
        
        cv2.putText(display_frame, f"Raw Conf: {raw_conf:.4f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20
        cv2.putText(display_frame, f"Smooth Conf: {smooth_conf:.4f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)
        y += 25
        
        # Confidence bar
        bar_width = panel_width - 20
        bar_height = 15
        cv2.rectangle(display_frame, (10, y), (10 + bar_width, y + bar_height), (50, 50, 50), -1)
        fill_width = int(bar_width * min(smooth_conf, 1.0))
        cv2.rectangle(display_frame, (10, y), (10 + fill_width, y + bar_height), conf_color, -1)
        
        # Threshold markers
        threshold_x = int(10 + bar_width * self.alert_threshold)
        cv2.line(display_frame, (threshold_x, y), (threshold_x, y + bar_height), (255, 255, 255), 2)
        y += 35
        
        # Motion analysis
        cv2.putText(display_frame, "MOTION ANALYSIS:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        motion_score = motion_data.get('motion_score', 0)
        cv2.putText(display_frame, f"Motion Score: {motion_score:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20
        
        # Body parts motion
        parts = motion_data.get('body_parts_motion', {})
        for part, score in parts.items():
            color = (0, 0, 255) if score > 30 else (0, 165, 255) if score > 15 else (0, 255, 0)
            cv2.putText(display_frame, f"  {part}: {score:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 16
        
        y += 20
        
        # Statistics
        cv2.putText(display_frame, "CAPTURES:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        cv2.putText(display_frame, f"Saved: {self.stats['captures_saved']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 20
        cv2.putText(display_frame, f"Alerts: {self.stats['alerts_saved']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        y += 20
        cv2.putText(display_frame, f"Max Conf: {self.stats['max_confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Alert banner
        if seizure_result.get('alert_triggered', False):
            banner_text = "!!! SEIZURE ALERT - CAPTURING !!!"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)[0]
            text_x = (w - text_size[0]) // 2
            
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(display_frame, (text_x-10, h-60), 
                            (text_x + text_size[0]+10, h-20), (0, 0, 255), -1)
                cv2.putText(display_frame, banner_text, (text_x, h-30),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        
        return display_frame
    
    def run(self):
        """Run capture test"""
        print("\n" + "="*80)
        print(f"🎬 SEIZURE DETECTION CAPTURE TEST")
        print("="*80)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print(f"💾 Output: {self.output_folder}")
        print(f"🎯 Capture Threshold: {self.capture_threshold}")
        print(f"🚨 Alert Threshold: {self.alert_threshold}")
        print("="*80)
        print("\n⌨️ Controls: SPACE=Pause | Q=Quit | S=Manual Save | C=Force Capture")
        print("="*80 + "\n")
        
        # Initialize models
        if not self.initialize_models():
            print("❌ Model initialization failed")
            return
        
        # Setup camera - IMPORTANT: set resolution to None to keep original size
        camera_config = {
            'video_path': str(self.video_path),
            'camera_id': f'capture_video_{self.video_name}',
            'loop': False,
            'resolution': None  # Keep original video resolution, don't resize!
        }
        
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print(f"❌ Failed to open video")
            return
        
        total_frames = camera.total_frames
        video_fps = camera.video_fps
        
        # Get video dimensions
        video_width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"▶️ Playing: {total_frames} frames, {video_fps:.2f} FPS")
        print(f"📺 Resolution: {video_width}x{video_height}\n")
        
        # Create window
        cv2.namedWindow('Seizure Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Seizure Capture', video_width, video_height)
        
        # Processing loop
        frame_number = 0
        paused = False
        start_time = time.time()
        last_capture_frame = -10  # Avoid duplicate captures
        
        try:
            while True:
                if not paused:
                    frame = camera.get_frame()
                    if frame is None:
                        print("\n✅ Video finished!")
                        break
                    
                    frame_number += 1
                    self.stats['total_frames'] = frame_number
                    
                    # Detect persons
                    persons = self.detect_persons(frame)
                    
                    # Detect pose - pass person bboxes để match đúng
                    person_bboxes = [p['bbox'] for p in persons] if persons else None
                    all_keypoints = self.detect_pose(frame, person_bboxes)
                    
                    if len(persons) > 0:
                        self.stats['frames_with_person'] += 1
                    
                    # Motion analysis
                    motion_data = {'motion_score': 0, 'body_parts_motion': {}}
                    is_lying = False
                    lying_reason = "no_person"
                    lying_confidence = 0.0
                    
                    if len(all_keypoints) > 0 and len(persons) > 0:
                        # Check if person is lying down
                        is_lying, lying_reason, lying_confidence = self.is_person_lying(
                            keypoints=all_keypoints[0],
                            bbox=persons[0]['bbox']
                        )
                        
                        # Only calculate motion if person is lying
                        if is_lying:
                            motion_data = self.calculate_motion(all_keypoints[0])
                            self.motion_history.append(motion_data)
                            self.keypoint_history.append(all_keypoints[0])
                        else:
                            # Reset motion history when person is not lying
                            if len(self.motion_history) > 0:
                                self.motion_history.clear()
                                self.keypoint_history.clear()
                                self.prev_keypoints = None
                    
                    # Seizure detection - ONLY if person is lying
                    seizure_result = {
                        'raw_confidence': 0.0,
                        'smoothed_confidence': 0.0,
                        'alert_triggered': False,
                        'is_lying': is_lying,
                        'lying_reason': lying_reason
                    }
                    
                    # Skip seizure detection if person is NOT lying
                    if not is_lying:
                        # Person is standing/sitting - skip detection
                        if frame_number % 50 == 0 and len(persons) > 0:
                            logger.info(f"⏭️ Frame {frame_number}: Person NOT lying ({lying_reason}) - skipping seizure detection")
                    else:
                        # Person IS lying - proceed with seizure detection
                        # Use VSViG if available
                        if SEIZURE_AVAILABLE and self.seizure_detector and len(persons) > 0:
                            for person in persons:
                                bbox = person['bbox']
                                detection = self.seizure_detector.detect_seizure(frame, bbox)
                                
                                if detection.get('temporal_ready', False):
                                    confidence = detection.get('confidence', 0.0)
                                    prediction = self.seizure_predictor.update_prediction(confidence)
                                    
                                    seizure_result = {
                                        'raw_confidence': confidence,
                                        'smoothed_confidence': prediction['smoothed_confidence'],
                                        'alert_triggered': prediction['alert_level'] in ['alert', 'critical'],
                                        'alert_level': prediction['alert_level'],
                                        'is_lying': is_lying,
                                        'lying_reason': lying_reason
                                    }
                        
                        # Fallback: Motion-based detection (only if lying)
                        if seizure_result['smoothed_confidence'] == 0 and len(self.motion_history) >= 5:
                            motion_seizure = self.detect_seizure_by_motion(list(self.motion_history))
                            seizure_result['raw_confidence'] = motion_seizure['seizure_confidence']
                            seizure_result['smoothed_confidence'] = motion_seizure['seizure_confidence']
                            seizure_result['alert_triggered'] = motion_seizure['seizure_confidence'] >= self.alert_threshold
                    
                    # Update stats
                    smooth_conf = seizure_result['smoothed_confidence']
                    if smooth_conf > self.stats['max_confidence']:
                        self.stats['max_confidence'] = smooth_conf
                    
                    self.stats['confidence_timeline'].append({
                        'frame': frame_number,
                        'confidence': smooth_conf,
                        'motion': motion_data.get('motion_score', 0)
                    })
                    
                    # Auto capture logic - ONLY if person is lying
                    should_capture = False
                    is_alert = False
                    
                    # Only capture if person is lying
                    if is_lying:
                        if seizure_result['alert_triggered']:
                            should_capture = True
                            is_alert = True
                        elif smooth_conf >= self.capture_threshold and self.save_all_detections:
                            should_capture = True
                        elif smooth_conf >= self.capture_threshold and frame_number - last_capture_frame >= 30:
                            # Capture every 30 frames if above threshold
                            should_capture = True
                    
                    # Save capture
                    if should_capture and len(persons) > 0:
                        keypoints = all_keypoints[0] if len(all_keypoints) > 0 else None
                        self.save_capture(
                            frame, persons[0]['bbox'], keypoints,
                            frame_number, smooth_conf, motion_data, is_alert
                        )
                        last_capture_frame = frame_number
                    
                    # Draw results
                    display_frame = self.draw_results(
                        frame, persons, all_keypoints,
                        seizure_result, motion_data, frame_number
                    )
                    
                    # Display
                    cv2.imshow("Seizure Capture", display_frame)
                    
                    # Progress
                    if frame_number % 100 == 0:
                        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                        lying_status = "LYING" if is_lying else "STAND"
                        print(f"⏳ {frame_number}/{total_frames} ({progress:.1f}%) | "
                              f"[{lying_status}] | Conf: {smooth_conf:.3f} | Motion: {motion_data.get('motion_score', 0):.3f} | "
                              f"Captures: {self.stats['captures_saved']} | Alerts: {self.stats['alerts_saved']}")
                
                # Keyboard
                key = cv2.waitKey(1 if not paused else 100) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('s'):
                    # Manual save
                    if len(persons) > 0:
                        keypoints = all_keypoints[0] if len(all_keypoints) > 0 else None
                        self.save_capture(
                            frame, persons[0]['bbox'], keypoints,
                            frame_number, smooth_conf, motion_data, False
                        )
                        print(f"💾 Manual save at frame {frame_number}")
                elif key == ord('c'):
                    # Force capture as alert
                    if len(persons) > 0:
                        keypoints = all_keypoints[0] if len(all_keypoints) > 0 else None
                        self.save_capture(
                            frame, persons[0]['bbox'], keypoints,
                            frame_number, 1.0, motion_data, True
                        )
                        print(f"🚨 Force capture as ALERT at frame {frame_number}")
        
        finally:
            cv2.destroyAllWindows()
            camera.disconnect()
        
        # Save final report
        self.save_report(time.time() - start_time)
    
    def save_report(self, processing_time: float):
        """Save final report"""
        report = {
            'video': {
                'name': self.video_name,
                'path': str(self.video_path),
                'number': self.video_number
            },
            'thresholds': {
                'capture': self.capture_threshold,
                'alert': self.alert_threshold
            },
            'statistics': {
                'total_frames': self.stats['total_frames'],
                'frames_with_person': self.stats['frames_with_person'],
                'captures_saved': self.stats['captures_saved'],
                'alerts_saved': self.stats['alerts_saved'],
                'max_confidence': self.stats['max_confidence'],
                'processing_time': processing_time,
                'processing_fps': self.stats['total_frames'] / max(processing_time, 1)
            },
            'alert_frames': self.stats['alert_frames'],
            'capture_frames': self.stats['capture_frames']
        }
        
        report_path = self.output_folder / "capture_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 FINAL REPORT")
        print("="*80)
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Frames with Person: {self.stats['frames_with_person']}")
        print(f"📸 Captures Saved: {self.stats['captures_saved']}")
        print(f"🚨 Alerts Saved: {self.stats['alerts_saved']}")
        print(f"Max Confidence: {self.stats['max_confidence']:.3f}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Processing FPS: {self.stats['total_frames'] / max(processing_time, 1):.2f}")
        print(f"\n📁 Output folder: {self.output_folder}")
        print(f"   📂 Full frames: {self.full_frames_folder}")
        print(f"   📂 Cropped persons: {self.cropped_persons_folder}")
        print(f"   📂 Keypoints: {self.keypoints_folder}")
        print(f"   📂 Alerts: {self.alerts_folder}")
        print("="*80 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test seizure detection with capture')
    parser.add_argument('--video', type=int, default=1, help='Video number (1-39)')
    parser.add_argument('--capture-threshold', type=float, default=0.30, 
                       help='Confidence threshold to capture (default: 0.30)')
    parser.add_argument('--alert-threshold', type=float, default=0.70,
                       help='Confidence threshold for alert (default: 0.70)')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all detections above capture threshold')
    args = parser.parse_args()
    
    tester = SeizureCaptureTest(
        video_number=args.video,
        capture_threshold=args.capture_threshold,
        alert_threshold=args.alert_threshold,
        save_all_detections=args.save_all
    )
    tester.run()


if __name__ == "__main__":
    main()
