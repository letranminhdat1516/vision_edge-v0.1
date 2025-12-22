#!/usr/bin/env python3
"""
SEIZURE DETECTION ONLY - Test Script
Chỉ test Seizure Detection, không fall
Focus vào việc phân tích chi tiết seizure detection

Usage:
    python test_seizure_only.py --video 1              # Video co giật
    python test_seizure_only.py --video 1 --slow       # Chậm để phân tích
    python test_seizure_only.py --video 1 --fast       # Nhanh, không hiển thị
    python test_seizure_only.py --video 1 --realtime   # Sync với video FPS
    
Controls:
    Q/ESC   = Quit
    SPACE   = Pause/Resume
    S       = Save frame manually
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import argparse

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import video camera service (local)
from video_camera_service import VideoCameraService

# ============================================================================
# IMPORT SEIZURE DETECTION MODULES
# ============================================================================

try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor
    SEIZURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Seizure detection not available: {e}")
    SEIZURE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available")
    YOLO_AVAILABLE = False

# Logging will be setup in class with file handler
logger = logging.getLogger(__name__)


class SeizureDetectionTest:
    """Test Seizure Detection với video - Chi tiết từng frame"""
    
    def __init__(self, video_input: str = "1", slow_mode: bool = False, 
                 fast_mode: bool = False, realtime_mode: bool = False,
                 start_frame: int = 0, end_frame: int = -1):
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        
        # Find video FIRST (needed for output folder)
        self.video_path = self.find_video(video_input)
        self.video_name = self.video_path.stem
        
        # Output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.script_dir / "test_results" / f"seizure_only_{self.video_name}_{timestamp}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 🆕 Setup logging với FILE HANDLER
        self.log_file = self.output_folder / "seizure_detection.log"
        self._setup_logging()
        
        # Settings
        self.slow_mode = slow_mode
        self.fast_mode = fast_mode
        self.realtime_mode = realtime_mode
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'frames_with_keypoints': 0,
            'seizure_detections': 0,
            'seizure_alerts': 0,
            'seizure_warnings': 0,
            'seizure_events': [],
            'max_confidence': 0.0,
            'alert_levels': {},  # Đếm theo alert level
            'confidence_history': [],  # Lưu lịch sử confidence
        }
        
        # Components
        self.person_detector = None
        self.seizure_detector = None
        self.seizure_predictor = None
        
        print(f"📁 Output: {self.output_folder}")
        print(f"📝 Log file: {self.log_file}")
    
    def _setup_logging(self):
        """Setup logging với cả console và file"""
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Format
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler (INFO level - less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
        
        # File handler (DEBUG level - full detail)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to: {self.log_file}")
    
    def find_video(self, video_input: str) -> Path:
        """Find video file"""
        # Check exact name first (with various extensions)
        for ext in ['.mp4', '.MP4', '.avi', '.mov', '.mkv']:
            exact_path = self.resource_folder / f"{video_input}{ext}"
            if exact_path.exists():
                return exact_path
        
        # Search in folder
        if self.resource_folder.exists():
            for video_file in self.resource_folder.glob("*"):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    if video_input.lower() in video_file.stem.lower():
                        return video_file
        
        raise FileNotFoundError(f"Video not found: {video_input}")
    
    def initialize(self):
        """Initialize detection components"""
        print("\n" + "="*60)
        print("🔧 INITIALIZING SEIZURE DETECTION")
        print("="*60)
        
        try:
            # Person detector
            print("👤 Loading person detector (YOLOv8n)...")
            model_path = PROJECT_ROOT / "yolov8n.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "yolov8s.pt"
            self.person_detector = YOLO(str(model_path))
            print(f"   ✅ Loaded: {model_path.name}")
            
            if not SEIZURE_AVAILABLE:
                print("⚠️ Seizure detection not available!")
                return False
            
            # Seizure detector (VSViG)
            print("🧠 Loading VSViG seizure detector...")
            self.seizure_detector = VSViGSeizureDetector(
                confidence_threshold=0.65,  # Lower for testing
                device='auto'
            )
            
            if self.seizure_detector.load_models():
                print(f"   ✅ VSViG loaded (threshold: {self.seizure_detector.confidence_threshold})")
            else:
                print("   ⚠️ VSViG models not loaded - will use fallback")
            
            # Seizure predictor (temporal analysis)
            print("📊 Loading seizure predictor...")
            self.seizure_predictor = SeizurePredictor(
                temporal_window=15,      # 15 frames temporal window
                alert_threshold=0.70,    # Alert at 70%
                warning_threshold=0.50   # Warning at 50%
            )
            print(f"   ✅ Predictor loaded (alert: {self.seizure_predictor.alert_threshold}, warning: {self.seizure_predictor.warning_threshold})")
            
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Init error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_person(self, frame) -> dict:
        """Detect person in frame"""
        results = self.person_detector(frame, classes=[0], conf=0.15, verbose=False)
        
        best_person = None
        for result in results:
            for box in result.boxes:
                if box.cls[0] == 0:  # Person
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    if best_person is None or conf > best_person['confidence']:
                        best_person = {
                            'bbox': bbox,
                            'confidence': conf
                        }
        
        return best_person
    
    def process_frame(self, frame, frame_number, timestamp) -> dict:
        """Process single frame for seizure detection"""
        result = {
            'frame': frame_number,
            'timestamp': timestamp,
            'person_detected': False,
            'keypoints_detected': False,
            'seizure_detected': False,
            'raw_confidence': 0.0,
            'smoothed_confidence': 0.0,
            'alert_level': 'normal',
            'alert_message': '',
            'temporal_ready': False,
            'bbox': None,
            'keypoints': None
        }
        
        # Detect person
        person = self.detect_person(frame)
        if person is None:
            return result
        
        result['person_detected'] = True
        result['bbox'] = person['bbox'].tolist()
        self.stats['frames_with_person'] += 1
        
        # Run seizure detection
        try:
            # Convert bbox to list of integers (required for slicing in vsvig_detector)
            bbox_int = [int(v) for v in person['bbox']]
            
            seizure_result = self.seizure_detector.detect_seizure(
                frame=frame,
                person_bbox=bbox_int
            )
            
            result['temporal_ready'] = seizure_result.get('temporal_ready', False)
            result['raw_confidence'] = seizure_result.get('confidence', 0.0)
            result['keypoints'] = seizure_result.get('keypoints')
            result['motion_intensity'] = seizure_result.get('motion_intensity', 0.0)
            
            if result['keypoints'] is not None:
                result['keypoints_detected'] = True
                self.stats['frames_with_keypoints'] += 1
            
            # Log raw detection
            if result['raw_confidence'] > 0:
                logger.debug(f"🔍 Frame {frame_number}: raw_conf={result['raw_confidence']:.3f}, temporal_ready={result['temporal_ready']}")
            
            # 🔥 USE VSVIG_DETECTOR'S ALERT_LEVEL DIRECTLY (includes motion + accumulation check)
            # Don't override with seizure_predictor - vsvig_detector already does proper filtering
            result['alert_level'] = seizure_result.get('alert_level', 'normal')
            result['seizure_detected'] = seizure_result.get('seizure_detected', False)
            result['accumulation_frames'] = seizure_result.get('accumulation_frames', 0)
            result['accumulation_threshold'] = seizure_result.get('accumulation_threshold', 90)
            
            # Still use predictor for smoothed_confidence display only
            prediction = self.seizure_predictor.update_prediction(
                confidence=result['raw_confidence'],
                timestamp=timestamp
            )
            result['smoothed_confidence'] = prediction.get('smoothed_confidence', 0.0)
            result['alert_message'] = prediction.get('alert_message', '')
            result['seizure_duration'] = prediction.get('seizure_duration', 0.0)
            result['temporal_analysis'] = prediction.get('temporal_analysis', {})
            
            # Track statistics
            self.stats['confidence_history'].append({
                'frame': frame_number,
                'raw': result['raw_confidence'],
                'smoothed': result['smoothed_confidence'],
                'alert_level': result['alert_level']
            })
            
            # Track alert levels
            level = result['alert_level']
            self.stats['alert_levels'][level] = self.stats['alert_levels'].get(level, 0) + 1
            
            # Track max confidence
            if result['smoothed_confidence'] > self.stats['max_confidence']:
                self.stats['max_confidence'] = result['smoothed_confidence']
            
            # Log significant events
            if result['alert_level'] == 'critical':
                logger.warning(f"🚨 SEIZURE CRITICAL at frame {frame_number}: smoothed={result['smoothed_confidence']:.3f}")
            elif result['alert_level'] == 'warning':
                logger.info(f"⚠️ SEIZURE WARNING at frame {frame_number}: smoothed={result['smoothed_confidence']:.3f}")
            elif result['smoothed_confidence'] > 0.3:
                logger.debug(f"📊 Frame {frame_number}: smoothed={result['smoothed_confidence']:.3f}, level={result['alert_level']}")
            
        except Exception as e:
            logger.error(f"❌ Seizure detection error at frame {frame_number}: {e}")
        
        return result
    
    def save_frame(self, frame, frame_number, result, prefix=""):
        """Save frame with annotations"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw bbox
        if result['bbox']:
            x1, y1, x2, y2 = [int(v) for v in result['bbox']]
            
            # Color based on alert level
            if result['alert_level'] == 'critical':
                color = (0, 0, 255)  # Red
            elif result['alert_level'] == 'warning':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints if available
        if result.get('keypoints') is not None:
            self._draw_keypoints(display, result['keypoints'])
        
        # Info panel
        y = 30
        cv2.putText(display, f"Frame: {frame_number}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Raw Conf: {result['raw_confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(display, f"Smoothed: {result['smoothed_confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(display, f"Alert: {result['alert_level']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Alert banner
        if result['alert_level'] == 'critical':
            cv2.rectangle(display, (0, h-50), (w, h), (0, 0, 255), -1)
            cv2.putText(display, f"!!! SEIZURE DETECTED !!! conf={result['smoothed_confidence']:.2f}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif result['alert_level'] == 'warning':
            cv2.rectangle(display, (0, h-50), (w, h), (0, 165, 255), -1)
            cv2.putText(display, f"⚠️ SEIZURE WARNING conf={result['smoothed_confidence']:.2f}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save
        filename = f"{prefix}frame_{frame_number:05d}_{result['alert_level']}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), display)
        
        return filepath
    
    def _draw_keypoints(self, frame, keypoints):
        """Draw keypoints on frame"""
        if keypoints is None:
            return
        
        # COCO keypoint connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Convert keypoints to list if needed
        if hasattr(keypoints, 'cpu'):
            kpts = keypoints.cpu().numpy()
        else:
            kpts = np.array(keypoints)
        
        if kpts.ndim == 3:
            kpts = kpts[0]  # Take first person
        
        # Draw connections
        for i, j in connections:
            if i < len(kpts) and j < len(kpts):
                x1, y1 = int(kpts[i][0]), int(kpts[i][1])
                x2, y2 = int(kpts[j][0]), int(kpts[j][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw keypoints
        for i, kpt in enumerate(kpts):
            x, y = int(kpt[0]), int(kpt[1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
    
    def draw_display(self, frame, result) -> np.ndarray:
        """Draw display frame with info"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw bbox
        if result['bbox']:
            x1, y1, x2, y2 = [int(v) for v in result['bbox']]
            
            # Color based on alert level
            if result['alert_level'] == 'critical':
                color = (0, 0, 255)  # Red
                thickness = 4
            elif result['alert_level'] == 'warning':
                color = (0, 165, 255)  # Orange
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        
        # Draw keypoints
        if result.get('keypoints') is not None:
            self._draw_keypoints(display, result['keypoints'])
        
        # Info panel (left)
        panel_w = 400
        cv2.rectangle(display, (0, 0), (panel_w, 250), (0, 0, 0), -1)
        
        y = 30
        cv2.putText(display, f"Frame: {result['frame']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35
        
        # Raw confidence
        raw_color = (0, 255, 0) if result['raw_confidence'] < 0.5 else \
                    (0, 165, 255) if result['raw_confidence'] < 0.7 else (0, 0, 255)
        cv2.putText(display, f"Raw Confidence: {result['raw_confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, raw_color, 2)
        y += 30
        
        # Smoothed confidence
        smooth_color = (0, 255, 0) if result['smoothed_confidence'] < 0.5 else \
                       (0, 165, 255) if result['smoothed_confidence'] < 0.7 else (0, 0, 255)
        cv2.putText(display, f"Smoothed: {result['smoothed_confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, smooth_color, 2)
        y += 30
        
        # Alert level
        alert_color = (0, 255, 0) if result['alert_level'] == 'normal' else \
                      (0, 165, 255) if result['alert_level'] == 'warning' else (0, 0, 255)
        cv2.putText(display, f"Alert: {result['alert_level'].upper()}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        y += 30
        
        # Temporal status
        temporal_status = "READY" if result.get('temporal_ready') else "BUFFERING"
        cv2.putText(display, f"Temporal: {temporal_status}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y += 30
        
        # Keypoints status
        kp_status = "✓" if result.get('keypoints_detected') else "✗"
        cv2.putText(display, f"Keypoints: {kp_status}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y += 35
        
        # Statistics
        cv2.putText(display, f"Seizures: {self.stats['seizure_detections']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Confidence bar (right side)
        bar_x = w - 60
        bar_h = 300
        bar_y = 50
        bar_w = 40
        
        # Background
        cv2.rectangle(display, (bar_x - 5, bar_y - 5), (bar_x + bar_w + 5, bar_y + bar_h + 5), (50, 50, 50), -1)
        
        # Fill based on smoothed confidence
        fill_h = int(bar_h * result['smoothed_confidence'])
        fill_color = (0, 255, 0) if result['smoothed_confidence'] < 0.5 else \
                     (0, 165, 255) if result['smoothed_confidence'] < 0.7 else (0, 0, 255)
        cv2.rectangle(display, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), fill_color, -1)
        
        # Threshold lines
        warning_y = bar_y + int(bar_h * (1 - self.seizure_predictor.warning_threshold))
        alert_y = bar_y + int(bar_h * (1 - self.seizure_predictor.alert_threshold))
        cv2.line(display, (bar_x - 10, warning_y), (bar_x + bar_w + 10, warning_y), (0, 165, 255), 2)
        cv2.line(display, (bar_x - 10, alert_y), (bar_x + bar_w + 10, alert_y), (0, 0, 255), 2)
        
        # Labels
        cv2.putText(display, f"{result['smoothed_confidence']:.1%}", (bar_x - 5, bar_y + bar_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Banner based on alert
        if result['alert_level'] == 'critical':
            cv2.rectangle(display, (0, h-60), (w, h), (0, 0, 255), -1)
            text = f"!!! SEIZURE DETECTED !!! conf={result['smoothed_confidence']:.2%}"
            cv2.putText(display, text, (w//2 - 250, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif result['alert_level'] == 'warning':
            cv2.rectangle(display, (0, h-60), (w, h), (0, 165, 255), -1)
            text = f"⚠️ SEIZURE WARNING conf={result['smoothed_confidence']:.2%}"
            cv2.putText(display, text, (w//2 - 200, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(display, (0, h-40), (w, h), (0, 100, 0), -1)
            cv2.putText(display, f"NORMAL - {result['alert_level']}", (10, h-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display
    
    def run(self):
        """Run seizure detection test"""
        print("\n" + "="*70)
        print("🧠 SEIZURE DETECTION TEST")
        print("="*70)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print(f"💾 Output: {self.output_folder}")
        print(f"🐌 Slow mode: {self.slow_mode}")
        print(f"🚀 Fast mode: {self.fast_mode}")
        print(f"⏱️ Realtime mode: {self.realtime_mode}")
        print(f"🎬 Frame range: {self.start_frame} - {self.end_frame if self.end_frame > 0 else 'END'}")
        print("="*70 + "\n")
        
        if not self.initialize():
            return
        
        # Open video
        camera_config = {
            'video_path': str(self.video_path),
            'camera_id': f'seizure_test_{self.video_name}',
            'loop': False,
            'resolution': None  # 🔥 Keep original video resolution (không resize)
        }
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print("❌ Failed to open video")
            return
        
        total_frames = camera.total_frames
        video_fps = camera.video_fps
        
        # Apply end_frame limit
        effective_end = self.end_frame if self.end_frame > 0 else total_frames
        effective_end = min(effective_end, total_frames)
        
        print(f"▶️ Video: {total_frames} frames, {video_fps:.1f} FPS")
        print(f"▶️ Processing frames: {self.start_frame} to {effective_end}")
        print(f"⌨️ Controls: Q=Quit, SPACE=Pause, S=Save frame")
        print("-"*70 + "\n")
        
        # Skip to start frame
        if self.start_frame > 0:
            print(f"⏩ Skipping to frame {self.start_frame}...")
            camera.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        # Create window if not fast mode
        if not self.fast_mode:
            cv2.namedWindow('Seizure Detection Test', cv2.WINDOW_NORMAL)
        
        # Process loop
        frame_number = self.start_frame
        paused = False
        start_time = time.time()
        frame_start_time = time.time()
        
        try:
            while frame_number < effective_end:
                if not paused:
                    frame_start_time = time.time()
                    frame = camera.get_frame()
                    if frame is None:
                        break
                    
                    frame_number += 1
                    self.stats['total_frames'] = frame_number - self.start_frame
                    timestamp = frame_number / video_fps
                    
                    # Process frame
                    result = self.process_frame(frame, frame_number, timestamp)
                    
                    # Log significant events
                    if result['alert_level'] == 'critical':
                        self.stats['seizure_detections'] += 1
                        event = {
                            'frame': frame_number,
                            'timestamp': timestamp,
                            'raw_confidence': result['raw_confidence'],
                            'smoothed_confidence': result['smoothed_confidence'],
                            'alert_level': result['alert_level'],
                            'bbox': result['bbox']
                        }
                        self.stats['seizure_events'].append(event)
                        
                        # Save frame
                        saved_path = self.save_frame(frame, frame_number, result, prefix="SEIZURE_")
                        print(f"\n🚨 SEIZURE DETECTED at frame {frame_number}!")
                        print(f"   Raw: {result['raw_confidence']:.3f}")
                        print(f"   Smoothed: {result['smoothed_confidence']:.3f}")
                        print(f"   Saved: {saved_path.name}")
                    
                    elif result['alert_level'] == 'warning':
                        self.stats['seizure_warnings'] += 1
                        if self.stats['seizure_warnings'] % 10 == 1:  # Log every 10th warning
                            print(f"   Frame {frame_number}: WARNING - smoothed={result['smoothed_confidence']:.3f}")
                    
                    # Display
                    if not self.fast_mode:
                        display = self.draw_display(frame, result)
                        cv2.imshow('Seizure Detection Test', display)
                    
                    # Progress
                    if frame_number % 100 == 0:
                        progress = (frame_number / total_frames) * 100
                        print(f"⏳ {frame_number}/{total_frames} ({progress:.1f}%) | Seizures: {self.stats['seizure_detections']} | Warnings: {self.stats['seizure_warnings']}")
                
                # Calculate elapsed time for this frame
                frame_elapsed = time.time() - frame_start_time
                
                # Keyboard with proper FPS sync
                if self.fast_mode:
                    key = cv2.waitKey(1) & 0xFF
                elif self.slow_mode:
                    remaining = max(1, int(100 - frame_elapsed * 1000))  # 10 FPS
                    key = cv2.waitKey(remaining) & 0xFF
                elif self.realtime_mode:
                    target_ms = 1000 / video_fps
                    remaining = max(1, int(target_ms - frame_elapsed * 1000))
                    key = cv2.waitKey(remaining) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('s') and not paused:
                    saved = self.save_frame(frame, frame_number, result, prefix="MANUAL_")
                    print(f"💾 Saved: {saved.name}")
        
        finally:
            cv2.destroyAllWindows()
            camera.disconnect()
        
        # Generate report
        processing_time = time.time() - start_time
        self.generate_report(processing_time)
    
    def generate_report(self, processing_time):
        """Generate final report"""
        print("\n" + "="*70)
        print("📊 SEIZURE DETECTION REPORT")
        print("="*70)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print("-"*70)
        print(f"🎬 Total Frames: {self.stats['total_frames']}")
        print(f"👤 Frames with Person: {self.stats['frames_with_person']}")
        print(f"🦴 Frames with Keypoints: {self.stats['frames_with_keypoints']}")
        print("-"*70)
        print(f"🚨 Seizure Detections (critical): {self.stats['seizure_detections']}")
        print(f"⚠️ Seizure Warnings: {self.stats['seizure_warnings']}")
        print(f"📈 Max Confidence: {self.stats['max_confidence']:.3f}")
        print("-"*70)
        
        print("\n📂 Alert Levels breakdown:")
        for level, count in sorted(self.stats['alert_levels'].items(), key=lambda x: -x[1]):
            pct = count / self.stats['total_frames'] * 100 if self.stats['total_frames'] > 0 else 0
            print(f"   {level}: {count} ({pct:.1f}%)")
        
        print("-"*70)
        
        if self.stats['seizure_events']:
            print("\n🚨 Seizure Events:")
            for i, event in enumerate(self.stats['seizure_events']):
                print(f"   [{i+1}] Frame {event['frame']} (t={event['timestamp']:.2f}s)")
                print(f"       Raw: {event['raw_confidence']:.3f}")
                print(f"       Smoothed: {event['smoothed_confidence']:.3f}")
        else:
            print("\n✅ No seizures detected in this video")
        
        print("-"*70)
        print(f"⏱️ Processing Time: {processing_time:.2f}s")
        print(f"📈 Processing FPS: {self.stats['total_frames']/processing_time:.1f}")
        print("="*70)
        
        # Save JSON report
        report = {
            'video': str(self.video_path),
            'video_name': self.video_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_frames': self.stats['total_frames'],
                'frames_with_person': self.stats['frames_with_person'],
                'frames_with_keypoints': self.stats['frames_with_keypoints'],
                'seizure_detections': self.stats['seizure_detections'],
                'seizure_warnings': self.stats['seizure_warnings'],
                'max_confidence': self.stats['max_confidence'],
                'alert_levels': self.stats['alert_levels'],
                'processing_time': processing_time,
                'processing_fps': self.stats['total_frames'] / processing_time
            },
            'seizure_events': self.stats['seizure_events'],
            'settings': {
                'vsvig_threshold': self.seizure_detector.confidence_threshold if self.seizure_detector else None,
                'alert_threshold': self.seizure_predictor.alert_threshold,
                'warning_threshold': self.seizure_predictor.warning_threshold,
                'temporal_window': self.seizure_predictor.temporal_window
            }
        }
        
        report_path = self.output_folder / "seizure_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Report saved: {report_path}")
        print(f"📁 Output folder: {self.output_folder}")
        
        # Save confidence history to CSV
        if self.stats['confidence_history']:
            csv_path = self.output_folder / "confidence_history.csv"
            with open(csv_path, 'w') as f:
                f.write("frame,raw_confidence,smoothed_confidence,alert_level\n")
                for entry in self.stats['confidence_history']:
                    f.write(f"{entry['frame']},{entry['raw']:.4f},{entry['smoothed']:.4f},{entry['alert_level']}\n")
            print(f"📈 Confidence history: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Seizure Detection Only')
    parser.add_argument('--video', type=str, default="1", help='Video name/number (default: 1)')
    parser.add_argument('--slow', action='store_true', help='Slow mode (10 FPS) for analysis')
    parser.add_argument('--fast', action='store_true', help='Fast mode (no display)')
    parser.add_argument('--realtime', action='store_true', help='Realtime mode (sync with video FPS)')
    parser.add_argument('--start', type=int, default=0, help='Start frame (default: 0)')
    parser.add_argument('--end', type=int, default=-1, help='End frame (default: -1 = all)')
    args = parser.parse_args()
    
    if not SEIZURE_AVAILABLE:
        print("❌ Seizure detection modules not available!")
        print("   Please install required dependencies")
        sys.exit(1)
    
    tester = SeizureDetectionTest(
        video_input=args.video,
        slow_mode=args.slow,
        fast_mode=args.fast,
        realtime_mode=args.realtime,
        start_frame=args.start,
        end_frame=args.end
    )
    tester.run()


if __name__ == "__main__":
    main()
