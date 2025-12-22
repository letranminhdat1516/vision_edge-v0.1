#!/usr/bin/env python3
"""
FALL DETECTION ONLY - Test Script
Chỉ test Fall Detection, không seizure
Focus vào video "chung" để phân tích chi tiết

Usage:
    python test_fall_only.py --video chung
    python test_fall_only.py --video chung --slow    # Chậm để phân tích
    python test_fall_only.py --video chung --fast    # Nhanh, không hiển thị
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
# IMPORT FALL DETECTION MODULE
# ============================================================================

try:
    from fall_detection.simple_fall_detector import SimpleFallDetector
    from ultralytics import YOLO
    AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Modules not available: {e}")
    AVAILABLE = False

# Logging will be setup in class with file handler
logger = logging.getLogger(__name__)


class FallDetectionTest:
    """Test Fall Detection với video - Chi tiết từng frame"""
    
    def __init__(self, video_input: str = "chung", slow_mode: bool = False, fast_mode: bool = False, realtime_mode: bool = False):
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        
        # Find video FIRST (needed for output folder)
        self.video_path = self.find_video(video_input)
        self.video_name = self.video_path.stem
        
        # Output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.script_dir / "test_results" / f"fall_only_{self.video_name}_{timestamp}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 🆕 Setup logging với FILE HANDLER
        self.log_file = self.output_folder / "fall_detection.log"
        self._setup_logging()
        
        # Settings
        self.slow_mode = slow_mode  # Chậm để phân tích
        self.fast_mode = fast_mode  # Nhanh, không hiển thị
        self.realtime_mode = realtime_mode  # Sync với video FPS thật
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'fall_detections': 0,
            'fall_events': [],  # Chi tiết từng event
            'max_confidence': 0.0,
            'categories': {},  # Đếm theo category
            'bbox_history': [],  # Lưu lịch sử bbox
        }
        
        # Components
        self.person_detector = None
        self.fall_detector = None
        self.previous_frame = None
        
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
        # Check exact name first
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
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
        print("🔧 INITIALIZING FALL DETECTION")
        print("="*60)
        
        try:
            # Person detector
            print("👤 Loading person detector (YOLOv8n)...")
            model_path = PROJECT_ROOT / "yolov8n.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "src" / "yolov8s.pt"
            self.person_detector = YOLO(str(model_path))
            print(f"   ✅ Loaded: {model_path.name}")
            
            # Fall detector
            print("🚨 Loading fall detector...")
            self.fall_detector = SimpleFallDetector(confidence_threshold=0.28)
            print(f"   ✅ Loaded (threshold: {self.fall_detector.confidence_threshold})")
            
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Init error: {e}")
            return False
    
    def calculate_motion(self, frame) -> float:
        """Calculate motion level between frames"""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.previous_frame, gray)
        motion = np.mean(diff) / 255.0
        self.previous_frame = gray
        return motion
    
    def detect_person(self, frame) -> dict:
        """Detect person in frame"""
        results = self.person_detector(frame, classes=[0], verbose=False)
        
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
        """Process single frame for fall detection"""
        result = {
            'frame': frame_number,
            'timestamp': timestamp,
            'person_detected': False,
            'fall_detected': False,
            'confidence': 0.0,
            'category': 'no-person',
            'bbox': None,
            'motion_level': 0.0
        }
        
        # Calculate motion
        motion = self.calculate_motion(frame)
        result['motion_level'] = motion
        
        # Detect person
        person = self.detect_person(frame)
        if person is None:
            return result
        
        result['person_detected'] = True
        result['bbox'] = person['bbox'].tolist()
        self.stats['frames_with_person'] += 1
        
        # Detect fall
        fall_result = self.fall_detector.detect_fall(
            frame,
            timestamp=timestamp,
            person_bbox=person['bbox'],
            motion_level=motion
        )
        
        result['fall_detected'] = fall_result.get('fall_detected', False)
        result['confidence'] = fall_result.get('confidence', 0.0)
        result['category'] = fall_result.get('category', 'no-fall')
        result['method'] = fall_result.get('method', 'unknown')
        result['fall_type'] = fall_result.get('fall_type', None)
        result['alert_level'] = fall_result.get('alert_level', None)
        
        # Track max confidence
        if result['confidence'] > self.stats['max_confidence']:
            self.stats['max_confidence'] = result['confidence']
        
        # Track categories
        cat = result['category']
        self.stats['categories'][cat] = self.stats['categories'].get(cat, 0) + 1
        
        # Save bbox history
        self.stats['bbox_history'].append({
            'frame': frame_number,
            'bbox': result['bbox'],
            'motion': motion
        })
        
        return result
    
    def save_frame(self, frame, frame_number, result, prefix=""):
        """Save frame with annotations"""
        # Draw annotations
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw bbox
        if result['bbox']:
            x1, y1, x2, y2 = [int(v) for v in result['bbox']]
            color = (0, 0, 255) if result['fall_detected'] else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 5, (255, 0, 255), -1)
            
            # Bbox info
            bw, bh = x2 - x1, y2 - y1
            aspect = bw / bh if bh > 0 else 0
            cv2.putText(display, f"W:{bw} H:{bh} AR:{aspect:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Info panel
        y = 30
        cv2.putText(display, f"Frame: {frame_number}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
        cv2.putText(display, f"Motion: {result['motion_level']:.4f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(display, f"Category: {result['category']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(display, f"Confidence: {result['confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Fall banner
        if result['fall_detected']:
            cv2.rectangle(display, (0, h-50), (w, h), (0, 0, 255), -1)
            cv2.putText(display, f"FALL DETECTED! conf={result['confidence']:.2f}",
                       (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save
        filename = f"{prefix}frame_{frame_number:05d}_{result['category']}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), display)
        
        return filepath
    
    def run(self):
        """Run fall detection test"""
        print("\n" + "="*70)
        print("🎬 FALL DETECTION TEST")
        print("="*70)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print(f"💾 Output: {self.output_folder}")
        print(f"🐌 Slow mode: {self.slow_mode}")
        print(f"🚀 Fast mode: {self.fast_mode}")
        print("="*70 + "\n")
        
        if not self.initialize():
            return
        
        # Open video
        camera_config = {
            'video_path': str(self.video_path),
            'camera_id': f'fall_test_{self.video_name}',
            'loop': False
        }
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print("❌ Failed to open video")
            return
        
        total_frames = camera.total_frames
        video_fps = camera.video_fps
        
        print(f"▶️ Video: {total_frames} frames, {video_fps:.1f} FPS")
        print(f"⌨️ Controls: Q=Quit, SPACE=Pause, S=Save frame")
        print("-"*70 + "\n")
        
        # Create window if not fast mode
        if not self.fast_mode:
            cv2.namedWindow('Fall Detection Test', cv2.WINDOW_NORMAL)
        
        # Process loop
        frame_number = 0
        paused = False
        start_time = time.time()
        
        try:
            while True:
                if not paused:
                    frame_start_time = time.time()  # Track frame processing time
                    frame = camera.get_frame()
                    if frame is None:
                        break
                    
                    frame_number += 1
                    self.stats['total_frames'] = frame_number
                    timestamp = frame_number / video_fps
                    
                    # Process frame
                    result = self.process_frame(frame, frame_number, timestamp)
                    
                    # Log interesting events
                    if result['fall_detected']:
                        self.stats['fall_detections'] += 1
                        event = {
                            'frame': frame_number,
                            'timestamp': timestamp,
                            'confidence': result['confidence'],
                            'category': result['category'],
                            'method': result.get('method'),
                            'bbox': result['bbox'],
                            'motion': result['motion_level']
                        }
                        self.stats['fall_events'].append(event)
                        
                        # Save frame
                        saved_path = self.save_frame(frame, frame_number, result, prefix="FALL_")
                        print(f"\n🚨 FALL DETECTED at frame {frame_number}!")
                        print(f"   Confidence: {result['confidence']:.3f}")
                        print(f"   Category: {result['category']}")
                        print(f"   Saved: {saved_path.name}")
                    
                    # Log filtered events (để phân tích tại sao không detect)
                    elif result['category'] not in ['no-fall', 'no-person']:
                        print(f"   Frame {frame_number}: FILTERED - {result['category']} (motion={result['motion_level']:.4f})")
                    
                    # Display
                    if not self.fast_mode:
                        display = self.draw_display(frame, result)
                        cv2.imshow('Fall Detection Test', display)
                    
                    # Progress
                    if frame_number % 100 == 0:
                        progress = (frame_number / total_frames) * 100
                        print(f"⏳ {frame_number}/{total_frames} ({progress:.1f}%) | Falls: {self.stats['fall_detections']}")
                
                # Calculate elapsed time for this frame
                frame_elapsed = time.time() - frame_start_time
                
                # Keyboard with proper FPS sync
                if self.fast_mode:
                    key = cv2.waitKey(1) & 0xFF
                elif self.slow_mode:
                    # 10 FPS = 100ms per frame, subtract processing time
                    remaining = max(1, int(100 - frame_elapsed * 1000))
                    key = cv2.waitKey(remaining) & 0xFF
                elif self.realtime_mode:
                    # Sync với video FPS thật (30 FPS = 33ms per frame)
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
    
    def draw_display(self, frame, result) -> np.ndarray:
        """Draw display frame with info"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw bbox
        if result['bbox']:
            x1, y1, x2, y2 = [int(v) for v in result['bbox']]
            color = (0, 0, 255) if result['fall_detected'] else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            
            # Center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 8, (255, 0, 255), -1)
            
            # Dimensions
            bw, bh = x2 - x1, y2 - y1
            aspect = bw / bh if bh > 0 else 0
            
            # Position info
            cv2.putText(display, f"W:{bw} H:{bh}", (x1, y1-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, f"AR:{aspect:.2f} CY:{cy}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Info panel (left)
        panel_w = 350
        cv2.rectangle(display, (0, 0), (panel_w, 200), (0, 0, 0), -1)
        
        y = 30
        cv2.putText(display, f"Frame: {result['frame']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35
        cv2.putText(display, f"Motion: {result['motion_level']:.4f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y += 30
        cv2.putText(display, f"Category: {result['category']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y += 30
        
        conf_color = (0, 0, 255) if result['confidence'] >= 0.6 else \
                     (0, 165, 255) if result['confidence'] >= 0.4 else (100, 100, 100)
        cv2.putText(display, f"Confidence: {result['confidence']:.3f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        y += 35
        cv2.putText(display, f"Falls: {self.stats['fall_detections']}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Banner
        if result['fall_detected']:
            cv2.rectangle(display, (0, h-60), (w, h), (0, 0, 255), -1)
            text = f"!!! FALL DETECTED !!! conf={result['confidence']:.2f}"
            cv2.putText(display, text, (w//2 - 250, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(display, (0, h-40), (w, h), (0, 100, 0), -1)
            cv2.putText(display, f"NORMAL - {result['category']}", (10, h-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display
    
    def generate_report(self, processing_time):
        """Generate final report"""
        print("\n" + "="*70)
        print("📊 FALL DETECTION REPORT")
        print("="*70)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print("-"*70)
        print(f"🎬 Total Frames: {self.stats['total_frames']}")
        print(f"👤 Frames with Person: {self.stats['frames_with_person']}")
        print(f"🚨 Fall Detections: {self.stats['fall_detections']}")
        print(f"📈 Max Confidence: {self.stats['max_confidence']:.3f}")
        print("-"*70)
        
        print("\n📂 Categories breakdown:")
        for cat, count in sorted(self.stats['categories'].items(), key=lambda x: -x[1]):
            pct = count / self.stats['total_frames'] * 100 if self.stats['total_frames'] > 0 else 0
            print(f"   {cat}: {count} ({pct:.1f}%)")
        
        print("-"*70)
        
        if self.stats['fall_events']:
            print("\n🚨 Fall Events:")
            for i, event in enumerate(self.stats['fall_events']):
                print(f"   [{i+1}] Frame {event['frame']} (t={event['timestamp']:.2f}s)")
                print(f"       Confidence: {event['confidence']:.3f}")
                print(f"       Category: {event['category']}")
                print(f"       Motion: {event['motion']:.4f}")
        else:
            print("\n✅ No falls detected in this video")
        
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
                'fall_detections': self.stats['fall_detections'],
                'max_confidence': self.stats['max_confidence'],
                'categories': self.stats['categories'],
                'processing_time': processing_time,
                'processing_fps': self.stats['total_frames'] / processing_time
            },
            'fall_events': self.stats['fall_events'],
            'settings': {
                'confidence_threshold': self.fall_detector.confidence_threshold,
                'min_time_interval': self.fall_detector.min_time_interval,
                'max_buffer_size': self.fall_detector.max_buffer_size
            }
        }
        
        report_path = self.output_folder / "fall_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 Report saved: {report_path}")
        print(f"📁 Output folder: {self.output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Test Fall Detection Only')
    parser.add_argument('--video', type=str, default="chung", help='Video name (default: chung)')
    parser.add_argument('--slow', action='store_true', help='Slow mode (10 FPS) for analysis')
    parser.add_argument('--fast', action='store_true', help='Fast mode (no display)')
    parser.add_argument('--realtime', action='store_true', help='Realtime mode (sync with video FPS)')
    args = parser.parse_args()
    
    tester = FallDetectionTest(
        video_input=args.video,
        slow_mode=args.slow,
        fast_mode=args.fast,
        realtime_mode=args.realtime
    )
    tester.run()


if __name__ == "__main__":
    main()
