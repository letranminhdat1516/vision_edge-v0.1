#!/usr/bin/env python3
"""
SEIZURE DETECTION REALTIME TEST - WITH VIDEO DISPLAY
Test co giật với hiển thị video realtime và kết quả
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import video camera
from video_camera_service import VideoCameraService

# Import PRODUCTION seizure detection
try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor
    from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator
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


class RealtimeSeizureTest:
    """Test co giật với hiển thị realtime"""
    
    def __init__(self, video_number: int = 1):
        """
        Initialize realtime test
        
        Args:
            video_number: Số video cần test (1-35)
        """
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        
        # Find video
        self.video_path = self.find_video(video_number)
        self.video_name = self.video_path.stem
        
        # Output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.script_dir / "test_results" / f"seizure_video_{video_number}_{timestamp}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'seizure_detections': 0,
            'seizure_alerts': 0,
            'max_confidence': 0.0,
            'alert_frames': []
        }
        
        # Models
        self.person_detector = None
        self.seizure_detector = None
        self.seizure_predictor = None
    
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
                print("   Trying alternate path...")
                yolo_path = Path("yolov8n.pt")
            
            self.person_detector = YOLO(str(yolo_path))
            print(f"✅ YOLO loaded: {yolo_path}")
            
            if not SEIZURE_AVAILABLE:
                print("⚠️ Seizure detection not available - person detection only")
                return True
            
            # 2. VSViG Seizure Detector
            print("🧠 Loading VSViG seizure detector...")
            self.seizure_detector = VSViGSeizureDetector(
                confidence_threshold=0.65,
                device='auto'
            )
            
            # Load models
            if not self.seizure_detector.load_models():
                print("⚠️ VSViG models not loaded - using fallback")
            else:
                print("✅ VSViG loaded successfully")
            
            # 3. Seizure Predictor
            print("📊 Initializing seizure predictor...")
            self.seizure_predictor = SeizurePredictor(
                temporal_window=15,
                alert_threshold=0.70,
                warning_threshold=0.55
            )
            print("✅ Seizure predictor initialized")
            
            print("="*80)
            print("✅ ALL MODELS READY!")
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
            return False
    
    def detect_persons(self, frame: np.ndarray):
        """Detect persons and return bboxes"""
        results = self.person_detector(frame, conf=0.3, classes=[0], verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                persons.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        return persons
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, bbox: list):
        """Draw skeleton keypoints and connections"""
        if keypoints is None or len(keypoints) == 0:
            return
        
        # COCO pose connections
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
                    if start_point[2] > 0.3 and end_point[2] > 0.3:  # confidence threshold
                        cv2.line(frame, 
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                (0, 255, 255), 2)  # Yellow lines
        
        # Draw keypoints
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.3:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 0, 255), -1)  # Magenta points
    
    def draw_results(self, frame: np.ndarray, persons: list, 
                    seizure_result: dict, detection_details: dict, frame_number: int) -> np.ndarray:
        """Draw detection results on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw person bounding boxes and skeleton
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            
            # Color based on seizure detection
            if seizure_result.get('alert_triggered', False):
                color = (0, 0, 255)  # Red for alert
                thickness = 3
            elif seizure_result.get('seizure_detected', False):
                color = (0, 165, 255)  # Orange for warning
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence
            conf_text = f"Person: {person['confidence']:.2f}"
            cv2.putText(display_frame, conf_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw skeleton if available
            if 'keypoints' in person and person['keypoints'] is not None:
                self.draw_skeleton(display_frame, person['keypoints'], person['bbox'])
        
        # Draw detailed info panel (left side)
        panel_width = 400
        panel_height = h
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Title
        y_offset = 25
        cv2.putText(display_frame, "SEIZURE DETECTION ANALYSIS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_offset += 30
        cv2.line(display_frame, (10, y_offset-10), (panel_width-10, y_offset-10), (100, 100, 100), 1)
        
        # Frame info
        cv2.putText(display_frame, f"Frame: {frame_number}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(display_frame, f"Persons Detected: {len(persons)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        cv2.line(display_frame, (10, y_offset-10), (panel_width-10, y_offset-10), (100, 100, 100), 1)
        
        # Seizure Detection Details
        if SEIZURE_AVAILABLE and seizure_result:
            cv2.putText(display_frame, "SEIZURE METRICS:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            y_offset += 25
            raw_conf = seizure_result.get('seizure_confidence', 0.0)
            cv2.putText(display_frame, f"Raw Confidence: {raw_conf:.4f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 20
            smooth_conf = seizure_result.get('smoothed_confidence', 0.0)
            conf_color = (0, 255, 0) if smooth_conf < 0.55 else (0, 165, 255) if smooth_conf < 0.70 else (0, 0, 255)
            cv2.putText(display_frame, f"Smoothed Confidence: {smooth_conf:.4f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)
            
            # Confidence bar)
        stats_y = h - 100
        cv2.rectangle(overlay, (0, stats_y-10), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        cv2.putText(display_frame, "CUMULATIVE STATISTICS:", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        stats_y += 25
        cv2.putText(display_frame, f"Total Seizure Detections: {self.stats['seizure_detections']} | "
                   f"Alerts Triggered: {self.stats['seizure_alerts']} | "
                   f"Max Confidence: {self.stats['max_confidence']:.3f}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        stats_y += 25
        cv2.putText(display_frame, f"Frames Processed: {self.stats['total_frames']} | "
                   f"Frames with Person: {self.stats['frames_with_person']}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return display_frame
    
    def save_alert_frame(self, frame: np.ndarray, frame_number: int, confidence: float):
        """Save alert frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"seizure_alert_frame_{frame_number}_{timestamp}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), frame)
        logger.info(f"🚨 Alert saved: {filename}")
            
            y_offset += 25
            if detection_details.get('temporal_ready', False):
                cv2.putText(display_frame, "[OK] Temporal Analysis Ready", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "[..] Building Temporal Buffer", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            y_offset += 20
            if 'keypoints' in detection_details and detection_details['keypoints'] is not None:
                cv2.putText(display_frame, "[OK] Pose Keypoints Extracted", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "[X] No Pose Keypoints", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            y_offset += 20
            if detection_details.get('vsvig_processed', False):
                cv2.putText(display_frame, "[OK] VSViG Model Processed", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "[..] Waiting for VSViG", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            y_offset += 30
            cv2.line(display_frame, (10, y_offset-10), (panel_width-10, y_offset-10), (100, 100, 100), 1)
            
            # Movement indicators
            cv2.putText(display_frame, "SEIZURE INDICATORS:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            y_offset += 25
            if smooth_conf > 0.70:
                cv2.putText(display_frame, "[!!!] Abnormal Movement Pattern", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif smooth_conf > 0.55:
                cv2.putText(display_frame, "[!] Suspicious Movement", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            else:
                cv2.putText(display_frame, "[OK] Normal Movement", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            y_offset += 20
            if smooth_conf > 0.40:
                cv2.putText(display_frame, "[+] Rapid Body Oscillation", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            
            y_offset += 20
            if smooth_conf > 0.30:
                cv2.putText(display_frame, "[+] Jerky Limb Movement", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            
            y_offset += 20
            if detection_details.get('temporal_ready', False):
                cv2.putText(display_frame, "[+] Consistent Pattern", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        
        # Alert banner
        if seizure_result.get('alert_triggered', False):
            banner_text = "!!! SEIZURE ALERT !!!"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_BOLD, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            
            # Flashing effect
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(display_frame, (text_x-10, h-80), 
                            (text_x + text_size[0]+10, h-20), (0, 0, 255), -1)
                cv2.putText(display_frame, banner_text, (text_x, h-40),
                           cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3)
        
        # Statistics panel (bottom right)
        stats_x = w - 300
        stats_y = h - 120
        cv2.rectangle(overlay, (stats_x-10, stats_y-10), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        cv2.putText(display_frame, "STATISTICS:", (stats_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        stats_y += 25
        cv2.putText(display_frame, f"Detections: {self.stats['seizure_detections']}", 
                   (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        stats_y += 20
        cv2.putText(display_frame, f"Alerts: {self.stats['seizure_alerts']}", 
                   (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        stats_y += 20
        cv2.putText(display_frame, f"Max Conf: {self.stats['max_confidence']:.3f}", 
                   (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def save_alert_frame(self, frame: np.ndarray, frame_number: int, confidence: float):
        """Save alert frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"seizure_alert_frame_{frame_number}_{timestamp}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), frame)
        logger.info(f"🚨 Alert saved: {filename}")
    
    def run(self):
        """Run realtime test"""
        print("\n" + "="*80)
        print(f"🎬 SEIZURE DETECTION REALTIME TEST")
        print("="*80)
        print(f"📹 Video: {self.video_name}")
        print(f"📁 Path: {self.video_path}")
        print(f"💾 Output: {self.output_folder}")
        print("="*80)
        print("\n⌨️ Controls:")
        print("   SPACE - Pause/Resume")
        print("   Q - Quit")
        print("   S - Save current frame")
        print("="*80 + "\n")
        
        # Initialize models
        if not self.initialize_models():
            print("❌ Model initialization failed")
            return
        
        # Setup camera
        camera_config = {
            'video_path': str(self.video_path),
            'camera_id': f'test_video_{self.video_name}',
            'camera_name': self.video_name,
            'loop': False,
            'fps': 30
        }
        
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print(f"❌ Failed to open video")
            return
        
        # Get video info
        total_frames = camera.total_frames
        video_fps = camera.video_fps
        
        print(f"▶️ Playing video...")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {video_fps:.2f}")
        print(f"   Duration: {total_frames / video_fps:.2f}s\n")
        
        # Processing loop
        frame_number = 0
        paused = False
        start_time = time.time()
        
        # Create window with original size (no resize)
        cvGet first frame to determine size
        first_frame = camera.get_frame()
        if first_frame is None:
            print("❌ Cannot read first frame")
            return
        
        frame_h, frame_w = first_frame.shape[:2]
        
        # Reset video to start
        camera.disconnect()
        camera.connect()
        
        # Create window with EXACT original video size
        cv2.namedWindow('Seizure Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Seizure Detection Test', frame_w, frame_h)
        
        print(f"📺 Window created with exact size: {frame_w}x{frame_h}\n"
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
                    
                    if len(persons) > 0:
                        self.stats['frames_with_person'] += 1
                    
                    # Seizure detection
                    seizure_result = {}
                    if SEIZURE_AVAILABLE and self.seizure_detector and len(persons) > 0:
                    detection_details = {
                        'temporal_ready': False,
                        'keypoints': None,
                        'vsvig_processed': False
                    }
                    
                    if SEIZURE_AVAILABLE and self.seizure_detector and len(persons) > 0:
                        for idx, person in enumerate(persons):
                            bbox = person['bbox']
                            
                            # VSViG detection
                            detection = self.seizure_detector.detect_seizure(frame, bbox)
                            
                            # Store detection details
                            detection_details['temporal_ready'] = detection.get('temporal_ready', False)
                            detection_details['keypoints'] = detection.get('keypoints')
                            detection_details['vsvig_processed'] = True
                            
                            # Add keypoints to person for skeleton drawing
                            if detection.get('keypoints') is not None:
                                person['keypoints'] = detection['keypoints']
                            
                            if detection.get('temporal_ready', False):
                                confidence = detection.get('confidence', 0.0)
                                
                                # Update predictor
                                prediction = self.seizure_predictor.update_prediction(confidence)
                                
                                seizure_result = {
                                    'seizure_confidence': confidence,
                                    'smoothed_confidence': prediction['smoothed_confidence'],
                                    'alert_level': prediction['alert_level'],
                                    'seizure_detected': prediction.get('seizure_detected', False),
                                    'alert_triggered': prediction['alert_level'] in ['alert', 'critical']
                                }
                                
                                # Update stats
                                if seizure_result['seizure_detected']:
                                    self.stats['seizure_detections'] += 1
                                
                                if seizure_result['alert_triggered']:
                                    if frame_number not in self.stats['alert_frames']:
                                        self.stats['seizure_alerts'] += 1
                                        self.stats['alert_frames'].append(frame_number)
                                        self.save_alert_frame(frame, frame_number, confidence)
                                
                                if confidence > self.stats['max_confidence']:
                                    self.stats['max_confidence'] = confidence
                    
                    # Draw results
                    display_frame = self.draw_results(frame, persons, seizure_result, detection_details
                    # Show frame
                    cv2.imshow('Seizure Detection Test', display_frame)
                    
                    # Progress
                    if frame_number % 100 == 0:
                        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                        print(f"⏳ Frame {frame_number}/{total_frames} ({progress:.1f}%) | "
                              f"Alerts: {self.stats['seizure_alerts']}")
                
                # Handle keyboard
                key = cv2.waitKey(1 if not paused else 100) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord('s'):  # S
                    save_path = self.output_folder / f"manual_save_frame_{frame_number}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    print(f"💾 Frame saved: {save_path.name}")
        
        finally:
            cv2.destroyAllWindows()
            camera.disconnect()
        
        # Final statistics
        processing_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS")
        print("="*80)
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Frames with Person: {self.stats['frames_with_person']}")
        print(f"Seizure Detections: {self.stats['seizure_detections']}")
        print(f"🚨 Seizure Alerts: {self.stats['seizure_alerts']}")
        print(f"Max Confidence: {self.stats['max_confidence']:.3f}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Processing FPS: {frame_number / processing_time:.2f}")
        print(f"📁 Output: {self.output_folder}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test seizure detection với video display')
    parser.add_argument('--video', type=int, default=1,
                       help='Video number to test (1-35, default: 1)')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = RealtimeSeizureTest(video_number=args.video)
    tester.run()


if __name__ == "__main__":
    main()
