#!/usr/bin/env python3
"""
Advanced Healthcare Monitor - Dual Detection System
Tích hợp: Fall Detection + Seizure Detection + Real-time Statistics
Features: IMOU Camera + YOLO + Fall Detection + VSViG Seizure Detection
"""

import sys
import os
import time
import cv2
import threading
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from camera.simple_camera import SimpleIMOUCamera
from video_processing.simple_processing import IntegratedVideoProcessor
from fall_detection.simple_fall_detector_v2 import SimpleFallDetector
from seizure_detection.vsvig_detector import VSViGSeizureDetector
from seizure_detection.seizure_predictor import SeizurePredictor


class AdvancedHealthcareMonitor:
    """
    Advanced Healthcare Monitor với Dual Detection System
    Features: Fall Detection + Seizure Detection + Keypoint Visualization
    """
    
    def __init__(self, show_keypoints: bool = True, show_statistics: bool = True):
        """
        Initialize advanced healthcare monitoring system
        
        Args:
            show_keypoints: Whether to display pose keypoints on video
            show_statistics: Whether to display real-time statistics
        """
        # Setup logging
        self.setup_logging()
        
        self.logger.info("🏥 Initializing Advanced Healthcare Monitor - Dual Detection System")
        print("[🏥] Initializing Advanced Healthcare Monitor - Dual Detection System")
        
        # Display settings
        self.show_keypoints = show_keypoints
        self.show_statistics = show_statistics
        self.show_dual_windows = True  # Always show both windows
        
        # Camera configuration cho IMOU
        self.camera_config = {
            'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
            'buffer_size': 1,
            'fps': 15,
            'resolution': (640, 480),
            'auto_reconnect': True
        }
        
        # Initialize components
        self.camera = None
        self.video_processor = None
        self.fall_detector = None
        self.seizure_detector = None
        self.seizure_predictor = None
        self.running = False
        
        # Dual detection statistics
        self.stats = {
            # General stats
            'frames_processed': 0,
            'persons_detected': 0,
            'total_frames': 0,
            'keyframes_detected': 0,
            'motion_frames': 0,
            'start_time': time.time(),
            'last_detection_time': None,
            
            # Fall detection stats
            'fall_detections': 0,
            'last_fall_time': None,
            'fall_confidence_avg': 0.0,
            'fall_false_positives': 0,
            
            # Seizure detection stats
            'seizure_detections': 0,
            'last_seizure_time': None,
            'seizure_confidence_avg': 0.0,
            'seizure_warnings': 0,
            'pose_extraction_failures': 0,
            
            # Emergency stats
            'critical_alerts': 0,
            'total_alerts': 0,
            'last_alert_time': None,
            'alert_type': 'normal'
        }
        
        # Performance tracking
        self.performance = {
            'fps': 0.0,
            'processing_time': 0.0,
            'fall_detection_time': 0.0,
            'seizure_detection_time': 0.0,
            'total_detection_time': 0.0
        }
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('AdvancedHealthcareMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = log_dir / f"advanced_healthcare_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Advanced logging system initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            print("[🏥] Initializing system components...")
            
            # Initialize camera
            print("📹 Initializing IMOU camera...")
            self.camera = SimpleIMOUCamera(self.camera_config)
            if not self.camera.connect():
                self.logger.error("Failed to connect to IMOU camera")
                return False
            
            # Initialize video processor
            print("🎬 Initializing video processor...")
            processor_config = 120  # Motion threshold as integer
            self.video_processor = IntegratedVideoProcessor(processor_config)
            
            # Initialize fall detector
            print("🩹 Initializing fall detection...")
            self.fall_detector = SimpleFallDetector(
                confidence_threshold=0.7
            )
            
            # Initialize seizure detector
            print("🧠 Initializing seizure detection...")
            try:
                self.seizure_detector = VSViGSeizureDetector(
                    confidence_threshold=0.65  # Giảm để dễ detect hơn
                )
                self.seizure_predictor = SeizurePredictor(
                    temporal_window=30,
                    alert_threshold=0.65,      # Giảm để dễ detect
                    warning_threshold=0.45     # Giảm để sensitive
                )
                print("✅ Seizure detection initialized (models loading on first use)")
            except Exception as e:
                print(f"⚠️ Seizure detection initialization failed: {e}")
                print("🔧 Continuing with fall detection only...")
                self.seizure_detector = None
                self.seizure_predictor = None
            
            print("✅ All components initialized successfully!")
            self.logger.info("All healthcare monitoring components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    def process_dual_detection(self, frame: np.ndarray, person_detections: list) -> Dict:
        """
        Process dual detection (fall + seizure) for healthcare monitoring
        
        Args:
            frame: Input frame
            person_detections: List of person detections from YOLO
            
        Returns:
            dict: Dual detection results
        """
        start_time = time.time()
        
        result = {
            'fall_detected': False,
            'fall_confidence': 0.0,
            'seizure_detected': False,
            'seizure_confidence': 0.0,
            'seizure_ready': False,
            'alert_level': 'normal',
            'keypoints': None,
            'emergency_type': None
        }
        
        if not person_detections:
            return result
        
        # Get primary person (largest detection)
        primary_person = max(person_detections, key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
        person_bbox = [
            int(primary_person['bbox'][0]),
            int(primary_person['bbox'][1]),
            int(primary_person['bbox'][0] + primary_person['bbox'][2]),
            int(primary_person['bbox'][1] + primary_person['bbox'][3])
        ]
        
        # Fall detection
        fall_start = time.time()
        try:
            fall_result = self.fall_detector.detect_fall(frame, primary_person)
            result['fall_detected'] = fall_result['fall_detected']
            result['fall_confidence'] = fall_result['confidence']
            
            if fall_result['fall_detected']:
                self.stats['fall_detections'] += 1
                self.stats['last_fall_time'] = time.time()
                self.logger.info(f"🩹 Fall detected! Confidence: {fall_result['confidence']:.2f}")
                
        except Exception as e:
            self.logger.error(f"Fall detection error: {str(e)}")
        
        self.performance['fall_detection_time'] = time.time() - fall_start
        
        # Seizure detection
        seizure_start = time.time()
        if self.seizure_detector is not None:
            try:
                seizure_result = self.seizure_detector.detect_seizure(frame, person_bbox)
                result['seizure_ready'] = seizure_result['temporal_ready']
                result['keypoints'] = seizure_result['keypoints']
                
                if seizure_result['temporal_ready']:
                    # Update seizure predictor
                    pred_result = self.seizure_predictor.update_prediction(
                        seizure_result['confidence']
                    )
                    
                    result['seizure_confidence'] = pred_result['smoothed_confidence']
                    result['seizure_detected'] = pred_result['seizure_detected']
                    
                    if pred_result['seizure_detected']:
                        self.stats['seizure_detections'] += 1
                        self.stats['last_seizure_time'] = time.time()
                        self.logger.info(f"🧠 Seizure detected! Confidence: {pred_result['smoothed_confidence']:.2f}")
                    
                    elif pred_result['alert_level'] == 'warning':
                        self.stats['seizure_warnings'] += 1
                
            except Exception as e:
                self.logger.error(f"Seizure detection error: {str(e)}")
                self.stats['pose_extraction_failures'] += 1
        
        self.performance['seizure_detection_time'] = time.time() - seizure_start
        
        # Determine overall alert level
        if result['seizure_detected']:
            result['alert_level'] = 'critical'
            result['emergency_type'] = 'seizure'
            self.stats['critical_alerts'] += 1
        elif result['fall_detected']:
            result['alert_level'] = 'high'
            result['emergency_type'] = 'fall'
        elif result['seizure_confidence'] > 0.4:
            result['alert_level'] = 'warning'
            result['emergency_type'] = 'seizure_warning'
            
        if result['alert_level'] != 'normal':
            self.stats['total_alerts'] += 1
            self.stats['last_alert_time'] = time.time()
            self.stats['alert_type'] = result['alert_level']
        
        self.performance['total_detection_time'] = time.time() - start_time
        return result
    
    def visualize_dual_detection(self, frame: np.ndarray, detection_result: Dict, person_detections: list) -> np.ndarray:
        """
        Visualize dual detection results on frame
        
        Args:
            frame: Input frame
            detection_result: Results from dual detection
            person_detections: Person detections
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        frame_vis = frame.copy()
        
        # Draw person detections
        for person in person_detections:
            bbox = person['bbox']
            confidence = person['confidence']
            
            # Person bounding box
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)  # Green for person
            
            # Change color based on alerts
            if detection_result['alert_level'] == 'critical':
                color = (0, 0, 255)  # Red for critical
            elif detection_result['alert_level'] == 'high':
                color = (0, 165, 255)  # Orange for high
            elif detection_result['alert_level'] == 'warning':
                color = (0, 255, 255)  # Yellow for warning
            
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_vis, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw keypoints if enabled and available
        if self.show_keypoints and detection_result['keypoints'] is not None:
            keypoints = detection_result['keypoints']
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:
                    color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                    cv2.circle(frame_vis, (int(x), int(y)), 3, color, -1)
                    
                    # Add keypoint index
                    cv2.putText(frame_vis, str(i), (int(x), int(y-5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw skeleton connections
            connections = [
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # arms
                (11, 12), (11, 13), (12, 14)  # legs
            ]
            
            for p1, p2 in connections:
                if (p1 < len(keypoints) and p2 < len(keypoints) and 
                    keypoints[p1, 2] > 0.3 and keypoints[p2, 2] > 0.3):
                    pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
                    pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
                    cv2.line(frame_vis, pt1, pt2, (255, 255, 0), 2)
        
        # Add detection alerts
        alert_y = 30
        
        # Fall detection status
        if detection_result['fall_detected']:
            cv2.putText(frame_vis, f"🩹 FALL DETECTED: {detection_result['fall_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        
        # Seizure detection status
        if detection_result['seizure_detected']:
            cv2.putText(frame_vis, f"🧠 SEIZURE DETECTED: {detection_result['seizure_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        elif detection_result['seizure_confidence'] > 0.4:
            cv2.putText(frame_vis, f"⚠️ Seizure Warning: {detection_result['seizure_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 30
        
        # Seizure buffer status
        if self.seizure_detector and not detection_result['seizure_ready']:
            buffer_size = len(self.seizure_detector.frame_buffer)
            frames_needed = 30 - buffer_size
            cv2.putText(frame_vis, f"📊 Seizure Buffer: {frames_needed} frames needed",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame_vis
    
    def draw_statistics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw real-time statistics overlay"""
        if not self.show_statistics:
            return frame
        
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        
        # Statistics panel
        panel_width = 300
        panel_height = 350
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame_vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        frame_vis = cv2.addWeighted(frame_vis, 0.7, overlay, 0.3, 0)
        
        # Calculate runtime statistics
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['total_frames'] / runtime if runtime > 0 else 0
        
        # Processing efficiency
        efficiency = (1 - self.stats['frames_processed'] / max(self.stats['total_frames'], 1)) * 100
        
        # Statistics text
        stats_text = [
            "🏥 DUAL DETECTION SYSTEM",
            f"Runtime: {runtime/60:.1f} minutes",
            f"FPS: {fps:.1f}",
            f"Efficiency: {efficiency:.1f}% skipped",
            "",
            "📊 DETECTION STATS:",
            f"Total Frames: {self.stats['total_frames']}",
            f"Processed: {self.stats['frames_processed']}",
            f"Keyframes: {self.stats['keyframes_detected']}",
            f"Persons: {self.stats['persons_detected']}",
            "",
            "🩹 FALL DETECTION:",
            f"Falls Detected: {self.stats['fall_detections']}",
            f"Avg Confidence: {self.stats['fall_confidence_avg']:.2f}",
            "",
            "🧠 SEIZURE DETECTION:",
            f"Seizures: {self.stats['seizure_detections']}",
            f"Warnings: {self.stats['seizure_warnings']}",
            f"Pose Failures: {self.stats['pose_extraction_failures']}",
            "",
            "🚨 EMERGENCY ALERTS:",
            f"Critical: {self.stats['critical_alerts']}",
            f"Total Alerts: {self.stats['total_alerts']}",
            f"Status: {self.stats['alert_type']}",
            "",
            "⚡ PERFORMANCE:",
            f"Fall Det: {self.performance['fall_detection_time']*1000:.1f}ms",
            f"Seizure Det: {self.performance['seizure_detection_time']*1000:.1f}ms",
            f"Total: {self.performance['total_detection_time']*1000:.1f}ms"
        ]
        
        # Draw statistics
        text_y = panel_y + 20
        for line in stats_text:
            if line == "":
                text_y += 5
                continue
                
            # Choose color based on content
            color = (255, 255, 255)  # White default
            if "🚨" in line or "CRITICAL" in line:
                color = (0, 0, 255)  # Red
            elif "🩹" in line or "🧠" in line:
                color = (0, 255, 255)  # Yellow
            elif "📊" in line:
                color = (0, 255, 0)  # Green
            
            cv2.putText(frame_vis, line, (panel_x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            text_y += 15
        
        return frame_vis
    
    def create_normal_camera_window(self, frame: np.ndarray, person_detections: list) -> np.ndarray:
        """
        Create normal camera window with minimal overlay
        
        Args:
            frame: Input frame
            person_detections: Person detections
            
        Returns:
            np.ndarray: Frame with minimal visualization
        """
        frame_normal = frame.copy()
        
        # Draw only basic person detections (no keypoints)
        for person in person_detections:
            bbox = person['bbox']
            confidence = person['confidence']
            
            # Person bounding box - simple green
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)  # Always green for normal view
            
            cv2.rectangle(frame_normal, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_normal, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add simple title
        cv2.putText(frame_normal, "Normal Camera View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add basic status
        person_count = len(person_detections)
        status_text = f"Persons Detected: {person_count}"
        cv2.putText(frame_normal, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_normal, timestamp, (10, frame_normal.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_normal
    
    def update_statistics(self, detection_result: Dict, person_count: int):
        """Update system statistics"""
        self.stats['frames_processed'] += 1
        self.stats['persons_detected'] += person_count
        
        if person_count > 0:
            self.stats['last_detection_time'] = time.time()
        
        # Update confidence averages only when there are actual detections
        # Fall detection average
        if detection_result.get('fall_detected', False) and detection_result.get('fall_confidence', 0) > 0:
            current_avg = self.stats['fall_confidence_avg']
            current_count = self.stats['fall_detections']
            new_confidence = detection_result['fall_confidence']
            
            if current_count == 0:
                self.stats['fall_confidence_avg'] = new_confidence
            else:
                self.stats['fall_confidence_avg'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
        
        # Seizure detection average  
        if detection_result.get('seizure_detected', False) and detection_result.get('seizure_confidence', 0) > 0:
            current_avg = self.stats['seizure_confidence_avg']
            current_count = self.stats['seizure_detections']
            new_confidence = detection_result['seizure_confidence']
            
            if current_count == 0:
                self.stats['seizure_confidence_avg'] = new_confidence
            else:
                self.stats['seizure_confidence_avg'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
    
    def run_monitoring(self):
        """Main monitoring loop with dual detection"""
        if not self.initialize_components():
            return False
        
        self.running = True
        frame_count = 0
        
        print(f"\n🏥 ADVANCED HEALTHCARE MONITOR STARTED")
        print(f"📊 Keypoints Display: {'ON' if self.show_keypoints else 'OFF'}")
        print(f"📈 Statistics Display: {'ON' if self.show_statistics else 'OFF'}")
        print(f"�️  Dual Windows: {'ON' if self.show_dual_windows else 'OFF'}")
        print("\n🎮 Keyboard Controls:")
        print(f"   'k' = toggle keypoints | 's' = toggle statistics | 'd' = toggle dual windows")
        print(f"   'h' = show help | ' ' = screenshot | 'q' = quit")
        print("=" * 70)
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                self.stats['total_frames'] += 1
                frame_count += 1
                
                # Process frame
                processing_result = self.video_processor.process_frame(frame)
                
                if processing_result['processed']:
                    self.stats['keyframes_detected'] += 1
                    
                    # Dual detection processing
                    detection_result = self.process_dual_detection(
                        frame, processing_result['detections']
                    )
                    
                    # Update statistics
                    self.update_statistics(detection_result, len(processing_result['detections']))
                    
                    # Visualize results
                    frame_vis = self.visualize_dual_detection(
                        frame, detection_result, processing_result['detections']
                    )
                    
                    # Add statistics overlay
                    frame_vis = self.draw_statistics_overlay(frame_vis)
                    
                    # Create normal camera window
                    frame_normal = self.create_normal_camera_window(
                        frame, processing_result['detections']
                    )
                    
                    # Display both windows
                    cv2.imshow('Healthcare Monitor - Analysis View', frame_vis)
                    cv2.imshow('Healthcare Monitor - Normal View', frame_normal)
                    
                    # Log critical events
                    if detection_result['alert_level'] == 'critical':
                        alert_msg = f"🚨 CRITICAL ALERT: {detection_result['emergency_type']} detected!"
                        self.logger.critical(alert_msg)
                        print(f"[CRITICAL] {alert_msg}")
                
                else:
                    # Show both windows even when no processing
                    frame_with_stats = self.draw_statistics_overlay(frame)
                    frame_normal = self.create_normal_camera_window(frame, [])
                    
                    cv2.imshow('Healthcare Monitor - Analysis View', frame_with_stats)
                    cv2.imshow('Healthcare Monitor - Normal View', frame_normal)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('k'):
                    self.show_keypoints = not self.show_keypoints
                    print(f"🔧 Keypoints display: {'ON' if self.show_keypoints else 'OFF'}")
                elif key == ord('s'):
                    self.show_statistics = not self.show_statistics
                    print(f"🔧 Statistics display: {'ON' if self.show_statistics else 'OFF'}")
                elif key == ord('d'):
                    self.show_dual_windows = not self.show_dual_windows
                    print(f"🔧 Dual windows: {'ON' if self.show_dual_windows else 'OFF'}")
                    if not self.show_dual_windows:
                        cv2.destroyWindow('Healthcare Monitor - Normal View')
                elif key == ord('h'):
                    print("\n🎮 Keyboard Controls:")
                    print("   'q' = Quit")
                    print("   'k' = Toggle keypoints")
                    print("   's' = Toggle statistics")
                    print("   'd' = Toggle dual windows")
                    print("   'h' = Show this help")
                    print("   ' ' = Screenshot both windows")
                elif key == ord(' '):  # Spacebar for screenshot
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if 'frame_vis' in locals():
                            cv2.imwrite(f"screenshot_analysis_{timestamp}.jpg", frame_vis)
                        if 'frame_normal' in locals():
                            cv2.imwrite(f"screenshot_normal_{timestamp}.jpg", frame_normal)
                        print(f"📸 Screenshots saved: screenshot_*_{timestamp}.jpg")
                    except Exception as e:
                        print(f"❌ Screenshot failed: {e}")
                
                # Update motion stats
                if processing_result.get('motion_detected', False):
                    self.stats['motion_frames'] += 1
        
        except KeyboardInterrupt:
            print("\n[🏥] Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        # Print final statistics
        self.print_final_statistics()
        
        # Cleanup components
        if self.camera:
            self.camera.disconnect()
        
        cv2.destroyAllWindows()
        self.logger.info("Advanced Healthcare Monitor stopped")
    
    def print_final_statistics(self):
        """Print comprehensive final statistics"""
        runtime = time.time() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("🏥 ADVANCED HEALTHCARE MONITOR - FINAL STATISTICS")
        print("="*60)
        print(f"📊 Runtime: {runtime/60:.1f} minutes")
        print(f"📊 Total Frames: {self.stats['total_frames']}")
        print(f"📊 Frames Processed: {self.stats['frames_processed']}")
        print(f"📊 Processing Efficiency: {(1-self.stats['frames_processed']/max(self.stats['total_frames'],1))*100:.1f}% skipped")
        print(f"📊 Average FPS: {self.stats['total_frames']/runtime:.1f}")
        print()
        print("🩹 FALL DETECTION:")
        print(f"   Falls Detected: {self.stats['fall_detections']}")
        print(f"   Average Confidence: {self.stats['fall_confidence_avg']:.2f}")
        print(f"   Last Fall: {datetime.fromtimestamp(self.stats['last_fall_time']).strftime('%H:%M:%S') if self.stats['last_fall_time'] else 'None'}")
        print()
        print("🧠 SEIZURE DETECTION:")
        print(f"   Seizures Detected: {self.stats['seizure_detections']}")
        print(f"   Seizure Warnings: {self.stats['seizure_warnings']}")
        print(f"   Average Confidence: {self.stats['seizure_confidence_avg']:.2f}")
        print(f"   Pose Extraction Failures: {self.stats['pose_extraction_failures']}")
        print(f"   Last Seizure: {datetime.fromtimestamp(self.stats['last_seizure_time']).strftime('%H:%M:%S') if self.stats['last_seizure_time'] else 'None'}")
        print()
        print("🚨 EMERGENCY ALERTS:")
        print(f"   Critical Alerts: {self.stats['critical_alerts']}")
        print(f"   Total Alerts: {self.stats['total_alerts']}")
        print(f"   Current Status: {self.stats['alert_type']}")
        print("="*60)


def main():
    """Main function with configuration options"""
    print("🏥 Advanced Healthcare Monitor - Dual Detection System")
    print("=" * 60)
    
    # Configuration options
    print("🔧 Configuration Options:")
    print("1. Show keypoints on video? (y/n, default: y)")
    print("2. Show real-time statistics? (y/n, default: y)")
    
    try:
        show_keypoints_input = input("Show keypoints [y/n]: ").lower().strip()
        show_keypoints = show_keypoints_input != 'n'
        
        show_stats_input = input("Show statistics [y/n]: ").lower().strip()
        show_statistics = show_stats_input != 'n'
        
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        return
    
    print(f"\n✅ Configuration:")
    print(f"   Keypoints Display: {'ON' if show_keypoints else 'OFF'}")
    print(f"   Statistics Display: {'ON' if show_statistics else 'OFF'}")
    print(f"   Dual Windows: ON (Normal View + Analysis View)")
    print(f"   📺 Two windows will open:")
    print(f"      - Normal View: Clean camera feed with basic person detection")
    print(f"      - Analysis View: Full analysis with keypoints, statistics & alerts")
    print(f"   Controls: 'k'=keypoints, 's'=stats, 'd'=dual windows, 'h'=help, 'q'=quit")
    
    # Initialize and run monitor
    monitor = AdvancedHealthcareMonitor(
        show_keypoints=show_keypoints,
        show_statistics=show_statistics
    )
    
    success = monitor.run_monitoring()
    
    if success:
        print("\n✅ Healthcare monitoring completed successfully!")
    else:
        print("\n❌ Healthcare monitoring failed!")


if __name__ == "__main__":
    main()
