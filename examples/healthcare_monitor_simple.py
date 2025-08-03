#!/usr/bin/env python3
"""
Simple Healthcare Monitor - Tích hợp Camera IMOU + YOLO Detection
Mục đích: Monitor healthcare sử dụng camera IMOU để quan sát hành vi người dùng
Tính năng: Detect người, detect té ngã, detect co giật, optimized performance
"""

import sys
import os
import time
import cv2
import threading
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from camera.simple_camera import SimpleIMOUCamera
from video_processing.simple_processing import IntegratedVideoProcessor


class SimpleHealthcareMonitor:
    """Simplified Healthcare Monitor - Dual display với performance optimization"""
    
    def __init__(self):
        """Initialize healthcare monitoring system"""
        # Setup logging
        self.setup_logging()
        
        self.logger.info("Initializing Healthcare Monitor with Fall Detection...")
        print("[🏥] Initializing Healthcare Monitor with Fall Detection...")
        
        # Camera configuration cho IMOU
        self.camera_config = {
            'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',  # IMOU RTSP URL đúng
            'buffer_size': 1,           # Giảm buffer delay
            'fps': 15,                  # Optimized FPS cho healthcare
            'resolution': (640, 480),   # Standard resolution
            'auto_reconnect': True      # Auto reconnect when lost
        }
        
        # Initialize components
        self.camera = None
        self.video_processor = None
        self.healthcare_analyzer = None
        self.running = False
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'alerts_triggered': 0,
            'fall_detections': 0,       # New fall detection counter
            'last_detection_time': None,
            'last_fall_time': None,     # New fall detection time
            'total_frames': 0,
            'keyframes_detected': 0,
            'motion_frames': 0,
            'start_time': time.time()
        }
    
    def setup_logging(self):
        """Setup logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('HealthcareMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with UTF-8 encoding
        log_file = log_dir / f"healthcare_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Formatter without emojis for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging system initialized")
        print(f"📝 Logging to: {log_file}")
        
        # Store log file path for reference
        self.log_file = log_file
        
    def setup_camera(self) -> bool:
        """Setup IMOU camera connection"""
        try:
            self.logger.info("Setting up IMOU camera...")
            print("[📹] Setting up IMOU camera...")
            
            self.camera = SimpleIMOUCamera(self.camera_config)
            
            if self.camera.connect():
                self.logger.info("Camera connected successfully!")
                print("✅ Camera connected successfully!")
                return True
            else:
                self.logger.error("Camera connection failed!")
                print("❌ Camera connection failed!")
                return False
                
        except Exception as e:
            self.logger.error(f"Camera setup error: {e}")
            print(f"❌ Camera setup error: {e}")
            return False
    
    def setup_video_processing(self) -> bool:
        """Setup video processing với Integrated Keyframe Detection và Fall Detection"""
        try:
            self.logger.info("Setting up AI processing with Keyframe Detection và Fall Detection...")
            print("[🤖] Setting up AI processing with Keyframe Detection và Fall Detection...")
            
            # Initialize Integrated Video Processor với keyframe detection và fall detection
            self.video_processor = IntegratedVideoProcessor(
                motion_threshold=120,       # Motion detection threshold (optimized)
                keyframe_threshold=0.25,    # Keyframe detection threshold (lower for more detection)
                yolo_confidence=0.4,        # YOLO confidence (optimized for person detection)
                save_frames=True,           # Enable frame saving
                base_save_path="data/saved_frames"  # Save path
            )
            
            # Check if fall detection is available
            fall_available = hasattr(self.video_processor, 'fall_detector') and self.video_processor.fall_detector is not None
            
            self.logger.info(f"AI processing setup complete with Keyframe Detection! Fall Detection: {'✅' if fall_available else '❌'}")
            print(f"✅ AI processing setup complete with Keyframe Detection! Fall Detection: {'✅' if fall_available else '❌'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Video processing setup error: {e}")
            print(f"❌ Video processing setup error: {e}")
            return False
    
    def display_dual_windows(self):
        """Display dual camera windows với real-time processing"""
        try:
            print("[🖥️] Starting dual display...")
            
            # Create windows
            cv2.namedWindow('IMOU Camera - Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Healthcare Monitor - AI Processing', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('IMOU Camera - Original', 640, 480)
            cv2.resizeWindow('Healthcare Monitor - AI Processing', 640, 480)
            
            frame_count = 0
            last_stats_time = time.time()
            
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    print("⚠️ No frame from camera")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Original camera display
                cv2.imshow('IMOU Camera - Original', frame)
                
                # AI Processing (every 3rd frame for performance)
                if frame_count % 3 == 0:
                    try:
                        # Process với IntegratedVideoProcessor - keyframe detection
                        results = self.video_processor.process_frame(frame)
                        
                        # Check if frame was processed
                        if results.get('processed', False):
                            # Update stats
                            self.stats['frames_processed'] += 1
                            
                            # Log keyframe detection
                            if results.get('is_keyframe', False):
                                kf_conf = results.get('keyframe_confidence', 0)
                                self.logger.info(f"Keyframe detected: confidence={kf_conf:.3f}")
                            
                            if results.get('person_count', 0) > 0:
                                person_count = results['person_count']
                                self.stats['persons_detected'] += person_count
                                self.stats['last_detection_time'] = time.time()
                                self.logger.info(f"Persons detected: {person_count}")
                            
                            # Check for fall detection
                            if results.get('fall_detected', False):
                                fall_confidence = results.get('fall_confidence', 0)
                                self.stats['fall_detections'] += 1
                                self.stats['last_fall_time'] = time.time()
                                self.logger.critical(f"🚨 FALL DETECTED! Confidence: {fall_confidence:.1%}")
                                print(f"🚨 FALL ALERT! Confidence: {fall_confidence:.1%}")
                            
                            if results.get('alerts', []):
                                alert_count = len(results['alerts'])
                                self.stats['alerts_triggered'] += alert_count
                                self.logger.warning(f"Healthcare alerts triggered: {alert_count}")
                                
                                # Log each alert
                                for alert in results['alerts']:
                                    alert_type = alert.get('type', 'unknown')
                                    confidence = alert.get('confidence', 0)
                                    severity = alert.get('severity', 'unknown')
                                    message = alert.get('message', '')
                                    
                                    if alert_type == 'fall_detected':
                                        self.logger.critical(f"🚨 FALL ALERT: {message} (confidence: {confidence:.1%})")
                                        print(f"🚨 FALL ALERT: {message}")
                                    else:
                                        self.logger.warning(f"HEALTHCARE ALERT: {alert_type} (confidence: {confidence:.2f})")
                                
                            # Display processed frame (from YOLO if available)
                            health_analysis = results.get('health_analysis', {})
                            detections = results.get('detections', [])
                            fall_detected = results.get('fall_detected', False)
                            fall_confidence = results.get('fall_confidence', 0)
                            
                            # Draw detections on frame
                            processed_frame = frame.copy()
                            for detection in detections:
                                bbox = detection.get('bbox', [])
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    
                                    # Color based on fall detection
                                    color = (0, 0, 255) if fall_detected else (0, 255, 0)  # Red if fall, green otherwise
                                    
                                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Add label
                                    label = f"{detection.get('class_name', 'unknown')}: {detection.get('confidence', 0):.2f}"
                                    if fall_detected:
                                        label += f" [FALL: {fall_confidence:.1%}]"
                                    
                                    cv2.putText(processed_frame, label, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add keyframe indicator at bottom left
                            if results.get('is_keyframe', False):
                                h, w = processed_frame.shape[:2]
                                kf_text = f"KEYFRAME ({results.get('keyframe_confidence', 0):.3f})"
                                cv2.putText(processed_frame, kf_text, 
                                          (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Add statistics overlay (top right)
                            self.add_stats_overlay(processed_frame, results)
                            
                            cv2.imshow('Healthcare Monitor - AI Processing', processed_frame)
                        else:
                            # Frame not processed (no motion or not keyframe)
                            simple_frame = frame.copy()
                            reason = results.get('reason', 'No processing')
                            cv2.putText(simple_frame, f"Skipped: {reason}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                            
                            # Still show stats overlay
                            self.add_stats_overlay(simple_frame, results)
                            cv2.imshow('Healthcare Monitor - AI Processing', simple_frame)
                            
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                        cv2.imshow('Healthcare Monitor - AI Processing', frame)
                
                # Print stats every 10 seconds
                if time.time() - last_stats_time > 10:
                    self.print_statistics()
                    last_stats_time = time.time()
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("[👋] Exit requested...")
                    break
                    
        except KeyboardInterrupt:
            print("\n[👋] Keyboard interrupt...")
        except Exception as e:
            print(f"❌ Display error: {e}")
        finally:
            self.cleanup()
    
    def add_stats_overlay(self, frame, results=None):
        """Add statistics overlay to frame - Hiển thị ở góc phải"""
        try:
            h, w = frame.shape[:2]
            
            # Calculate current stats
            current_time = time.time()
            uptime = current_time - self.stats['start_time']
            
            # Update total frames counter
            self.stats['total_frames'] += 1
            
            # Get processing stats if available
            proc_stats = {}
            if results and 'processing_stats' in results:
                proc_stats = results['processing_stats']
                self.stats['keyframes_detected'] = proc_stats.get('keyframes', 0)
                self.stats['motion_frames'] = proc_stats.get('motion_frames', 0)
            
            # Stats text - Fixed position at top right
            stats_lines = [
                "HEALTHCARE MONITOR",
                f"Uptime: {uptime/60:.1f}m",
                f"Total: {self.stats['total_frames']}",
                f"Processed: {self.stats['frames_processed']}",
                f"Motion: {self.stats['motion_frames']}",
                f"Keyframes: {self.stats['keyframes_detected']}",
                f"Persons: {self.stats['persons_detected']}",
                f"Alerts: {self.stats['alerts_triggered']}",
                f"Falls: {self.stats['fall_detections']}",  # New fall counter
                ""
            ]
            
            # Add fall detection status
            if self.stats['last_fall_time']:
                time_since_fall = current_time - self.stats['last_fall_time']
                if time_since_fall < 10:  # Show for 10 seconds
                    stats_lines.append("🚨 FALL DETECTED!")
                else:
                    time_str = f"{time_since_fall/60:.1f}m ago" if time_since_fall > 60 else f"{time_since_fall:.0f}s ago"
                    stats_lines.append(f"Last fall: {time_str}")
            
            # Add processing efficiency if available
            if proc_stats:
                efficiency = proc_stats.get('processing_efficiency', 'N/A')
                keyframe_rate = proc_stats.get('keyframe_rate', 0) * 100
                fall_rate = proc_stats.get('fall_detection_rate', 0) * 100
                stats_lines.extend([
                    f"Efficiency: {efficiency}",
                    f"Keyframe Rate: {keyframe_rate:.1f}%",
                    f"Fall Rate: {fall_rate:.1f}%",
                    ""
                ])
            
            # Add current frame info if available
            if results:
                if results.get('fall_detected'):
                    fall_conf = results.get('fall_confidence', 0)
                    stats_lines.append(f"🚨 FALL: {fall_conf:.1%}")
                elif results.get('is_keyframe'):
                    kf_conf = results.get('keyframe_confidence', 0)
                    stats_lines.append(f"KEYFRAME: {kf_conf:.3f}")
                else:
                    reason = results.get('reason', 'Skipped')
                    stats_lines.append(f"Status: {reason}")
                
                if results.get('person_count', 0) > 0:
                    stats_lines.append(f"Current Persons: {results['person_count']}")
            
            # Calculate text size and position (top right)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            line_height = 16
            
            # Find max text width
            max_width = 0
            for line in stats_lines:
                if line:  # Skip empty lines
                    (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                    max_width = max(max_width, text_width)
            
            # Panel dimensions
            panel_width = max_width + 20
            panel_height = len([l for l in stats_lines if l]) * line_height + 20
            
            # Position at top right
            panel_x = w - panel_width - 10
            panel_y = 10
            
            # Draw background panel
            cv2.rectangle(frame, 
                         (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)  # Black background
            cv2.rectangle(frame, 
                         (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 255, 0), 2)  # Green border
            
            # Draw text
            y_offset = panel_y + 20
            for line in stats_lines:
                if line:  # Skip empty lines
                    cv2.putText(frame, line, 
                              (panel_x + 10, y_offset), 
                              font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    y_offset += line_height
                           
        except Exception as e:
            self.logger.error(f"Overlay error: {e}")
            print(f"⚠️ Overlay error: {e}")
    
    def print_statistics(self):
        """Print monitoring statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        stats_summary = [
            "="*50,
            "📊 HEALTHCARE MONITOR STATISTICS",
            "="*50,
            f"Uptime: {uptime/60:.1f} minutes",
            f"Total Frames: {self.stats['total_frames']}",
            f"Frames Processed: {self.stats['frames_processed']}",
            f"Motion Frames: {self.stats['motion_frames']}",
            f"Keyframes Detected: {self.stats['keyframes_detected']}",
            f"Persons Detected: {self.stats['persons_detected']}",
            f"Alerts Triggered: {self.stats['alerts_triggered']}",
            f"Fall Detections: {self.stats['fall_detections']}"  # New fall counter
        ]
        
        if self.stats['last_detection_time']:
            time_since = current_time - self.stats['last_detection_time']
            stats_summary.append(f"Last Detection: {time_since:.1f}s ago")
        else:
            stats_summary.append("Last Detection: Never")
        
        if self.stats['last_fall_time']:
            time_since_fall = current_time - self.stats['last_fall_time']
            stats_summary.append(f"Last Fall: {time_since_fall:.1f}s ago")
        else:
            stats_summary.append("Last Fall: Never")
        
        # Add processing efficiency stats if available
        if self.video_processor:
            try:
                proc_stats = self.video_processor.get_processing_stats()
                stats_summary.extend([
                    f"Processing Efficiency: {proc_stats.get('processing_efficiency', 'N/A')}",
                    f"YOLO Processes: {proc_stats.get('yolo_processed', 0)}",
                    f"Keyframe Rate: {proc_stats.get('keyframe_rate', 0)*100:.1f}%",
                    f"Motion Rate: {proc_stats.get('motion_rate', 0)*100:.1f}%",
                    f"Fall Detection Rate: {proc_stats.get('fall_detection_rate', 0)*100:.1f}%"
                ])
            except Exception as e:
                self.logger.warning(f"Error getting processor stats: {e}")
        
        stats_summary.append("="*50)
        
        # Print to console
        for line in stats_summary:
            print(line)
        
        # Log summary
        self.logger.info("=== STATISTICS SUMMARY ===")
        for line in stats_summary[2:-1]:  # Skip decorative lines
            if not line.startswith("="):  # Skip separator lines
                self.logger.info(line)
    
    def run(self):
        """Main run method"""
        try:
            self.logger.info("Healthcare Monitor Starting...")
            print("\n🏥 HEALTHCARE MONITOR STARTING...")
            print("Mục đích: Sử dụng camera IMOU để quan sát và đánh giá hành vi người dùng")
            print("Tính năng: Detect người, detect té ngã, detect co giật, optimized performance")
            print("\n")
            
            # Setup camera
            if not self.setup_camera():
                self.logger.error("Camera setup failed!")
                print("❌ Camera setup failed!")
                return False
            
            # Setup video processing  
            if not self.setup_video_processing():
                self.logger.error("Video processing setup failed!")
                print("❌ Video processing setup failed!")
                return False
            
            # Start monitoring
            self.running = True
            self.logger.info("All systems ready!")
            print("✅ All systems ready!")
            print("\n[📹] Starting healthcare monitoring...")
            print("Press 'q' or ESC to exit")
            print("-" * 50)
            
            # Start dual display
            self.display_dual_windows()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            print(f"❌ Runtime error: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        print("\n[🧹] Cleaning up...")
        
        self.running = False
        
        if self.camera:
            self.camera.disconnect()
            self.logger.info("Camera disconnected")
            
        cv2.destroyAllWindows()
        
        # Final stats
        self.print_statistics()
        
        self.logger.info("Healthcare Monitor stopped")
        print("🏥 Healthcare Monitor stopped.")


def main():
    """Main function"""
    print("="*60)
    print("🏥 HEALTHCARE MONITORING SYSTEM")
    print("IMOU Camera + YOLO AI Detection + Healthcare Analytics")
    print("="*60)
    
    # Create and run monitor
    monitor = SimpleHealthcareMonitor()
    
    try:
        success = monitor.run()
        if success:
            print("\n✅ Healthcare monitoring completed successfully!")
        else:
            print("\n❌ Healthcare monitoring failed!")
            
    except KeyboardInterrupt:
        print("\n[👋] Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n🏥 Healthcare Monitor terminated.")


if __name__ == "__main__":
    main()
