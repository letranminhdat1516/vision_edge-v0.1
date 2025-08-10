#!/usr/bin/env python3
"""
Healthcare Monitor Main Application
Function-based healthcare monitoring system entry point
"""

import sys
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import function-based modules
from core.config import load_config
from core.logger import setup_logging
from core.utils import calculate_runtime_stats
from camera.controls import initialize_camera, FrameBuffer, start_camera_thread, release_camera
from processing.video import detect_objects_yolo, find_largest_person, calculate_frame_quality
from detection.motion import calculate_motion_level, analyze_motion_patterns
from detection.fall import process_fall_detection, initialize_fall_detector
from detection.seizure import process_seizure_detection, initialize_seizure_detector
from visualization.display import (draw_bounding_box, draw_motion_indicators, 
                                  draw_detection_alerts, draw_statistics_overlay)
from alerts.management import AlertManager, format_alert_message


class HealthcareMonitorApp:
    """Main healthcare monitoring application"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        
        # Initialize components
        self.camera = None
        self.frame_buffer = FrameBuffer(max_size=5)
        self.running = threading.Event()
        self.camera_thread = None
        
        # Initialize detectors
        self.yolo_model = None
        self.fall_detector = None
        self.fall_predictor = None
        self.seizure_detector = None
        self.seizure_predictor = None
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        
        # State variables
        self.frame_count = 0
        self.start_time = time.time()
        self.previous_frame = None
        self.motion_history = []
        self.fall_confidence_history = []
        self.seizure_confidence_history = []
        self.fall_confirmation_frames = 0
        self.seizure_confirmation_frames = 0
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'fall_alerts': 0,
            'seizure_alerts': 0,
            'motion_alerts': 0,
            'avg_fps': 0.0,
            'avg_processing_time': 0.0
        }
        
        logging.info("Healthcare Monitor initialized")
    
    def initialize_models(self) -> bool:
        """Initialize detection models"""
        try:
            # Initialize YOLO model
            try:
                from ultralytics import YOLO
                model_path = self.config.get('detection', {}).get('yolo_model', 'yolov8s.pt')
                self.yolo_model = YOLO(model_path)
                logging.info(f"YOLO model loaded: {model_path}")
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {str(e)}")
                return False
            
            # Initialize fall detection
            self.fall_detector, self.fall_predictor = initialize_fall_detector(
                self.config.get('detection', {}).get('fall_threshold', 0.7)
            )
            logging.info("Fall detection initialized")
            
            # Initialize seizure detection
            self.seizure_detector, self.seizure_predictor = initialize_seizure_detector(
                self.config.get('detection', {}).get('seizure_threshold', 0.7)
            )
            logging.info("Seizure detection initialized")
            
            return True
            
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            return False
    
    def initialize_camera_system(self) -> bool:
        """Initialize camera system"""
        try:
            camera_config = self.config.get('camera', {})
            
            # Initialize camera
            self.camera, status_info = initialize_camera(camera_config)
            
            if not status_info['connected']:
                logging.error(f"Camera initialization failed: {status_info['error']}")
                return False
            
            # Start camera thread
            self.running.set()
            fps_limit = camera_config.get('fps', 15)
            self.camera_thread = start_camera_thread(
                self.camera, self.frame_buffer, self.running, fps_limit
            )
            
            logging.info(f"Camera system initialized: {status_info['resolution']} @ {status_info['fps']} FPS")
            return True
            
        except Exception as e:
            logging.error(f"Camera system initialization failed: {str(e)}")
            return False
    
    def process_frame(self, frame) -> Dict[str, Any]:
        """Process a single frame through the detection pipeline"""
        processing_start = time.time()
        
        result = {
            'detections': [],
            'motion_level': 0.0,
            'fall_detected': False,
            'seizure_detected': False,
            'alerts': [],
            'processing_time': 0.0
        }
        
        try:
            # Object detection
            detections = detect_objects_yolo(
                frame, self.yolo_model, 
                confidence_threshold=self.config.get('detection', {}).get('confidence_threshold', 0.5)
            )
            result['detections'] = detections
            
            # Find largest person
            person_detection = find_largest_person(detections, person_class_id=0)
            
            if person_detection:
                person_bbox = person_detection['bbox']
                
                # Calculate motion level
                motion_level = 0.0
                if self.previous_frame is not None:
                    motion_level = calculate_motion_level(self.previous_frame, frame, person_bbox)
                    result['motion_level'] = motion_level
                    
                    # Update motion history
                    self.motion_history.append(motion_level)
                    if len(self.motion_history) > 30:  # Keep last 30 frames
                        self.motion_history = self.motion_history[-30:]
                
                # Fall detection
                if self.fall_detector is not None:
                    fall_result = process_fall_detection(
                        self.fall_detector, self.fall_predictor, frame, person_bbox,
                        self.fall_confidence_history, self.fall_confirmation_frames, 
                        motion_level, self.config.get('detection', {})
                    )
                    
                    result['fall_detected'] = fall_result['fall_detected']
                    self.fall_confidence_history = fall_result['confidence_history']
                    self.fall_confirmation_frames = fall_result['confirmation_frames']
                    
                    # Create fall alert if detected
                    if fall_result['fall_detected']:
                        alert_message = format_alert_message('fall', fall_result['smoothed_confidence'])
                        alert = self.alert_manager.add_alert(
                            'fall', alert_message, fall_result['smoothed_confidence'],
                            {'bbox': person_bbox, 'motion_level': motion_level}
                        )
                        if alert:
                            result['alerts'].append(alert)
                            self.stats['fall_alerts'] += 1
                
                # Seizure detection
                if self.seizure_detector is not None:
                    seizure_result = process_seizure_detection(
                        self.seizure_detector, self.seizure_predictor, frame, person_bbox,
                        self.seizure_confidence_history, self.seizure_confirmation_frames,
                        motion_level, self.config.get('detection', {})
                    )
                    
                    result['seizure_detected'] = seizure_result['seizure_detected']
                    self.seizure_confidence_history = seizure_result['confidence_history']
                    self.seizure_confirmation_frames = seizure_result['confirmation_frames']
                    
                    # Create seizure alert if detected
                    if seizure_result['seizure_detected']:
                        alert_message = format_alert_message('seizure', seizure_result['smoothed_confidence'])
                        alert = self.alert_manager.add_alert(
                            'seizure', alert_message, seizure_result['smoothed_confidence'],
                            {'bbox': person_bbox, 'motion_level': motion_level}
                        )
                        if alert:
                            result['alerts'].append(alert)
                            self.stats['seizure_alerts'] += 1
                
                # Motion alert for very high motion
                if motion_level > 0.9:
                    alert_message = format_alert_message('motion', motion_level, {'motion_level': motion_level})
                    alert = self.alert_manager.add_alert(
                        'motion', alert_message, motion_level,
                        {'bbox': person_bbox, 'motion_level': motion_level}
                    )
                    if alert:
                        result['alerts'].append(alert)
                        self.stats['motion_alerts'] += 1
            
            # Store previous frame
            self.previous_frame = frame.copy()
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - processing_start
        return result
    
    def create_display_frame(self, frame, processing_result: Dict[str, Any]):
        """Create display frame with visualizations"""
        display_frame = frame.copy()
        
        try:
            # Draw person detection
            for detection in processing_result.get('detections', []):
                if detection['class_id'] == 0:  # Person class
                    display_frame = draw_bounding_box(
                        display_frame, detection['bbox'],
                        color=(0, 255, 0), thickness=2,
                        label="Person", confidence=detection['confidence']
                    )
            
            # Draw motion indicators
            motion_level = processing_result.get('motion_level', 0.0)
            display_frame = draw_motion_indicators(display_frame, motion_level, "top-right")
            
            # Draw alerts
            alerts = processing_result.get('alerts', [])
            if alerts:
                display_frame = draw_detection_alerts(display_frame, alerts)
            
            # Draw statistics
            current_stats = self.get_current_statistics()
            display_frame = draw_statistics_overlay(display_frame, current_stats)
            
        except Exception as e:
            logging.error(f"Display creation error: {str(e)}")
        
        return display_frame
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        runtime_stats = calculate_runtime_stats(self.start_time, self.frame_count)
        
        current_stats = {
            'Runtime': f"{runtime_stats['runtime_minutes']:.1f}m",
            'FPS': f"{runtime_stats['avg_fps']:.1f}",
            'Frames': self.frame_count,
            'Fall Alerts': self.stats['fall_alerts'],
            'Seizure Alerts': self.stats['seizure_alerts'],
            'Motion Alerts': self.stats['motion_alerts']
        }
        
        # Add active alerts count
        active_alerts = self.alert_manager.get_active_alerts()
        current_stats['Active Alerts'] = len(active_alerts)
        
        return current_stats
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        logging.info("Starting healthcare monitoring...")
        
        try:
            while self.running.is_set():
                # Get frame from buffer
                frame = self.frame_buffer.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                processing_result = self.process_frame(frame)
                
                # Update statistics
                self.frame_count += 1
                self.stats['frames_processed'] = self.frame_count
                
                # Create display frame
                display_frame = self.create_display_frame(frame, processing_result)
                
                # Show frame (in a real application, this might send to a display or stream)
                # For now, just log processing results
                if processing_result.get('alerts'):
                    logging.info(f"Frame {self.frame_count}: {len(processing_result['alerts'])} alerts generated")
                
                # Clean up old alerts periodically
                if self.frame_count % 300 == 0:  # Every 300 frames (~20 seconds at 15 FPS)
                    self.alert_manager.clear_old_alerts()
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Monitoring loop error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up resources...")
        
        # Stop camera
        if self.camera is not None:
            release_camera(self.camera, self.camera_thread, self.running)
        
        # Log final statistics
        final_stats = self.get_current_statistics()
        logging.info(f"Final statistics: {final_stats}")
        
        logging.info("Healthcare monitor shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Healthcare Monitor - Function-based Architecture')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create and run application
    app = HealthcareMonitorApp(config_path=args.config)
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled")
    
    # Initialize models
    if not app.initialize_models():
        logging.error("Model initialization failed, exiting")
        return 1
    
    # Initialize camera
    if not app.initialize_camera_system():
        logging.error("Camera initialization failed, exiting")
        return 1
    
    # Run monitoring loop
    try:
        app.run_monitoring_loop()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
