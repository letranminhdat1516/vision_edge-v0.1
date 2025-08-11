#!/usr/bin/env python3
"""
Advanced Healthcare Monitor - Dual Detection System với WebSocket
Tích hợp: Fall Detection + Seizure Detection + Real-time Statistics + WebSocket Streaming
Features: IMOU Camera + YOLO + Fall Detection + VSViG Seizure Detection + Server Integration
"""

import sys
import os
import time
import cv2
import threading
import logging
import numpy as np
import asyncio
import websockets
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from camera.simple_camera import SimpleIMOUCamera
try:
    from video_processing.simple_processing import IntegratedVideoProcessor
except ImportError:
    print("⚠️ Warning: video_processing module not found, using simple fallback")
    class IntegratedVideoProcessor:
        def __init__(self, config):
            self.config = config
        def process_frame(self, frame):
            return {
                'processed': True,
                'person_detections': [],
                'detections': [],
                'processing_time': 0.01
            }

# Import fall detection - SIMPLIFIED
from fall_detection.simple_fall_detector import SimpleFallDetector
print("✅ SimpleFallDetector (simplified) loaded")

try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor
except ImportError:
    print("⚠️ Warning: seizure_detection modules not found, using simple fallback")
    class VSViGSeizureDetector:
        def __init__(self, confidence_threshold=0.65):
            self.confidence_threshold = confidence_threshold
        def detect_seizure(self, frame, bbox):
            return {
                'temporal_ready': False,
                'keypoints': None,
                'confidence': 0.0
            }
    
    class SeizurePredictor:
        def __init__(self, temporal_window=30, alert_threshold=0.65, warning_threshold=0.45):
            pass
        def update_prediction(self, confidence):
            return {
                'smoothed_confidence': 0.0,
                'seizure_detected': False,
                'alert_level': 'normal',
                'ready': False
            }


class AdvancedHealthcareMonitor:
    """
    Advanced Healthcare Monitor với Dual Detection System
    Features: Fall Detection + Seizure Detection + Keypoint Visualization
    """
    
    def __init__(self, show_keypoints: bool = True, show_statistics: bool = True, 
                 websocket_url: str = "ws://localhost:8086", enable_streaming: bool = True,
                 enable_api_integration: bool = True):
        """
        Initialize advanced healthcare monitoring system
        
        Args:
            show_keypoints: Whether to display pose keypoints on video
            show_statistics: Whether to display real-time statistics
            websocket_url: WebSocket server URL for real-time streaming
            enable_streaming: Whether to enable WebSocket streaming
            enable_api_integration: Whether to send data to API servers
        """
        # Setup logging
        self.setup_logging()
        
        self.logger.info("🏥 Initializing Advanced Healthcare Monitor - Dual Detection System")
        print("[🏥] Initializing Advanced Healthcare Monitor - Dual Detection System")
        
        # Display settings
        self.show_keypoints = show_keypoints
        self.show_statistics = show_statistics
        self.show_dual_windows = True  # Always show both windows
        
        # WebSocket configuration for mobile alerts
        self.websocket_url = websocket_url
        self.enable_streaming = enable_streaming
        self.websocket = None
        
        # Initialize WebSocket server for mobile alerts
        if enable_streaming:
            try:
                from healthcare_websocket_production import HealthcareWebSocketServer
                self.mobile_server = HealthcareWebSocketServer(host="0.0.0.0", port=9999)
                self.mobile_server_thread = self.mobile_server.run_server_in_thread()
                print("📱 Mobile WebSocket server started on ws://0.0.0.0:9999")
            except Exception as e:
                print(f"⚠️ Mobile WebSocket server failed to start: {e}")
                self.mobile_server = None
        
        # API Integration configuration - DISABLED
        self.enable_api_integration = False  # Tắt API integration
        self.demo_api_url = None
        self.mobile_api_url = None
        
        # Alerts folder configuration
        self.alerts_folder = Path("examples/data/saved_frames/alerts")
        self.alerts_folder.mkdir(parents=True, exist_ok=True)
        
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
        
        # Detection smoothing and temporal filtering
        self.detection_history = {
            'fall_confidences': [],
            'seizure_confidences': [],
            'person_positions': [],
            'motion_levels': [],
            'max_history': 10,  # Keep last 10 frames for smoothing
            'fall_confirmation_frames': 0,
            'seizure_confirmation_frames': 0,
            'last_significant_motion': time.time()
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
    
    def save_alert_image(self, frame: np.ndarray, alert_type: str, confidence: float) -> bool:
        """
        Save alert image with metadata to alerts folder
        
        Args:
            frame: Frame to save
            alert_type: Type of alert (fall_detected, seizure_detected)
            confidence: Detection confidence
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Generate timestamp filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # milliseconds
            filename = f"alert_{timestamp}_{alert_type}_conf_{confidence:.3f}.jpg"
            filepath = self.alerts_folder / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), frame)
            
            if success:
                # Store last saved image filename for API
                self._last_saved_image = filename
                
                # Save metadata
                metadata = {
                    "alert_type": alert_type,
                    "confidence": str(confidence),
                    "keyframe_confidence": 0.0,  # placeholder
                    "timestamp": datetime.now().isoformat()
                }
                
                metadata_file = filepath.with_suffix('.jpg_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"🚨 Alert image saved: {filename}")
                print(f"🚨 Alert image saved: {filename}")
                
                # Send alert to mobile clients
                self.send_mobile_alert(alert_type, confidence, filename)
                
                return True
            else:
                self.logger.error(f"Failed to save alert image: {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving alert image: {e}")
            print(f"❌ Error saving alert image: {e}")
            return False
    
    def send_mobile_alert(self, alert_type: str, confidence: float, image_filename: str):
        """
        Send alert to mobile clients via WebSocket
        
        Args:
            alert_type: Type of alert (fall_detected, seizure_detected, fall_warning, seizure_warning, etc.)
            confidence: Detection confidence
            image_filename: Saved alert image filename
        """
        if not hasattr(self, 'mobile_server') or self.mobile_server is None:
            return
        
        try:
            # Construct image URL (assumes web server serving alerts folder)
            base_url = "http://localhost:9999/alerts"
            image_url = f"{base_url}/{image_filename}"
            
            # Determine action and status based on alert_type and confidence
            if "fall" in alert_type:
                detection_type = "fall"
                if confidence >= 0.2:  # Lowered from 0.8 - Fall detection confidence is typically lower
                    status = "danger"
                    action = "fall_detected"
                    log_level = "CRITICAL"
                    log_message = f"🚨 CRITICAL ALERT: fall detected!"
                elif confidence >= 0.15:  # Medium confidence  
                    status = "warning"
                    action = "fall_warning"
                    log_level = "WARNING"
                    log_message = f"⚠️ WARNING ALERT: possible fall detected (medium confidence)"
                else:  # Low confidence
                    status = "normal"
                    action = "normal_movement"
                    log_level = "INFO"
                    log_message = f"👤 Normal movement detected - monitoring for fall patterns"
                    
            elif "seizure" in alert_type:
                detection_type = "seizure"
                if confidence >= 0.7:  # High confidence for seizures
                    status = "danger"
                    action = "seizure_detected"
                    log_level = "CRITICAL"
                    log_message = f"🚨 CRITICAL ALERT: seizure detected!"
                elif confidence >= 0.4:  # Medium confidence  
                    status = "warning"
                    action = "seizure_warning"
                    log_level = "WARNING"
                    log_message = f"⚠️ WARNING ALERT: possible seizure detected (medium confidence)"
                else:  # Low confidence
                    status = "normal"
                    action = "normal_brain_activity"
                    log_level = "INFO"
                    log_message = f"🧠 Normal brain activity detected - monitoring for seizure patterns"
            else:
                # Default fallback
                status = "normal"
                action = "normal_activity"
                log_level = "INFO"
                log_message = f"✅ Normal status - {alert_type.replace('_', ' ')} monitoring active"
            
            # Log the appropriate message
            if log_level == "CRITICAL":
                self.logger.critical(log_message)
            elif log_level == "WARNING":
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            # Create alert data matching exact format requested: imageUrl, status, action, time
            alert_data = {
                "imageUrl": image_url,
                "status": status,
                "action": action,
                "time": int(time.time())  # timestamp format
            }
            
            # Send to mobile clients
            self.mobile_server.send_alert_sync(alert_data)
            
            self.logger.info(f"📱 Mobile alert sent: {action} (confidence: {confidence:.2f}, status: {status})")
            
        except Exception as e:
            self.logger.error(f"Failed to send mobile alert: {e}")
            print(f"❌ Mobile alert failed: {e}")
    
    def calculate_motion_level(self, person_detections: list) -> float:
        """
        Calculate motion level based on person position changes
        
        Args:
            person_detections: Current person detections
            
        Returns:
            float: Motion level (0.0 to 1.0)
        """
        if not person_detections:
            return 0.0
            
        # Get primary person center
        primary_person = max(person_detections, key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
        bbox = primary_person['bbox']
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        current_position = (center_x, center_y)
        
        # Add to position history
        self.detection_history['person_positions'].append(current_position)
        if len(self.detection_history['person_positions']) > self.detection_history['max_history']:
            self.detection_history['person_positions'].pop(0)
        
        # Calculate motion if we have enough history
        if len(self.detection_history['person_positions']) < 3:
            return 0.5  # Default medium motion
            
        # Calculate average displacement
        positions = self.detection_history['person_positions']
        total_displacement = 0.0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            displacement = np.sqrt(dx*dx + dy*dy)
            total_displacement += displacement
        
        # Normalize motion level (adjust these values based on your camera setup)
        avg_displacement = total_displacement / (len(positions) - 1)
        motion_level = min(avg_displacement / 50.0, 1.0)  # 50 pixels = max motion
        
        return motion_level
    
    def smooth_detection_confidence(self, current_confidence: float, detection_type: str) -> float:
        """
        Apply temporal smoothing to detection confidence
        
        Args:
            current_confidence: Current frame confidence
            detection_type: 'fall' or 'seizure'
            
        Returns:
            float: Smoothed confidence
        """
        if detection_type == 'fall':
            history_key = 'fall_confidences'
        else:
            history_key = 'seizure_confidences'
        
        # Add current confidence to history
        self.detection_history[history_key].append(current_confidence)
        if len(self.detection_history[history_key]) > self.detection_history['max_history']:
            self.detection_history[history_key].pop(0)
        
        # Calculate smoothed confidence using weighted average
        confidences = self.detection_history[history_key]
        if len(confidences) <= 1:
            return current_confidence
        
        # Give more weight to recent detections
        weights = np.linspace(0.5, 1.0, len(confidences))
        smoothed = np.average(confidences, weights=weights)
        
        return float(smoothed)
    
    def enhance_detection_with_motion(self, base_confidence: float, motion_level: float, detection_type: str) -> float:
        """
        Enhance detection confidence based on motion patterns
        
        Args:
            base_confidence: Base detection confidence
            motion_level: Motion level (0.0 to 1.0)
            detection_type: 'fall' or 'seizure'
            
        Returns:
            float: Enhanced confidence
        """
        if detection_type == 'fall':
            # For falls, high motion can indicate falling - MORE AGGRESSIVE BOOST
            if motion_level > 0.6:  # Giảm từ 0.7 xuống 0.6 để dễ trigger
                motion_boost = 0.3   # Tăng từ 0.2 lên 0.3 để boost mạnh hơn
            elif motion_level > 0.3: # Giảm từ 0.4 xuống 0.3
                motion_boost = 0.2   # Tăng từ 0.1 lên 0.2 để boost nhiều hơn
            elif motion_level > 0.1: # Thêm level trung gian
                motion_boost = 0.05  # Boost nhẹ cho motion thấp
            else:
                motion_boost = -0.05 # Giảm penalty từ -0.1 xuống -0.05
        else:  # seizure
            # For seizures, be more conservative - LESS AGGRESSIVE
            if motion_level > 0.8:   # Tăng từ 0.5 lên 0.8 để khó trigger hơn
                motion_boost = 0.1   # Giảm từ 0.15 xuống 0.1
            elif motion_level < 0.1: # Tăng từ 0.2 lên 0.1 để ít penalty hơn
                motion_boost = -0.02 # Giảm penalty từ -0.05 xuống -0.02
            else:
                motion_boost = 0.02  # Giảm từ 0.05 xuống 0.02
        
        enhanced_confidence = base_confidence + motion_boost
        return max(0.0, min(1.0, enhanced_confidence))  # Clamp to [0,1]
    
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
            
            # Initialize fall detector - MORE SENSITIVE FOR FALLS
            print("🩹 Initializing fall detection...")
            self.fall_detector = SimpleFallDetector(
                confidence_threshold=0.4  # Giảm từ 0.5 xuống 0.4 để sensitive hơn cho fall
            )
            
            # Initialize seizure detector - LESS SENSITIVE FOR SEIZURES
            print("🧠 Initializing seizure detection...")
            try:
                self.seizure_detector = VSViGSeizureDetector(
                    confidence_threshold=0.7  # Tăng từ 0.4 lên 0.7 để giảm false positive seizure
                )
                self.seizure_predictor = SeizurePredictor(
                    temporal_window=25,        # Tăng từ 20 lên 25 frames để ổn định hơn
                    alert_threshold=0.7,       # Tăng từ 0.5 lên 0.7 để khó detect hơn
                    warning_threshold=0.5      # Tăng từ 0.3 lên 0.5 để giảm warning
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
        Process dual detection (fall + seizure) for healthcare monitoring with enhanced accuracy
        
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
            # Reset confirmation frames when no person detected
            self.detection_history['fall_confirmation_frames'] = 0
            self.detection_history['seizure_confirmation_frames'] = 0
            return result
        
        # Calculate motion level for enhanced detection
        motion_level = self.calculate_motion_level(person_detections)
        self.detection_history['motion_levels'].append(motion_level)
        if len(self.detection_history['motion_levels']) > self.detection_history['max_history']:
            self.detection_history['motion_levels'].pop(0)
        
        # Update significant motion tracker
        if motion_level > 0.3:
            self.detection_history['last_significant_motion'] = time.time()
        
        # Get primary person (largest detection)
        primary_person = max(person_detections, key=lambda x: x.get('bbox', [0,0,0,0])[2] * x.get('bbox', [0,0,0,0])[3])
        person_bbox = [
            int(primary_person['bbox'][0]),
            int(primary_person['bbox'][1]),
            int(primary_person['bbox'][0] + primary_person['bbox'][2]),
            int(primary_person['bbox'][1] + primary_person['bbox'][3])
        ]
        
        # Fall detection with improvements
        fall_start = time.time()
        try:
            fall_result = self.fall_detector.detect_fall(frame, primary_person)
            base_fall_confidence = fall_result['confidence']
            
            # DEBUG: Log confidence values
            if base_fall_confidence > 0.1:  # Only log when there's some detection
                print(f"🔍 DEBUG Fall Detection - Base: {base_fall_confidence:.3f}, Motion: {motion_level:.3f}")
            
            # SIMPLIFIED: Use SimpleFallDetector result directly if confidence is high
            if base_fall_confidence >= 0.6:  # Giảm từ 0.7 xuống 0.6 để nhạy hơn
                result['fall_detected'] = True
                result['fall_confidence'] = base_fall_confidence
                
                self.stats['fall_detections'] += 1
                self.stats['last_fall_time'] = time.time()
                self.logger.info(f"🩹 Fall detected! Confidence: {base_fall_confidence:.2f} (Direct from detector)")
                print(f"✅ DIRECT FALL: Confidence {base_fall_confidence:.2f} -> Statistics updated!")
            else:
                # Apply motion enhancement and smoothing for lower confidence cases
                enhanced_fall_confidence = self.enhance_detection_with_motion(
                    base_fall_confidence, motion_level, 'fall'
                )
                smoothed_fall_confidence = self.smooth_detection_confidence(
                    enhanced_fall_confidence, 'fall'
                )
                
                # DEBUG: Log enhanced values
                if base_fall_confidence > 0.1:
                    print(f"🔍 DEBUG Enhanced: {enhanced_fall_confidence:.3f}, Smoothed: {smoothed_fall_confidence:.3f}, Threshold: 0.25")
                
                # Improved fall detection logic with confirmation frames - MORE SENSITIVE
                fall_threshold = 0.2  # Giảm từ 0.25 xuống 0.2 để nhạy hơn
                if smoothed_fall_confidence > fall_threshold:
                    self.detection_history['fall_confirmation_frames'] += 1
                else:
                    self.detection_history['fall_confirmation_frames'] = max(0, 
                        self.detection_history['fall_confirmation_frames'] - 1)
                
                # Require fewer frames for fall confirmation - FASTER RESPONSE
                min_confirmation_frames = 1  # Giảm từ 2 xuống 1 để nhanh hơn
                if self.detection_history['fall_confirmation_frames'] >= min_confirmation_frames:
                    result['fall_detected'] = True
                    result['fall_confidence'] = smoothed_fall_confidence
                    
                    self.stats['fall_detections'] += 1
                    self.stats['last_fall_time'] = time.time()
                    self.logger.info(f"🩹 Fall detected! Confidence: {smoothed_fall_confidence:.2f} (Motion: {motion_level:.2f})")
                    print(f"✅ ENHANCED FALL: Confidence {smoothed_fall_confidence:.2f} -> Statistics updated!")
                else:
                    result['fall_confidence'] = smoothed_fall_confidence
                
        except Exception as e:
            self.logger.error(f"Fall detection error: {str(e)}")
        
        self.performance['fall_detection_time'] = time.time() - fall_start
        
        # Seizure detection with improvements
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
                    
                    base_seizure_confidence = pred_result['smoothed_confidence']
                    
                    # Apply motion enhancement and additional smoothing
                    enhanced_seizure_confidence = self.enhance_detection_with_motion(
                        base_seizure_confidence, motion_level, 'seizure'
                    )
                    final_seizure_confidence = self.smooth_detection_confidence(
                        enhanced_seizure_confidence, 'seizure'
                    )
                    
                    # Improved seizure detection logic - LESS SENSITIVE
                    seizure_threshold = 0.6   # Tăng từ 0.3 lên 0.6 để giảm false positive
                    warning_threshold = 0.4   # Tăng từ 0.2 lên 0.4 để giảm warning spam
                    
                    if final_seizure_confidence > seizure_threshold:
                        self.detection_history['seizure_confirmation_frames'] += 1
                    else:
                        self.detection_history['seizure_confirmation_frames'] = max(0,
                            self.detection_history['seizure_confirmation_frames'] - 1)
                    
                    # Check for seizure detection with confirmation - STRICTER
                    min_seizure_confirmation = 5  # Tăng từ 3 lên 5 frames để chắc chắn hơn
                    if self.detection_history['seizure_confirmation_frames'] >= min_seizure_confirmation:
                        result['seizure_detected'] = True
                        result['seizure_confidence'] = final_seizure_confidence
                        
                        self.stats['seizure_detections'] += 1
                        self.stats['last_seizure_time'] = time.time()
                        self.logger.info(f"🧠 Seizure detected! Confidence: {final_seizure_confidence:.2f} (Motion: {motion_level:.2f})")
                    
                    elif final_seizure_confidence > warning_threshold and motion_level > 0.7:  # Cần cả high confidence VÀ high motion
                        result['seizure_confidence'] = final_seizure_confidence
                        self.stats['seizure_warnings'] += 1
                    else:
                        result['seizure_confidence'] = final_seizure_confidence
                
            except Exception as e:
                self.logger.error(f"Seizure detection error: {str(e)}")
                self.stats['pose_extraction_failures'] += 1
        
        self.performance['seizure_detection_time'] = time.time() - seizure_start
        
        # Enhanced alert level determination - BALANCED APPROACH
        if result['seizure_detected']:
            result['alert_level'] = 'critical'
            result['emergency_type'] = 'seizure'
            self.stats['critical_alerts'] += 1
            # Save seizure alert image
            self.save_alert_image(frame, 'seizure_detected', result['seizure_confidence'])
        elif result['fall_detected']:
            result['alert_level'] = 'high'
            result['emergency_type'] = 'fall'
            # Save fall alert image
            self.save_alert_image(frame, 'fall_detected', result['fall_confidence'])
        elif result['seizure_confidence'] > 0.45 and motion_level > 0.7:  # Giảm từ 0.5 xuống 0.45, từ 0.8 xuống 0.7 để nhạy hơn
            result['alert_level'] = 'warning'
            result['emergency_type'] = 'seizure_warning'
            # Save seizure warning image
            self.save_alert_image(frame, 'seizure_warning', result['seizure_confidence'])
        elif result['fall_confidence'] > 0.18:  # Giảm từ 0.25 xuống 0.18 để nhạy hơn
            result['alert_level'] = 'warning'
            result['emergency_type'] = 'fall_warning'
            self.save_alert_image(frame, 'fall_warning', result['fall_confidence'])
            
        if result['alert_level'] != 'normal':
            self.stats['total_alerts'] += 1
            self.stats['last_alert_time'] = time.time()
            self.stats['alert_type'] = result['alert_level']
            
            # Send to API servers
            frame_info = {
                "frame_number": self.stats['total_frames'],
                "processing_time": self.performance['total_detection_time'],
                "persons_detected": 1 if person_detections else 0,
                "motion_level": motion_level,
                "confirmation_frames": {
                    "fall": self.detection_history['fall_confirmation_frames'],
                    "seizure": self.detection_history['seizure_confirmation_frames']
                }
            }
            self.send_to_api_servers(result, frame_info)
        
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
        
        # Add detection alerts with enhanced information - BALANCED DISPLAY
        alert_y = 30
        
        # Fall detection status - MORE PROMINENT
        if detection_result['fall_detected']:
            cv2.putText(frame_vis, f"🩹 FALL DETECTED: {detection_result['fall_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        elif detection_result['fall_confidence'] > 0.25:  # Lowered fall warning threshold
            cv2.putText(frame_vis, f"⚠️ Fall Warning: {detection_result['fall_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 30
        
        # Seizure detection status - LESS PROMINENT
        if detection_result['seizure_detected']:
            cv2.putText(frame_vis, f"🧠 SEIZURE DETECTED: {detection_result['seizure_confidence']:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        elif detection_result['seizure_confidence'] > 0.5 and hasattr(self, 'detection_history') and self.detection_history['motion_levels']:
            current_motion = self.detection_history['motion_levels'][-1]
            if current_motion > 0.8:  # Only show seizure warning if high motion
                cv2.putText(frame_vis, f"⚠️ Seizure Warning: {detection_result['seizure_confidence']:.2f}",
                           (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                alert_y += 30
        
        # Motion level indicator
        if hasattr(self, 'detection_history') and self.detection_history['motion_levels']:
            current_motion = self.detection_history['motion_levels'][-1]
            motion_color = (0, 255, 0) if current_motion < 0.3 else (0, 255, 255) if current_motion < 0.7 else (0, 0, 255)
            cv2.putText(frame_vis, f"📊 Motion Level: {current_motion:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
            alert_y += 25
        
        # Confirmation frames status
        if hasattr(self, 'detection_history'):
            fall_conf = self.detection_history.get('fall_confirmation_frames', 0)
            seizure_conf = self.detection_history.get('seizure_confirmation_frames', 0)
            if fall_conf > 0 or seizure_conf > 0:
                cv2.putText(frame_vis, f"🔍 Conf Frames - Fall:{fall_conf} Seizure:{seizure_conf}",
                           (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                alert_y += 20
        
        # Seizure buffer status - with error handling and updated window size
        if self.seizure_detector and not detection_result['seizure_ready']:
            try:
                if hasattr(self.seizure_detector, 'frame_buffer'):
                    buffer_size = len(self.seizure_detector.frame_buffer)
                    frames_needed = 25 - buffer_size  # Updated to match new temporal window
                    cv2.putText(frame_vis, f"📊 Seizure Buffer: {frames_needed} frames needed",
                               (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(frame_vis, f"📊 Seizure Detection: Initializing (less sensitive)...",
                               (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            except Exception:
                cv2.putText(frame_vis, f"📊 Seizure Detection: Ready (Higher Threshold)",
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
            "🏥 DUAL DETECTION SYSTEM - ENHANCED",
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
            "🩹 FALL DETECTION - ENHANCED:",
            f"Falls Detected: {self.stats['fall_detections']}",
            f"Avg Confidence: {self.stats['fall_confidence_avg']:.2f}",
            f"Confirmation Frames: {self.detection_history.get('fall_confirmation_frames', 0)}",
            "",
            "🧠 SEIZURE DETECTION - ENHANCED:",
            f"Seizures: {self.stats['seizure_detections']}",
            f"Warnings: {self.stats['seizure_warnings']}",
            f"Pose Failures: {self.stats['pose_extraction_failures']}",
            f"Confirmation Frames: {self.detection_history.get('seizure_confirmation_frames', 0)}",
            "",
            "📊 MOTION ANALYSIS:",
            f"Current Motion: {self.detection_history['motion_levels'][-1]:.2f}" if self.detection_history['motion_levels'] else "Motion: N/A",
            f"Motion History: {len(self.detection_history['motion_levels'])}/10",
            f"Last Significant: {time.time() - self.detection_history['last_significant_motion']:.1f}s ago",
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
        if self.enable_streaming:
            # Run async version with WebSocket
            return asyncio.run(self.run_monitoring_async())
        else:
            # Run sync version without WebSocket
            return self.run_monitoring_sync()
    
    async def run_monitoring_async(self):
        """Async monitoring loop with WebSocket streaming"""
        if not self.initialize_components():
            return False
        
        # Setup WebSocket connection
        await self.setup_websocket_connection()
        
        self.running = True
        frame_count = 0
        
        print(f"\n🏥 ADVANCED HEALTHCARE MONITOR STARTED (WebSocket Mode)")
        print(f"🌐 WebSocket URL: {self.websocket_url}")
        print(f"📊 Keypoints Display: {'ON' if self.show_keypoints else 'OFF'}")
        print(f"📈 Statistics Display: {'ON' if self.show_statistics else 'OFF'}")
        print(f"🖼️ Dual Windows: {'ON' if self.show_dual_windows else 'OFF'}")
        if self.enable_api_integration:
            print(f"📤 API Integration: ON")
            print(f"   Demo API: http://localhost:8003")
            print(f"   Mobile API: http://localhost:8002")
        print("\n🎮 Keyboard Controls:")
        print(f"   'k' = toggle keypoints | 's' = toggle statistics | 'd' = toggle dual windows")
        print(f"   'h' = show help | ' ' = screenshot | 'q' = quit")
        print("=" * 70)
        
        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                frame_count += 1
                self.stats['total_frames'] = frame_count
                
                # Process frame
                processing_result = self.video_processor.process_frame(frame)
                if processing_result['processed']:
                    self.stats['frames_processed'] += 1
                    
                    # Dual detection
                    person_detections = processing_result.get('person_detections', processing_result.get('detections', []))
                    detection_result = self.process_dual_detection(
                        frame, person_detections
                    )
                    
                    # Send to WebSocket server if alert detected
                    if detection_result['alert_level'] != 'normal':
                        frame_info = {
                            "frame_number": frame_count,
                            "processing_time": processing_result.get('processing_time', 0),
                            "persons_detected": len(person_detections)
                        }
                        await self.send_detection_event(detection_result, frame_info)
                    
                    # Continue with normal visualization...
                    processing_result['person_detections'] = person_detections  # Ensure compatibility
                    self._handle_frame_display(frame, detection_result, processing_result)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if self._handle_keyboard_input(key):
                    break
                    
                # Small async delay
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n👋 Stopping healthcare monitor...")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            print(f"❌ Error: {e}")
            return False
        finally:
            await self.close_websocket()
            self.cleanup()
        
        return True
    
    def run_monitoring_sync(self):
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
                    person_detections = processing_result.get('person_detections', processing_result.get('detections', []))
                    detection_result = self.process_dual_detection(
                        frame, person_detections
                    )
                    
                    # Update statistics
                    self.update_statistics(detection_result, len(person_detections))
                    
                    # Visualize results
                    frame_vis = self.visualize_dual_detection(
                        frame, detection_result, person_detections
                    )
                    
                    # Add statistics overlay
                    frame_vis = self.draw_statistics_overlay(frame_vis)
                    
                    # Create normal camera window
                    frame_normal = self.create_normal_camera_window(
                        frame, person_detections
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
                
        except KeyboardInterrupt:
            print("👋 Stopping healthcare monitor...")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            print(f"❌ Error: {e}")
            return False
        finally:
            self.cleanup()

        return True
    
    def _handle_frame_display(self, frame, detection_result, processing_result):
        """Handle frame display for both analysis and normal views"""
        # Update statistics
        self.update_statistics(detection_result, len(processing_result['person_detections']))
        
        # Visualize results
        frame_vis = self.visualize_dual_detection(
            frame, detection_result, processing_result['person_detections']
        )
        
        # Add statistics overlay
        frame_vis = self.draw_statistics_overlay(frame_vis)
        
        # Create normal camera window
        frame_normal = self.create_normal_camera_window(
            frame, processing_result['person_detections']
        )
        
        # Display both windows
        cv2.imshow('Healthcare Monitor - Analysis View', frame_vis)
        cv2.imshow('Healthcare Monitor - Normal View', frame_normal)
        
        # Log critical events
        if detection_result['alert_level'] == 'critical':
            alert_msg = f"🚨 CRITICAL ALERT: {detection_result['emergency_type']} detected!"
            self.logger.critical(alert_msg)
            print(f"[CRITICAL] {alert_msg}")
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input, returns True if should quit"""
        if key == ord('q'):
            return True
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
            print("🎮 Keyboard Controls:")
            print("   'q' = Quit")
            print("   'k' = Toggle keypoints")
            print("   's' = Toggle statistics")
            print("   'd' = Toggle dual windows")
            print("   'h' = Show this help")
            print("   ' ' = Screenshot both windows")
        elif key == ord(' '):  # Spacebar for screenshot
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Screenshots would be handled by caller
                print(f"📸 Screenshot feature available")
            except Exception as e:
                print(f"❌ Screenshot failed: {e}")
        return False

    async def close_websocket(self):
        """Close WebSocket connection"""
        if self.websocket:
            try:
                # Send disconnect message
                disconnect_msg = {
                    "type": "disconnection",
                    "timestamp": time.time(),
                    "status": "monitoring_stopped",
                    "final_stats": {
                        "total_frames": self.stats['total_frames'],
                        "fall_detections": self.stats['fall_detections'],
                        "seizure_detections": self.stats['seizure_detections'],
                        "total_alerts": self.stats['total_alerts']
                    }
                }
                await self.websocket.send(json.dumps(disconnect_msg))
                await self.websocket.close()
                self.logger.info("WebSocket connection closed")
                print("[�] WebSocket disconnected")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
    
    async def setup_websocket_connection(self):
        """Setup WebSocket connection to server"""
        if not self.enable_streaming:
            return
            
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.logger.info(f"✅ WebSocket connected to {self.websocket_url}")
            print(f"[🌐] WebSocket connected to {self.websocket_url}")
            
            # Send initial connection message
            connect_msg = {
                "type": "connection",
                "timestamp": time.time(),
                "status": "monitoring_started",
                "device_info": {
                    "camera_type": "IMOU",
                    "detection_types": ["fall", "seizure"],
                    "resolution": self.camera_config['resolution']
                }
            }
            await self.websocket.send(json.dumps(connect_msg))
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            print(f"[❌] WebSocket connection failed: {e}")
            self.websocket = None
    
    async def send_detection_event(self, detection_result: Dict, frame_info: Optional[Dict] = None):
        """Send detection event to WebSocket server"""
        if not self.websocket or not self.enable_streaming:
            return
            
        try:
            # Get the latest saved image filename if available
            imgurl = ""
            if hasattr(self, '_last_saved_image'):
                imgurl = f"http://localhost:8003/api/demo/alerts/{getattr(self, '_last_saved_image', '')}"
            
            # Map alert level to action
            action_mapping = {
                'normal': 'monitoring',
                'warning': 'alert_warning',
                'high': 'alert_fall',
                'critical': 'alert_seizure'
            }
            
            # Map alert level to status
            status_mapping = {
                'normal': 'normal',
                'warning': 'warning', 
                'high': 'warning',
                'critical': 'danger'
            }
            
            alert_level = detection_result.get('alert_level', 'normal')
            
            # Prepare detection event data in requested format
            event_data = {
                "userId": "patient123",
                "sessionId": f"session_{int(time.time())}",
                "imageUrl": imgurl,
                "status": status_mapping.get(alert_level, 'normal'),
                "action": action_mapping.get(alert_level, 'monitoring'),
                "location": "Room A - Healthcare Monitor",
                "time": int(time.time() * 1000)  # timestamp in milliseconds
            }
            
            # Send to WebSocket server
            await self.websocket.send(json.dumps(event_data))
            
            # Log significant events
            if detection_result.get('alert_level') != 'normal':
                self.logger.info(f"🚨 Alert sent: {detection_result.get('alert_level')} - {detection_result.get('emergency_type')}")
                
        except Exception as e:
            self.logger.error(f"Failed to send detection event: {e}")
            # Try to reconnect
            await self.setup_websocket_connection()
    
    def send_to_api_servers(self, detection_result: Dict, frame_info: Optional[Dict] = None):
        """Send detection data to API servers - DISABLED for physical testing"""
        return  # Skip all API calls
            
        # Only send when there's an alert
        if detection_result.get('alert_level', 'normal') == 'normal':
            return
            
        try:
            # Get the latest saved image filename if available
            imgurl = ""
            if hasattr(self, '_last_saved_image'):
                imgurl = getattr(self, '_last_saved_image', "")
            
            # Create clean detection result without numpy arrays
            clean_detection_result = {
                'fall_detected': detection_result.get('fall_detected', False),
                'fall_confidence': float(detection_result.get('fall_confidence', 0.0)),
                'seizure_detected': detection_result.get('seizure_detected', False),
                'seizure_confidence': float(detection_result.get('seizure_confidence', 0.0)),
                'seizure_ready': detection_result.get('seizure_ready', False),
                'alert_level': detection_result.get('alert_level', 'normal'),
                'emergency_type': detection_result.get('emergency_type')
                # Exclude keypoints as they are numpy arrays
            }
            
            # Prepare data for Demo API (MO Format)
            demo_data = {
                "user_id": "patient123",
                "event_type": detection_result.get('emergency_type', 'normal'),
                "confidence": detection_result.get('fall_confidence', 0.0) if detection_result.get('fall_detected') else detection_result.get('seizure_confidence', 0.0),
                "metadata": {
                    "detection_result": clean_detection_result,
                    "frame_info": frame_info or {}
                },
                # Map to 3-value status format
                "status": self._map_alert_to_status(detection_result.get('alert_level', 'normal')),
                "imgurl": f"http://localhost:8003/api/demo/alerts/{imgurl}" if imgurl else "",
            }
            
            # Prepare data for Mobile API
            mobile_data = {
                "user_id": "patient123",
                "event_type": detection_result.get('emergency_type', 'normal'),
                "confidence": detection_result.get('fall_confidence', 0.0) if detection_result.get('fall_detected') else detection_result.get('seizure_confidence', 0.0),
                "metadata": demo_data["metadata"],
                "status": demo_data["status"],
                "imgurl": imgurl
            }
            
            # Send to Demo API in background thread
            # API calls disabled - no threading needed
            
            # print(f"📤 Demo API: http://localhost:8003")
            # print(f"📱 Mobile API: http://localhost:8002")
            
        except Exception as e:
            self.logger.error(f"Failed to send to API servers: {e}")
            print(f"❌ API send error: {e}")
    
    def _map_alert_to_status(self, alert_level: str) -> str:
        """Map alert level to 3-value status format"""
        if alert_level == 'critical':
            return 'danger'
        elif alert_level in ['high', 'warning']:
            return 'warning'
        else:
            return 'normal'
    
    def _send_to_demo_api(self, data: Dict):
        """Send data to Demo API server - DISABLED"""
        return  # Skip API calls
    
    def _send_to_mobile_api(self, data: Dict):
        """Send data to Mobile API server - DISABLED"""
        return  # Skip API calls

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
        """Print comprehensive final statistics with enhanced metrics"""
        runtime = time.time() - self.stats['start_time']
        
        print("\n" + "="*70)
        print("🏥 ADVANCED HEALTHCARE MONITOR - ENHANCED FINAL STATISTICS")
        print("="*70)
        print(f"📊 Runtime: {runtime/60:.1f} minutes")
        print(f"📊 Total Frames: {self.stats['total_frames']}")
        print(f"📊 Frames Processed: {self.stats['frames_processed']}")
        print(f"📊 Processing Efficiency: {(1-self.stats['frames_processed']/max(self.stats['total_frames'],1))*100:.1f}% skipped")
        print(f"📊 Average FPS: {self.stats['total_frames']/runtime:.1f}")
        print()
        print("🩹 ENHANCED FALL DETECTION:")
        print(f"   Falls Detected: {self.stats['fall_detections']}")
        print(f"   Average Confidence: {self.stats['fall_confidence_avg']:.2f}")
        print(f"   Final Confirmation Frames: {self.detection_history.get('fall_confirmation_frames', 0)}")
        print(f"   Last Fall: {datetime.fromtimestamp(self.stats['last_fall_time']).strftime('%H:%M:%S') if self.stats['last_fall_time'] else 'None'}")
        print()
        print("🧠 ENHANCED SEIZURE DETECTION:")
        print(f"   Seizures Detected: {self.stats['seizure_detections']}")
        print(f"   Seizure Warnings: {self.stats['seizure_warnings']}")
        print(f"   Average Confidence: {self.stats['seizure_confidence_avg']:.2f}")
        print(f"   Final Confirmation Frames: {self.detection_history.get('seizure_confirmation_frames', 0)}")
        print(f"   Pose Extraction Failures: {self.stats['pose_extraction_failures']}")
        print(f"   Last Seizure: {datetime.fromtimestamp(self.stats['last_seizure_time']).strftime('%H:%M:%S') if self.stats['last_seizure_time'] else 'None'}")
        print()
        print("📊 MOTION ANALYSIS:")
        if self.detection_history['motion_levels']:
            avg_motion = sum(self.detection_history['motion_levels']) / len(self.detection_history['motion_levels'])
            max_motion = max(self.detection_history['motion_levels'])
            print(f"   Average Motion Level: {avg_motion:.2f}")
            print(f"   Maximum Motion Level: {max_motion:.2f}")
            print(f"   Motion Samples: {len(self.detection_history['motion_levels'])}")
        else:
            print(f"   No motion data collected")
        print()
        print("🚨 EMERGENCY ALERTS:")
        print(f"   Critical Alerts: {self.stats['critical_alerts']}")
        print(f"   Total Alerts: {self.stats['total_alerts']}")
        print(f"   Current Status: {self.stats['alert_type']}")
        if self.enable_api_integration:
            print()
            print("📤 API INTEGRATION:")
            print(f"   Demo API Endpoint: http://localhost:8003")
            print(f"   Mobile API Endpoint: http://localhost:8002")
        print()
        print("⚡ DETECTION ENHANCEMENTS:")
        print(f"   ✅ Motion-based confidence boosting")
        print(f"   ✅ Temporal smoothing and filtering")
        print(f"   ✅ Multi-frame confirmation system")
        print(f"   ✅ Lowered detection thresholds for sensitivity")
        print(f"   ✅ Enhanced warning system")
        print("="*70)


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
