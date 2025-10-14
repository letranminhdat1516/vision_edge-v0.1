"""
Same Room Dual Camera Detection System
Enhanced 2 cameras in same room for complete coverage and enhanced accuracy
Optimized for healthcare emergency detection with dual camera fusion
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from service.camera_service import CameraService
from service.ai_vision_description_service import get_professional_caption_pipeline

@dataclass
class DetectionResult:
    """Detection result from single camera"""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    persons_detected: List[Dict]
    fall_confidence: float = 0.0
    seizure_confidence: float = 0.0
    motion_level: float = 0.0
    coverage_area: str = ""  # "left", "right", "center"

@dataclass
class FusedDetectionResult:
    """Fused result from multiple cameras"""
    primary_frame: np.ndarray
    secondary_frame: Optional[np.ndarray]
    combined_persons: List[Dict]
    max_fall_confidence: float
    max_seizure_confidence: float
    detection_sources: List[str]
    coverage_completeness: float  # 0-1, how much of room is covered
    consensus_score: float  # Agreement between cameras
    latest_frame: Optional[np.ndarray] = None  # Latest frame for caption generation

class SameRoomDualDetection:
    """Enhanced Dual camera detection for same room coverage - Optimized for healthcare monitoring"""
    
    def __init__(self, camera_configs: List[Dict], video_processor=None, fall_detector=None, seizure_detector=None):
        self.camera_configs = camera_configs
        self.cameras: Dict[str, CameraService] = {}
        
        # Detection services (can be passed or imported when needed)
        self.video_processor = video_processor
        self.fall_detector = fall_detector
        self.seizure_detector = seizure_detector
        
        # Threading
        self.running = False
        self.detection_threads = {}
        self.detection_results = {}  # Latest results from each camera
        self.result_locks = {}
        
        # Enhanced fusion parameters for dual camera setup
        self.consensus_threshold = 0.5    # Lowered for better sensitivity
        self.max_detection_age = 0.8      # Increased for better stability
        self.emergency_confidence_threshold = 0.3  # Lower threshold for dual camera benefit
        
        # Emergency detection state
        self.last_emergency_time = 0
        self.emergency_cooldown = 2.0  # Seconds between emergency alerts
        
        # Monitor display settings
        self.show_monitors = True
        self.show_statistics = True
        self.show_keypoints = True  # Enable keypoints by default
        self.monitor_windows = {}
        
        # Vietnamese Caption Service
        self.caption_pipeline = None
        try:
            self.caption_pipeline = get_professional_caption_pipeline()
            print(f"   🇻🇳 Vietnamese Caption: ✅ BLIP({self.caption_pipeline.blip_loaded}) + Translation({self.caption_pipeline.translator_loaded})")
        except Exception as e:
            print(f"   🇻🇳 Vietnamese Caption: ❌ {e}")
        
        # Display statistics for monitoring
        self.display_stats = {
            'start_time': time.time(),
            'total_frames': {},
            'processed_frames': {},
            'emergency_detections': 0,
            'fall_detections': 0,
            'seizure_detections': 0,
            'coverage_events': 0,
            'captions_generated': 0
        }
        
        # Initialize stats dictionaries with actual camera IDs
        for config in camera_configs:
            camera_id = config['camera_id']
            self.display_stats['total_frames'][camera_id] = 0
            self.display_stats['processed_frames'][camera_id] = 0
        
        # Statistics
        self.stats = {
            'total_fused_detections': 0,
            'left_camera_detections': 0,
            'right_camera_detections': 0,
            'consensus_detections': 0,
            'coverage_improvements': 0,
            'emergency_events': 0,
            'dual_camera_boost_applied': 0
        }
        
        print(f"🎥🎥 Enhanced Dual Detection System initialized for {len(camera_configs)} cameras")
        print(f"   🔧 Consensus threshold: {self.consensus_threshold}")
        print(f"   🔧 Emergency threshold: {self.emergency_confidence_threshold}")
        print(f"   🔧 Detection age limit: {self.max_detection_age}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced detection statistics for dual camera system"""
        total_detections = self.stats['left_camera_detections'] + self.stats['right_camera_detections']
        
        return {
            **self.stats,  # Include all original stats
            'total_camera_detections': total_detections,
            'fusion_efficiency': (self.stats['total_fused_detections'] / max(total_detections, 1)) * 100.0,
            'consensus_rate': (self.stats['consensus_detections'] / max(self.stats['total_fused_detections'], 1)) * 100.0,
            'dual_camera_active': len(self.cameras) > 1,
            'camera_count': len(self.cameras),
            'running': self.running
        }
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time status for monitoring"""
        try:
            emergency_status = self.detect_emergency_events()
            recent_detections = self.has_recent_detections()
            
            # Get latest camera states
            camera_states = {}
            for camera_id in self.cameras:
                with self.result_locks[camera_id]:
                    result = self.detection_results[camera_id]
                    camera_states[camera_id] = {
                        'active': result is not None,
                        'last_detection': result.timestamp if result else 0,
                        'fall_confidence': result.fall_confidence if result else 0,
                        'seizure_confidence': result.seizure_confidence if result else 0,
                        'persons_detected': len(result.persons_detected) if result else 0
                    }
            
            return {
                'system_status': 'ACTIVE' if self.running else 'INACTIVE',
                'emergency_detected': emergency_status['emergency'],
                'emergency_confidence': emergency_status['confidence'],
                'recent_activity': recent_detections,
                'camera_states': camera_states,
                'consensus_threshold': self.consensus_threshold,
                'dual_camera_operational': len([c for c in camera_states.values() if c['active']]) > 1,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'system_status': 'ERROR',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def has_recent_detections(self, max_age_seconds: float = 5.0) -> bool:
        """Check if there are recent detections"""
        current_time = time.time()
        
        for camera_id in self.detection_results:
            with self.result_locks[camera_id]:
                result = self.detection_results[camera_id]
                if result and (current_time - result.timestamp) <= max_age_seconds:
                    if result.fall_confidence > 0.3 or result.seizure_confidence > 0.3:
                        return True
        return False
    
    def detect_emergency_events(self) -> Dict[str, Any]:
        """Enhanced emergency event detection using dual camera fusion"""
        try:
            fused_result = self.get_fused_detection()
            if not fused_result:
                return {"emergency": False, "events": [], "confidence": 0.0}
            
            emergency_events = []
            max_confidence = 0.0
            
            # Enhanced fall detection with dual camera confidence
            if fused_result.max_fall_confidence > 0.25:  # Lower threshold for dual camera
                event_confidence = fused_result.max_fall_confidence
                
                # Boost confidence if consensus is high
                if fused_result.consensus_score > 0.7:
                    event_confidence = min(1.0, event_confidence * 1.2)
                
                # Boost confidence if multiple cameras detect
                if len(fused_result.detection_sources) > 1:
                    event_confidence = min(1.0, event_confidence * 1.15)
                
                emergency_events.append({
                    "type": "fall_detection",
                    "confidence": event_confidence,
                    "camera_sources": fused_result.detection_sources,
                    "consensus_score": fused_result.consensus_score,
                    "coverage": fused_result.coverage_completeness,
                    "timestamp": time.time()
                })
                max_confidence = max(max_confidence, event_confidence)
            
            # Enhanced seizure detection with dual camera confidence
            if fused_result.max_seizure_confidence > 0.20:  # Lower threshold for dual camera
                event_confidence = fused_result.max_seizure_confidence
                
                # Boost confidence if consensus is high
                if fused_result.consensus_score > 0.7:
                    event_confidence = min(1.0, event_confidence * 1.2)
                
                # Boost confidence if multiple cameras detect
                if len(fused_result.detection_sources) > 1:
                    event_confidence = min(1.0, event_confidence * 1.15)
                
                emergency_events.append({
                    "type": "seizure_detection", 
                    "confidence": event_confidence,
                    "camera_sources": fused_result.detection_sources,
                    "consensus_score": fused_result.consensus_score,
                    "coverage": fused_result.coverage_completeness,
                    "timestamp": time.time()
                })
                max_confidence = max(max_confidence, event_confidence)
            
            # Determine if this is a true emergency
            is_emergency = (
                max_confidence > 0.4 and  # Minimum confidence threshold
                fused_result.consensus_score > 0.5 and  # Cameras agree
                len(fused_result.combined_persons) > 0  # Person detected
            )
            
            # Apply emergency cooldown to prevent spam
            current_time = time.time()
            if is_emergency and (current_time - self.last_emergency_time) >= self.emergency_cooldown:
                self.stats['emergency_events'] += 1
                self.display_stats['emergency_detections'] += 1
                if any(e.get('type') == 'fall_detection' for e in emergency_events):
                    self.display_stats['fall_detections'] += 1
                if any(e.get('type') == 'seizure_detection' for e in emergency_events):
                    self.display_stats['seizure_detections'] += 1
                if fused_result.coverage_completeness > 0.8:
                    self.display_stats['coverage_events'] += 1
                self.last_emergency_time = current_time
            elif is_emergency and (current_time - self.last_emergency_time) < self.emergency_cooldown:
                is_emergency = False  # Suppress due to cooldown
            
            return {
                "emergency": is_emergency,
                "events": emergency_events,
                "confidence": max_confidence,
                "consensus": fused_result.consensus_score,
                "coverage": fused_result.coverage_completeness,
                "persons_detected": len(fused_result.combined_persons),
                "detection_sources": fused_result.detection_sources,
                "fused_result": fused_result,  # Include for further processing
                "cooldown_active": (current_time - self.last_emergency_time) < self.emergency_cooldown
            }
            
        except Exception as e:
            print(f"❌ Emergency detection error: {e}")
            return {"emergency": False, "events": [], "confidence": 0.0}
    
    def start(self):
        """Start detection system"""
        if self.initialize_cameras():
            self.start_dual_detection()
            return True
        return False
    
    def stop(self):
        """Stop detection system"""
        self.running = False
        
        # Close all monitor windows
        cv2.destroyAllWindows()
        
        # Wait for threads to finish
        for thread in self.detection_threads.values():
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Disconnect cameras
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except:
                pass
    
    def initialize_cameras(self) -> bool:
        """Initialize both cameras for same room"""
        success_count = 0
        
        for config in self.camera_configs:
            try:
                camera_id = config['camera_id']
                name = config['name']
                
                print(f"📹 Initializing {name}...")
                
                # Create camera service
                camera_service_config = {
                    'url': config['rtsp_url'],
                    'buffer_size': 1,
                    'fps': 15,
                    'resolution': (640, 480),
                    'auto_reconnect': True
                }
                
                camera = CameraService(camera_service_config)
                
                print(f"🔌 Attempting to connect {name}...")
                try:
                    # Use threading for timeout on Windows
                    import threading
                    import queue
                    
                    def connect_with_timeout():
                        result_queue = queue.Queue()
                        
                        def connect_worker():
                            try:
                                success = camera.connect()
                                result_queue.put(('success', success))
                            except Exception as e:
                                result_queue.put(('error', e))
                        
                        connect_thread = threading.Thread(target=connect_worker)
                        connect_thread.daemon = True
                        connect_thread.start()
                        connect_thread.join(timeout=10)  # 10 second timeout
                        
                        if connect_thread.is_alive():
                            # Timeout occurred
                            try:
                                camera.disconnect()
                            except:
                                pass
                            return False, "Connection timeout"
                        
                        if not result_queue.empty():
                            result_type, result_value = result_queue.get()
                            if result_type == 'success':
                                return result_value, None
                            else:
                                return False, str(result_value)
                        else:
                            return False, "Unknown connection error"
                    
                    success, error = connect_with_timeout()
                    
                    if success:
                        self.cameras[camera_id] = camera
                        self.detection_results[camera_id] = None
                        self.result_locks[camera_id] = threading.Lock()
                        
                        print(f"✅ {name} connected successfully")
                        success_count += 1
                    else:
                        print(f"❌ Failed to connect {name}: {error}")
                        
                        # Try alternative credentials for this camera
                        if "401" in str(error) or "Unauthorized" in str(error):
                            print(f"   🔄 Trying alternative credentials for {name}...")
                            alternative_passwords = ["123456", "admin", "L2C37340", "password", ""]
                            original_url = config['rtsp_url']
                            
                            for alt_password in alternative_passwords:
                                try:
                                    # Extract IP and construct new URL
                                    if '@' in original_url:
                                        protocol_part, rest = original_url.split('://', 1)
                                        creds_part, ip_part = rest.split('@', 1)
                                        username = creds_part.split(':')[0]
                                        
                                        alt_url = f"{protocol_part}://{username}:{alt_password}@{ip_part}"
                                        print(f"   🧪 Trying: {username}:{alt_password}@{ip_part.split(':')[0]}")
                                        
                                        # Update camera config and try again
                                        alt_camera_config = {
                                            'url': alt_url,
                                            'buffer_size': 1,
                                            'fps': 15,
                                            'resolution': (640, 480),
                                            'auto_reconnect': True
                                        }
                                        
                                        alt_camera = CameraService(alt_camera_config)
                                        
                                        # Quick test for alternative camera
                                        try:
                                            if alt_camera.connect():
                                                self.cameras[camera_id] = alt_camera
                                                self.detection_results[camera_id] = None
                                                self.result_locks[camera_id] = threading.Lock()
                                                print(f"   ✅ {name} connected with alternative credentials!")
                                                success_count += 1
                                                break
                                            else:
                                                alt_camera.disconnect()
                                        except Exception as alt_e:
                                            print(f"   ❌ Alt credential test failed: {alt_e}")
                                            try:
                                                alt_camera.disconnect()
                                            except:
                                                pass
                                            
                                except Exception as alt_e:
                                    print(f"   ❌ Alt credential failed: {alt_e}")
                                    continue
                        
                except KeyboardInterrupt:
                    print(f"⏰ {name} connection interrupted by user")
                    try:
                        camera.disconnect()
                    except:
                        pass
                except Exception as e:
                    print(f"❌ {name} connection error: {e}")
                    try:
                        camera.disconnect()
                    except:
                        pass
                    
            except Exception as e:
                print(f"❌ Error initializing {config.get('name', 'camera')}: {e}")
        
        print(f"📊 Cameras initialized: {success_count}/{len(self.camera_configs)}")
        
        if success_count == 0:
            print("❌ No cameras connected")
            return False
        elif success_count == 1:
            print("⚠️ Only 1 camera connected - Running in SINGLE CAMERA mode")
            print("   🔄 Dual detection will work with reduced coverage")
            return True
        else:
            print("✅ Multiple cameras connected - Full dual detection active")
            return True
    
    def initialize_detection_services(self):
        """Initialize detection services"""
        try:
            from service.video_processing_service import VideoProcessingService
            from service.fall_detection_service import FallDetectionService
            from service.seizure_detection_service import SeizureDetectionService
            
            self.video_processor = VideoProcessingService(120)
            self.fall_detector = FallDetectionService()
            self.seizure_detector = SeizureDetectionService()
            
            print("✅ Detection services initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize detection services: {e}")
            return False
    
    def start_dual_detection(self):
        """Start detection threads for both cameras"""
        if not self.initialize_detection_services():
            return False
            
        self.running = True
        
        for camera_id, camera in self.cameras.items():
            # Find camera config
            camera_config = None
            for config in self.camera_configs:
                if config['camera_id'] == camera_id:
                    camera_config = config
                    break
            
            if camera_config:
                thread = threading.Thread(
                    target=self._detection_loop,
                    args=(camera_id, camera, camera_config),
                    daemon=True
                )
                thread.start()
                self.detection_threads[camera_id] = thread
                
        print("🎬 Dual camera detection started")
        
        # Display monitor controls info
        if self.show_monitors:
            print("\n🖥️ DUAL CAMERA MONITORS ACTIVE")
            print(f"   📊 Statistics: {'ON' if self.show_statistics else 'OFF'}")
            print("\n🎮 Monitor Controls:")
            print("   'q' = Quit system")
            print("   's' = Toggle statistics overlay")
            print("   'm' = Toggle monitor windows")
            print("   'h' = Show help")
            print("=" * 50)
        
        return True
    
    def _detection_loop(self, camera_id: str, camera: CameraService, config: Dict):
        """Detection loop for individual camera"""
        position = config.get('position', 'unknown')
        
        # Wait for camera to stabilize after connection
        print(f"⏳ Waiting for {config.get('name', camera_id)} to stabilize...")
        time.sleep(2)
        
        # Test frame capture with retries
        frame_test_count = 0
        while self.running and frame_test_count < 10:
            test_frame = camera.get_frame()
            if test_frame is not None:
                print(f"✅ {config.get('name', camera_id)} frame capture working!")
                break
            frame_test_count += 1
            time.sleep(0.5)
        
        if frame_test_count >= 10:
            print(f"❌ {config.get('name', camera_id)} frame capture failed after retries")
            return
        
        while self.running:
            try:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame through detection pipeline
                if self.video_processor:
                    processing_result = self.video_processor.process_frame(frame)
                else:
                    processing_result = {'processed': False}
                
                # Ensure processing_result is a dict
                if not isinstance(processing_result, dict):
                    processing_result = {'processed': False, 'detections': []}
                
                if processing_result.get('processed', False):
                    # Extract person detections
                    persons = processing_result.get('person_detections', 
                                                  processing_result.get('detections', []))
                    
                    # Ensure persons is a list
                    if not isinstance(persons, list):
                        persons = []
                    
                    # Run fall detection
                    fall_confidence = 0.0
                    if persons and self.fall_detector:
                        try:
                            # Use detect_fall method for first person
                            fall_result = self.fall_detector.detect_fall(frame, persons[0])
                            fall_confidence = fall_result.get('confidence', 0.0)
                            
                            # Debug fall detection
                            if fall_confidence > 0.1:
                                print(f"🔍 {camera_id} Fall Detection: {fall_confidence:.2f}")
                                
                        except Exception as e:
                            print(f"❌ {camera_id} Fall detection error: {e}")
                            pass
                    
                    # Run seizure detection  
                    seizure_confidence = 0.0
                    if persons and self.seizure_detector:
                        try:
                            # Use detect_seizure method for first person
                            seizure_result = self.seizure_detector.detect_seizure(frame, persons[0])
                            seizure_confidence = seizure_result.get('confidence', 0.0)
                            
                            # Debug seizure detection
                            if seizure_confidence > 0.1:
                                print(f"🔍 {camera_id} Seizure Detection: {seizure_confidence:.2f}")
                                
                        except Exception as e:
                            print(f"❌ {camera_id} Seizure detection error: {e}")
                            pass
                    
                    # Create detection result
                    motion_result = processing_result.get('motion_result', {})
                    motion_pixels = 0
                    
                    # Safe motion pixel extraction
                    if isinstance(motion_result, dict):
                        motion_pixels = motion_result.get('motion_pixels', 0)
                    elif isinstance(motion_result, (int, float)):
                        motion_pixels = motion_result
                    
                    # Ensure motion_pixels is numeric
                    if not isinstance(motion_pixels, (int, float)):
                        motion_pixels = 0
                    
                    detection_result = DetectionResult(
                        camera_id=camera_id,
                        frame=frame,
                        timestamp=time.time(),
                        persons_detected=persons,
                        fall_confidence=fall_confidence,
                        seizure_confidence=seizure_confidence,
                        motion_level=motion_pixels,
                        coverage_area=position
                    )
                    
                    # Update detection results
                    with self.result_locks[camera_id]:
                        self.detection_results[camera_id] = detection_result
                        
                    # Update display statistics
                    self.display_stats['processed_frames'][camera_id] += 1
                    if fall_confidence > 0.3:
                        self.display_stats['fall_detections'] += 1
                    if seizure_confidence > 0.3:
                        self.display_stats['seizure_detections'] += 1
                        
                    # Update statistics
                    if position == 'left':
                        self.stats['left_camera_detections'] += 1
                    elif position == 'right':
                        self.stats['right_camera_detections'] += 1
                    
                    # Display monitoring windows
                    if self.show_monitors:
                        self._display_camera_monitors(camera_id, frame, detection_result, persons, config)
                
                # Update frame count
                self.display_stats['total_frames'][camera_id] += 1
                
            except Exception as e:
                print(f"❌ Detection error for {camera_id}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"   📍 Traceback: {traceback.format_exc().splitlines()[-2] if traceback.format_exc().splitlines() else 'No traceback'}")
                time.sleep(1)
    
    def _display_camera_monitors(self, camera_id: str, frame: np.ndarray, 
                               detection_result: DetectionResult, persons: list, config: Dict):
        """Display monitoring windows for individual camera"""
        try:
            camera_name = config.get('name', camera_id)
            
            # Create analysis view with detection visualization
            frame_analysis = self.visualize_camera_detection(frame, detection_result, camera_name)
            frame_analysis = self.draw_statistics_overlay(frame_analysis, camera_id)
            
            # Create normal view with minimal overlay
            frame_normal = self.create_normal_camera_window(frame, camera_name, persons)
            
            # Display windows
            analysis_window = f"{camera_name} - Analysis"
            normal_window = f"{camera_name} - Normal"
            
            cv2.imshow(analysis_window, frame_analysis)
            cv2.imshow(normal_window, frame_normal)
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.show_statistics = not self.show_statistics
                print(f"🔧 Statistics display: {'ON' if self.show_statistics else 'OFF'}")
            elif key == ord('m'):
                self.show_monitors = not self.show_monitors
                print(f"🔧 Monitor windows: {'ON' if self.show_monitors else 'OFF'}")
                if not self.show_monitors:
                    cv2.destroyAllWindows()
            elif key == ord('h'):
                print("\n🎮 Dual Camera Monitor Controls:")
                print("   'q' = Quit")
                print("   's' = Toggle statistics")
                print("   'm' = Toggle monitor windows")
                print("   'h' = Show this help")
                
        except Exception as e:
            print(f"❌ Monitor display error for {camera_id}: {e}")
    
    def get_fused_detection(self) -> Optional[FusedDetectionResult]:
        """Get enhanced fused detection result from both cameras"""
        try:
            # Collect recent detection results
            current_results = {}
            current_time = time.time()
            
            for camera_id in self.cameras.keys():
                with self.result_locks[camera_id]:
                    result = self.detection_results[camera_id]
                    if (result and 
                        current_time - result.timestamp < self.max_detection_age):
                        current_results[camera_id] = result
            
            if not current_results:
                return None
            
            # Perform enhanced fusion for dual camera
            fused_result = self._fuse_detections(current_results)
            
            if fused_result:
                self.stats['total_fused_detections'] += 1
                
                # Check for consensus and apply dual camera boost
                if fused_result.consensus_score >= self.consensus_threshold:
                    self.stats['consensus_detections'] += 1
                
                # Track dual camera boost applications
                if len(current_results) > 1:
                    self.stats['coverage_improvements'] += 1
                    if (fused_result.max_fall_confidence > 0.3 or 
                        fused_result.max_seizure_confidence > 0.3):
                        self.stats['dual_camera_boost_applied'] += 1
            
            return fused_result
            
        except Exception as e:
            print(f"❌ Enhanced fusion error: {e}")
            return None
    
    def _fuse_detections(self, results: Dict[str, DetectionResult]) -> Optional[FusedDetectionResult]:
        """Fuse detection results from multiple cameras - Optimized for dual camera detection"""
        if not results:
            return None
        
        try:
            # Enhanced fusion for dual camera system
            camera_results = list(results.values())
            
            # Get primary result - prioritize camera with higher confidence
            primary_result = max(camera_results, 
                               key=lambda r: max(r.fall_confidence, r.seizure_confidence, 0.1))
            
            # Smart person detection fusion - Remove duplicates and enhance coverage
            combined_persons = []
            person_positions = {}
            
            for result in camera_results:
                for person in result.persons_detected:
                    if not isinstance(person, dict):
                        print(f"⚠️ Warning: person is not dict: {type(person)}")
                        continue
                        
                    bbox = person.get('bbox', {})
                    
                    # Handle different bbox formats
                    center_x, center_y = 0, 0
                    
                    if isinstance(bbox, dict):
                        # Dict format: {'x': 10, 'y': 20, 'width': 100, 'height': 200}
                        center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
                        center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
                    elif isinstance(bbox, list) and len(bbox) >= 4:
                        # List format: [x, y, width, height] or [x1, y1, x2, y2]
                        x, y, w_or_x2, h_or_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        if w_or_x2 > x and h_or_y2 > y:
                            # [x1, y1, x2, y2] format
                            center_x = (x + w_or_x2) / 2
                            center_y = (y + h_or_y2) / 2
                        else:
                            # [x, y, width, height] format  
                            center_x = x + w_or_x2 / 2
                            center_y = y + h_or_y2 / 2
                    else:
                        print(f"⚠️ Warning: unrecognized bbox format: {type(bbox)}, {bbox}")
                        continue
                    
                    # Enhanced position-based deduplication
                    position_key = f"{int(center_x/40)}_{int(center_y/40)}"
                    
                    if position_key not in person_positions:
                        # Add confidence boost from camera coverage
                        enhanced_person = person.copy()
                        camera_coverage_boost = 0.05 if len(camera_results) > 1 else 0.0
                        if 'confidence' in enhanced_person:
                            enhanced_person['confidence'] = min(1.0, enhanced_person['confidence'] + camera_coverage_boost)
                        
                        combined_persons.append(enhanced_person)
                        person_positions[position_key] = result.camera_id
                    else:
                        # Merge confidence if same person detected by multiple cameras
                        for existing_person in combined_persons:
                            existing_bbox = existing_person.get('bbox', {})
                            existing_center_x, existing_center_y = 0, 0
                            
                            if isinstance(existing_bbox, dict):
                                existing_center_x = existing_bbox.get('x', 0) + existing_bbox.get('width', 0) / 2
                                existing_center_y = existing_bbox.get('y', 0) + existing_bbox.get('height', 0) / 2
                            elif isinstance(existing_bbox, list) and len(existing_bbox) >= 4:
                                x, y, w_or_x2, h_or_y2 = existing_bbox[0], existing_bbox[1], existing_bbox[2], existing_bbox[3]
                                if w_or_x2 > x and h_or_y2 > y:
                                    existing_center_x = (x + w_or_x2) / 2
                                    existing_center_y = (y + h_or_y2) / 2
                                else:
                                    existing_center_x = x + w_or_x2 / 2
                                    existing_center_y = y + h_or_y2 / 2
                            
                            existing_position_key = f"{int(existing_center_x/40)}_{int(existing_center_y/40)}"
                            
                            if existing_position_key == position_key:
                                if 'confidence' in existing_person and 'confidence' in person:
                                    existing_person['confidence'] = min(1.0, (existing_person['confidence'] + person['confidence']) / 2 + 0.1)
                                break
            
            # Enhanced confidence calculation with dual camera boost
            max_fall_confidence = max([r.fall_confidence for r in camera_results], default=0.0)
            max_seizure_confidence = max([r.seizure_confidence for r in camera_results], default=0.0)
            
            # Apply dual camera confidence boost for critical events
            if len(camera_results) > 1:
                dual_camera_boost = 0.15
                if max_fall_confidence > 0.3:
                    max_fall_confidence = min(1.0, max_fall_confidence + dual_camera_boost)
                if max_seizure_confidence > 0.3:
                    max_seizure_confidence = min(1.0, max_seizure_confidence + dual_camera_boost)
            
            # Calculate consensus score (agreement between cameras)
            consensus_score = self._calculate_consensus(results)
            
            # Enhanced coverage completeness for dual camera setup
            coverage_areas = [r.coverage_area for r in camera_results]
            unique_areas = set(filter(None, coverage_areas))
            coverage_completeness = len(unique_areas) / 2.0 if len(unique_areas) > 0 else 0.5
            
            # Get secondary frame if available
            secondary_frame = None
            if len(camera_results) > 1:
                secondary_results = [r for r in camera_results if r != primary_result]
                if secondary_results:
                    secondary_frame = secondary_results[0].frame
            
            return FusedDetectionResult(
                primary_frame=primary_result.frame,
                secondary_frame=secondary_frame,
                combined_persons=combined_persons,
                max_fall_confidence=max_fall_confidence,
                max_seizure_confidence=max_seizure_confidence,
                detection_sources=[r.camera_id for r in camera_results],
                coverage_completeness=coverage_completeness,
                consensus_score=consensus_score,
                latest_frame=primary_result.frame  # Use primary frame for caption
            )
            
        except Exception as e:
            print(f"❌ Detailed fusion error: {e}")
            print(f"   Results type: {type(results)}")
            if results:
                for key, value in results.items():
                    print(f"   {key}: {type(value)}")
            import traceback
            traceback.print_exc()
            return None
        
        return FusedDetectionResult(
            primary_frame=primary_result.frame,
            secondary_frame=secondary_frame,
            combined_persons=combined_persons,
            max_fall_confidence=max_fall_confidence,
            max_seizure_confidence=max_seizure_confidence,
            detection_sources=[r.camera_id for r in results.values()],
            coverage_completeness=coverage_completeness,
            consensus_score=consensus_score,
            latest_frame=primary_result.frame  # Use primary frame for caption
        )
    
    def _calculate_consensus(self, results: Dict[str, DetectionResult]) -> float:
        """Calculate consensus score between camera detections - Enhanced for dual camera"""
        if len(results) < 2:
            return 1.0  # Single camera, perfect consensus
        
        camera_results = list(results.values())
        
        # Enhanced person count agreement for dual camera
        person_counts = [len(r.persons_detected) for r in camera_results]
        max_count = max(person_counts) if person_counts else 1
        count_diff = abs(max(person_counts) - min(person_counts)) if len(person_counts) > 1 else 0
        count_agreement = 1.0 - (count_diff / max(max_count, 1))
        
        # Enhanced confidence agreement with threshold consideration
        fall_confidences = [r.fall_confidence for r in camera_results]
        seizure_confidences = [r.seizure_confidence for r in camera_results]
        
        # Calculate agreement based on both absolute difference and threshold crossing
        def calculate_confidence_agreement(confidences, threshold=0.3):
            if len(confidences) < 2:
                return 1.0
            
            # Absolute difference agreement
            max_conf = max(confidences)
            min_conf = min(confidences)
            diff_agreement = 1.0 - abs(max_conf - min_conf)
            
            # Threshold crossing agreement (both above or both below threshold)
            above_threshold = [c >= threshold for c in confidences]
            threshold_agreement = 1.0 if all(above_threshold) or not any(above_threshold) else 0.5
            
            return (diff_agreement + threshold_agreement) / 2.0
        
        fall_agreement = calculate_confidence_agreement(fall_confidences, 0.3)
        seizure_agreement = calculate_confidence_agreement(seizure_confidences, 0.3)
        
        # Motion consistency check for dual camera
        motion_levels = [r.motion_level for r in camera_results]
        motion_agreement = 1.0
        if len(motion_levels) > 1:
            motion_diff = abs(max(motion_levels) - min(motion_levels))
            motion_agreement = max(0.0, 1.0 - motion_diff / 10000.0)  # Normalize motion difference
        
        # Weighted consensus calculation optimized for healthcare detection
        consensus = (
            count_agreement * 0.3 +      # Person count consistency
            fall_agreement * 0.3 +       # Fall detection agreement  
            seizure_agreement * 0.3 +    # Seizure detection agreement
            motion_agreement * 0.1       # Motion consistency
        )
        
        return max(0.0, min(1.0, consensus))
    
    def create_dual_display(self, fused_result: FusedDetectionResult) -> Dict[str, np.ndarray]:
        """Create dual camera display"""
        try:
            primary_frame = fused_result.primary_frame.copy()
            
            # Add overlay information
            overlay_text = [
                f"Persons: {len(fused_result.combined_persons)}",
                f"Fall: {fused_result.max_fall_confidence:.2f}",
                f"Seizure: {fused_result.max_seizure_confidence:.2f}",
                f"Consensus: {fused_result.consensus_score:.2f}",
                f"Coverage: {fused_result.coverage_completeness:.1%}",
                f"Sources: {len(fused_result.detection_sources)}"
            ]
            
            # Add text overlay
            y_offset = 30
            for text in overlay_text:
                cv2.putText(primary_frame, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Draw person bounding boxes
            for person in fused_result.combined_persons:
                bbox = person.get('bbox', {})
                if bbox:
                    x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                    cv2.rectangle(primary_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            
            result = {"primary": primary_frame}
            
            # Add secondary frame if available
            if fused_result.secondary_frame is not None:
                secondary_frame = fused_result.secondary_frame.copy()
                cv2.putText(secondary_frame, "Secondary View", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                result["secondary"] = secondary_frame
            
            return result
            
        except Exception as e:
            print(f"❌ Display creation error: {e}")
            return {"primary": fused_result.primary_frame}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dual detection statistics"""
        return {
            'total_fused_detections': self.stats['total_fused_detections'],
            'left_camera_detections': self.stats['left_camera_detections'],
            'right_camera_detections': self.stats['right_camera_detections'],
            'consensus_detections': self.stats['consensus_detections'],
            'coverage_improvements': self.stats['coverage_improvements'],
            'cameras_active': len([c for c in self.cameras.values() if c]),
            'fusion_rate': self.stats['consensus_detections'] / max(self.stats['total_fused_detections'], 1)
        }
    
    def stop_detection(self):
        """Stop dual detection"""
        self.running = False
        
        for thread in self.detection_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
                
        print("🛑 Dual detection stopped")
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        self.stop_detection()
        
        for camera in self.cameras.values():
            camera.disconnect()
            
        print("🔌 All cameras disconnected")
    
    def visualize_camera_detection(self, frame: np.ndarray, detection_result: DetectionResult, 
                                 camera_name: str) -> np.ndarray:
        """Visualize detection results for individual camera"""
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        
        # Camera title
        cv2.putText(frame_vis, f"{camera_name} - Detection View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw person detections
        for person in detection_result.persons_detected:
            bbox = person.get('bbox', [])
            confidence = person.get('confidence', 0.0)
            
            if len(bbox) >= 4:
                # Handle different bbox formats
                if isinstance(bbox, dict):
                    x, y, w_box, h_box = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                elif len(bbox) == 4:
                    x, y, w_box, h_box = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    continue
                
                # Determine color based on detection confidence
                if detection_result.fall_confidence > 0.6 or detection_result.seizure_confidence > 0.6:
                    color = (0, 0, 255)  # Red for emergency
                elif detection_result.fall_confidence > 0.3 or detection_result.seizure_confidence > 0.3:
                    color = (0, 165, 255)  # Orange for warning  
                else:
                    color = (0, 255, 0)  # Green for normal
                
                cv2.rectangle(frame_vis, (int(x), int(y)), (int(x + w_box), int(y + h_box)), color, 2)
                cv2.putText(frame_vis, f"Person: {confidence:.2f}", (int(x), int(y - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw keypoints if enabled and available
                if self.show_keypoints and 'keypoints' in person:
                    keypoints = person['keypoints']
                    self._draw_keypoints(frame_vis, keypoints, color)
        
        # Detection status overlay
        alert_y = 70
        
        # Fall detection status
        if detection_result.fall_confidence > 0.6:
            cv2.putText(frame_vis, f"🩹 FALL DETECTED: {detection_result.fall_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif detection_result.fall_confidence > 0.3:
            cv2.putText(frame_vis, f"⚠️ Fall Warning: {detection_result.fall_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame_vis, f"Fall Detection: {detection_result.fall_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        alert_y += 30
        
        # Seizure detection status
        if detection_result.seizure_confidence > 0.6:
            cv2.putText(frame_vis, f"🧠 SEIZURE DETECTED: {detection_result.seizure_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif detection_result.seizure_confidence > 0.3:
            cv2.putText(frame_vis, f"⚠️ Seizure Warning: {detection_result.seizure_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame_vis, f"Seizure Detection: {detection_result.seizure_confidence:.2f}",
                       (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        alert_y += 30
        
        # Motion level
        motion_color = (0, 255, 0) if detection_result.motion_level < 500 else (0, 255, 255) if detection_result.motion_level < 1500 else (0, 0, 255)
        cv2.putText(frame_vis, f"Motion Level: {detection_result.motion_level:.0f}",
                   (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        alert_y += 30
        
        # Person count
        cv2.putText(frame_vis, f"Persons: {len(detection_result.persons_detected)}",
                   (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.fromtimestamp(detection_result.timestamp).strftime("%H:%M:%S")
        cv2.putText(frame_vis, timestamp, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_vis
    
    def create_normal_camera_window(self, frame: np.ndarray, camera_name: str, 
                                  person_detections: list) -> np.ndarray:
        """Create normal camera window with minimal overlay"""
        frame_normal = frame.copy()
        
        # Camera title
        cv2.putText(frame_normal, f"{camera_name} - Normal View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw basic person detections
        for person in person_detections:
            bbox = person.get('bbox', [])
            confidence = person.get('confidence', 0.0)
            
            if len(bbox) >= 4:
                # Handle different bbox formats
                if isinstance(bbox, dict):
                    x, y, w_box, h_box = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                elif len(bbox) == 4:
                    x, y, w_box, h_box = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    continue
                
                color = (0, 255, 0)  # Always green for normal view
                cv2.rectangle(frame_normal, (int(x), int(y)), (int(x + w_box), int(y + h_box)), color, 2)
                cv2.putText(frame_normal, f"Person: {confidence:.2f}", (int(x), int(y - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Basic status
        person_count = len(person_detections)
        cv2.putText(frame_normal, f"Persons Detected: {person_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame_normal, timestamp, (10, frame_normal.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_normal
    
    def draw_statistics_overlay(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Draw real-time statistics overlay"""
        if not self.show_statistics:
            return frame
        
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        
        # Statistics panel
        panel_width = 280
        panel_height = 300
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame_vis.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        frame_vis = cv2.addWeighted(frame_vis, 0.7, overlay, 0.3, 0)
        
        # Calculate runtime statistics
        runtime = time.time() - self.display_stats['start_time']
        total_frames = self.display_stats['total_frames'][camera_id]
        processed_frames = self.display_stats['processed_frames'][camera_id]
        fps = total_frames / runtime if runtime > 0 else 0
        
        stats_text = [
            "DUAL CAMERA MONITOR",
            f"Camera: {camera_id}",
            f"Runtime: {runtime/60:.1f}m",
            "",
            f"📊 FRAME STATS:",
            f"Total: {total_frames}",
            f"Processed: {processed_frames}",
            f"FPS: {fps:.1f}",
            f"Efficiency: {(processed_frames/max(total_frames,1)*100):.1f}%",
            "",
            f"🚨 DETECTIONS:",
            f"Emergency: {self.display_stats['emergency_detections']}",
            f"Falls: {self.display_stats['fall_detections']}",
            f"Seizures: {self.display_stats['seizure_detections']}",
            f"Coverage: {self.display_stats['coverage_events']}",
            "",
            f"🔄 FUSION STATS:",
            f"Consensus: {self.stats['consensus_detections']}",
            f"Boost Applied: {self.stats['dual_camera_boost_applied']}",
            f"Total Fused: {self.stats['total_fused_detections']}"
        ]
        
        # Draw statistics
        text_y = panel_y + 20
        for line in stats_text:
            if line == "":
                text_y += 5
                continue
                
            # Choose color based on content
            color = (255, 255, 255)  # White default
            if "🚨" in line or "Emergency" in line:
                color = (0, 0, 255)  # Red
            elif "🔄" in line or "FUSION" in line:
                color = (0, 255, 255)  # Yellow
            elif "📊" in line:
                color = (0, 255, 0)  # Green
            
            cv2.putText(frame_vis, line, (panel_x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            text_y += 15
        
        return frame_vis
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: list, color: tuple):
        """Draw COCO keypoints and skeleton on frame"""
        if not keypoints or len(keypoints) < 51:  # COCO has 17 keypoints * 3 (x,y,confidence)
            return
        
        # COCO keypoint names
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # COCO skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Extract keypoint coordinates
        kpts = []
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints):
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                kpts.append([x, y, conf])
        
        # Draw skeleton connections
        for connection in skeleton:
            kpt_a, kpt_b = connection[0] - 1, connection[1] - 1  # Convert to 0-based index
            if (0 <= kpt_a < len(kpts) and 0 <= kpt_b < len(kpts) and 
                kpts[kpt_a][2] > 0.3 and kpts[kpt_b][2] > 0.3):  # Confidence threshold
                
                pt_a = (int(kpts[kpt_a][0]), int(kpts[kpt_a][1]))
                pt_b = (int(kpts[kpt_b][0]), int(kpts[kpt_b][1]))
                cv2.line(frame, pt_a, pt_b, color, 2)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.3:  # Only draw confident keypoints
                center = (int(x), int(y))
                cv2.circle(frame, center, 4, color, -1)
                cv2.circle(frame, center, 6, (255, 255, 255), 2)

# Helper function
def create_same_room_camera_configs() -> List[Dict]:
    """Create camera configurations for same room setup"""
    from service.config_loader import config_loader
    
    config = config_loader.load_system_config()
    cameras_config = config.get('database', {}).get('cameras', {})
    
    camera_configs = []
    for camera_key, camera_data in cameras_config.items():
        camera_configs.append(camera_data)
    
    return camera_configs

# Global instance
dual_detection_system = None

def get_dual_detection_system() -> SameRoomDualDetection:
    """Get or create dual detection system"""
    global dual_detection_system
    
    if dual_detection_system is None:
        camera_configs = create_same_room_camera_configs()
        dual_detection_system = SameRoomDualDetection(camera_configs)
        
        if dual_detection_system.initialize_cameras():
            dual_detection_system.start_dual_detection()
        else:
            print("❌ Failed to initialize dual detection system")
    
    return dual_detection_system
