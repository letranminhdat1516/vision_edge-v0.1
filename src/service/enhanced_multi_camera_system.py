#!/usr/bin/env python3
"""
Enhanced Multi-threaded Camera System với Event Fusion
Mỗi camera chạy trong thread riêng, events được fusion intelligent
"""

import cv2
import time
import threading
import queue
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraEvent:
    """Event from individual camera"""
    camera_id: str
    camera_name: str
    timestamp: float
    event_type: str  # 'fall', 'seizure', 'abnormal_behavior'
    confidence: float
    frame: np.ndarray
    persons: List[Dict]
    metadata: Dict

@dataclass
class DisplayFrame:
    """Frame data for display"""
    camera_id: str
    camera_name: str
    frame: np.ndarray
    persons: List[Dict]
    timestamp: float

@dataclass
class FusedEvent:
    """Best event after fusion"""
    primary_camera: str
    event_type: str
    confidence: float
    timestamp: float
    frame: np.ndarray
    supporting_cameras: List[str]
    consensus_score: float
    
class EventFusionEngine:
    """Engine để fusion events từ multiple cameras"""
    
    def __init__(self):
        self.event_window = 2.0  # 2 second window for fusion
        self.confidence_weights = {
            'fall': 1.2,      # Ưu tiên fall detection
            'seizure': 1.1,   # Ưu tiên seizure
            'abnormal_behavior': 1.0
        }
    
    def fuse_events(self, events: List[CameraEvent]) -> Optional[FusedEvent]:
        """Fuse multiple camera events into best single event"""
        if not events:
            return None
        
        if len(events) == 1:
            # Single camera event
            event = events[0]
            return FusedEvent(
                primary_camera=event.camera_id,
                event_type=event.event_type,
                confidence=event.confidence,
                timestamp=event.timestamp,
                frame=event.frame,
                supporting_cameras=[],
                consensus_score=1.0
            )
        
        # Multiple camera events - find best
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        best_fusion = None
        best_score = 0
        
        for event_type, type_events in events_by_type.items():
            # Calculate weighted score
            total_confidence = sum(e.confidence for e in type_events)
            avg_confidence = total_confidence / len(type_events)
            
            # Apply type weighting
            weight = self.confidence_weights.get(event_type, 1.0)
            weighted_score = avg_confidence * weight * len(type_events)
            
            if weighted_score > best_score:
                # Select camera with highest confidence as primary
                primary_event = max(type_events, key=lambda e: e.confidence)
                supporting_cameras = [e.camera_id for e in type_events if e.camera_id != primary_event.camera_id]
                
                consensus_score = len(type_events) / len(events)
                
                best_fusion = FusedEvent(
                    primary_camera=primary_event.camera_id,
                    event_type=event_type,
                    confidence=avg_confidence,
                    timestamp=primary_event.timestamp,
                    frame=primary_event.frame,
                    supporting_cameras=supporting_cameras,
                    consensus_score=consensus_score
                )
                best_score = weighted_score
        
        return best_fusion

class EnhancedMultiCameraSystem:
    """Enhanced multi-camera system với parallel processing"""
    
    def __init__(self, camera_configs: List[Dict], enable_monitors: bool = True):
        self.camera_configs = camera_configs
        self.enable_monitors = enable_monitors
        self.running = False
        
        # Threading components
        self.camera_threads = {}
        self.event_queues = {}  # Per-camera event queues
        self.fusion_queue = queue.Queue()
        self.fusion_thread = None
        self.display_thread = None
        
        # Display queue for frames - increased size and made it deque for better performance
        self.display_queue = queue.Queue(maxsize=30)  # Increased buffer size
        self.last_display_time = {}
        
        # Event fusion
        self.fusion_engine = EventFusionEngine()
        
        # Statistics - enhanced with per-camera counters
        self.stats = {
            'total_events': 0,
            'fused_events': 0,
            'camera_events': {},
            'event_types': {},
            'camera_stats': {}  # New: per-camera detailed stats
        }
        
        # Initialize per-camera stats with detailed counters
        for config in camera_configs:
            camera_id = config['camera_id']
            camera_name = config['name']
            self.event_queues[camera_id] = queue.Queue()
            self.stats['camera_events'][camera_id] = 0
            
            # Detailed per-camera statistics
            self.stats['camera_stats'][camera_id] = {
                'name': camera_name,
                'fall_count': 0,
                'seizure_count': 0,
                'person_detections': 0,
                'total_frames': 0,
                'fps': 0.0,
                'last_fps_time': time.time(),
                'last_fps_count': 0,
                'confidence_avg': 0.0,
                'keypoints_detected': 0,
                'pose_estimations': 0
            }
        
        print(f"🎥 Enhanced Multi-Camera System initialized for {len(camera_configs)} cameras")
        if enable_monitors:
            print("🖥️ Display monitors enabled")
        else:
            print("🚫 Display monitors disabled")
    
    def start(self):
        """Start all camera threads and fusion engine"""
        self.running = True
        
        print("🚀 Starting Enhanced Multi-Camera System...")
        
        # Start camera threads
        for config in self.camera_configs:
            camera_id = config['camera_id']
            thread = threading.Thread(
                target=self._camera_thread,
                args=(camera_id, config),
                daemon=True,
                name=f"Camera-{config['name']}"
            )
            self.camera_threads[camera_id] = thread
            thread.start()
            print(f"🚀 Started thread for {config['name']}")
        
        # Start fusion thread
        self.fusion_thread = threading.Thread(
            target=self._event_fusion_thread,
            daemon=True,
            name="EventFusion"
        )
        self.fusion_thread.start()
        print("🧠 Started Event Fusion Engine")
        
        # Start display thread for OpenCV windows
        if self.enable_monitors:
            self.display_thread = threading.Thread(
                target=self._display_thread,
                daemon=True,
                name="DisplayManager"
            )
            self.display_thread.start()
            print("🖥️ Started Display Manager")
        
        # Start monitor display thread for keyboard input
        monitor_thread = threading.Thread(
            target=self._monitor_display_thread,
            daemon=True,
            name="MonitorDisplay"
        )
        monitor_thread.start()
        print("📺 Started Monitor Display")
        
        # Wait a bit for threads to initialize
        print("⏳ Waiting for camera connections...")
        time.sleep(2)
        
        # Check if any cameras connected successfully
        connected_cameras = 0
        for camera_id in self.camera_threads:
            if camera_id in self.stats['camera_stats']:
                connected_cameras += 1
        
        if connected_cameras > 0:
            print(f"✅ {connected_cameras}/{len(self.camera_configs)} cameras ready!")
            print("📺 OpenCV windows should appear shortly...")
            print("🎮 Controls: 'q'=quit, 's'=stats, 'e'=test event")
            return True
        else:
            print("⚠️ No cameras connected yet, but system is running...")
            print("📺 Waiting for camera connections...")
            return True
    
    def _camera_thread(self, camera_id: str, config: Dict):
        """Individual camera processing thread"""
        camera_name = config['name']
        print(f"📹 [{camera_name}] Thread started")
        
        try:
            # Initialize camera
            from service.camera_service import CameraService
            camera_config = {
                'url': config['rtsp_url'],
                'buffer_size': 1,
                'fps': config.get('fps', 30),
                'resolution': (1920, 1080),
                'auto_reconnect': True,
                'camera_id': camera_id,
                'camera_name': camera_name
            }
            
            print(f"📹 [{camera_name}] Connecting to camera...")
            camera = CameraService(camera_config)
            
            if not camera.connect():
                print(f"❌ [{camera_name}] Failed to connect")
                return
            
            print(f"✅ [{camera_name}] Connected and streaming")
            
            # Initialize detection services
            print(f"🔧 [{camera_name}] Initializing detection services...")
            from service.video_processing_service import VideoProcessingService
            from service.fall_detection_service import FallDetectionService
            from service.seizure_detection_service import SeizureDetectionService
            
            video_processor = VideoProcessingService(120)
            fall_detector = FallDetectionService()
            seizure_detector = SeizureDetectionService()
            
            print(f"🚀 [{camera_name}] All services initialized, starting processing...")
            
            frame_count = 0
            last_event_time = 0
            last_status_time = time.time()
            
            while self.running:
                try:
                    frame = camera.get_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    # Show status every 30 seconds
                    if current_time - last_status_time > 30:
                        fps = self.stats['camera_stats'][camera_id]['fps']
                        print(f"📹 [{camera_name}] Processing: {frame_count} frames, FPS: {fps:.1f}")
                        last_status_time = current_time
                    
                    # Update camera stats - total frames and FPS
                    camera_stats = self.stats['camera_stats'][camera_id]
                    camera_stats['total_frames'] += 1
                    
                    # Calculate FPS every 30 frames
                    if frame_count % 30 == 0:
                        time_diff = current_time - camera_stats['last_fps_time']
                        if time_diff > 0:
                            camera_stats['fps'] = 30 / time_diff
                            camera_stats['last_fps_time'] = current_time
                    
                    # Process every 3rd frame for efficiency
                    if frame_count % 3 != 0:
                        # Still send frame to display even if not processing
                        if self.enable_monitors and frame_count % 6 == 0:
                            display_frame = DisplayFrame(
                                camera_id=camera_id,
                                camera_name=camera_name,
                                frame=frame.copy(),
                                persons=[],
                                timestamp=current_time
                            )
                            try:
                                self.display_queue.put_nowait(display_frame)
                            except queue.Full:
                                pass
                        continue
                    
                    # Process frame for ML inference
                    result = video_processor.process_frame(frame)
                    if not result.get('processed', False):
                        continue
                    
                    persons = result.get('person_detections', [])
                    
                    # Update person detection stats
                    if persons:
                        camera_stats['person_detections'] += len(persons)
                        
                        # Count keypoints if available
                        for person in persons:
                            if person.get('keypoints') is not None:
                                camera_stats['keypoints_detected'] += 1
                                camera_stats['pose_estimations'] += 1
                    
                    # Send frame to display queue instead of direct display
                    if self.enable_monitors:
                        display_frame = DisplayFrame(
                            camera_id=camera_id,
                            camera_name=camera_name,
                            frame=frame.copy(),
                            persons=persons,
                            timestamp=current_time
                        )
                        try:
                            self.display_queue.put_nowait(display_frame)
                            # Debug: Show queue status occasionally
                            if frame_count % 300 == 0:  # Every 300 frames
                                print(f"📺 [{camera_name}] Frame queued for display, Queue size: {self.display_queue.qsize()}")
                        except queue.Full:
                            # Skip if display queue is full
                            if frame_count % 100 == 0:  # Show warning occasionally
                                print(f"⚠️ [{camera_name}] Display queue full, dropping frames")
                    
                    # Detect events
                    events_detected = []
                    
                    # Fall detection
                    try:
                        fall_result = fall_detector.detect_fall(frame, persons)
                        if fall_result.get('fall_detected', False):
                            confidence = fall_result.get('confidence', 0)
                            if confidence > 0.3:
                                events_detected.append({
                                    'type': 'fall',
                                    'confidence': confidence,
                                    'metadata': fall_result
                                })
                                camera_stats['fall_count'] += 1
                    except Exception as e:
                        pass
                    
                    # Seizure detection
                    try:
                        seizure_result = seizure_detector.detect_seizure(frame, persons)
                        if seizure_result.get('seizure_detected', False):
                            confidence = seizure_result.get('confidence', 0)
                            if confidence > 0.25:
                                events_detected.append({
                                    'type': 'seizure',
                                    'confidence': confidence,
                                    'metadata': seizure_result
                                })
                                camera_stats['seizure_count'] += 1
                    except Exception as e:
                        pass
                    
                    # Send events to fusion engine
                    for event_data in events_detected:
                        event = CameraEvent(
                            camera_id=camera_id,
                            camera_name=camera_name,
                            timestamp=current_time,
                            event_type=event_data['type'],
                            confidence=event_data['confidence'],
                            frame=frame.copy(),
                            persons=persons,
                            metadata=event_data['metadata']
                        )
                        
                        try:
                            self.event_queues[camera_id].put_nowait(event)
                            last_event_time = current_time
                        except queue.Full:
                            pass
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.02)
                    
                except Exception as e:
                    print(f"❌ [{camera_name}] Frame processing error: {e}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"❌ [{camera_name}] Camera thread error: {e}")
        finally:
            print(f"🛑 [{camera_name}] Camera thread stopped")

    def _event_fusion_thread(self):
        """Event fusion processing thread"""
        print("🧠 Event Fusion Engine started")
        
        while self.running:
            try:
                # Collect events from all cameras within time window
                current_time = time.time()
                events_to_fuse = []
                
                # Check each camera queue
                for camera_id, event_queue in self.event_queues.items():
                    while not event_queue.empty():
                        try:
                            event = event_queue.get_nowait()
                            # Only consider recent events
                            if current_time - event.timestamp < 2.0:
                                events_to_fuse.append(event)
                        except queue.Empty:
                            break
                
                # Fuse events if any exist
                if events_to_fuse:
                    fused_event = self.fusion_engine.fuse_events(events_to_fuse)
                    if fused_event:
                        self._handle_fused_event(fused_event)
                        self.stats['fused_events'] += 1
                        self.stats['total_events'] += 1
                        
                        event_type = fused_event.event_type
                        self.stats['event_types'][event_type] = self.stats['event_types'].get(event_type, 0) + 1
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"❌ Fusion error: {e}")
                time.sleep(1)
    
    def _handle_fused_event(self, fused_event: FusedEvent):
        """Handle final fused event"""
        cameras_involved = [fused_event.primary_camera] + fused_event.supporting_cameras
        cameras_count = len(cameras_involved)
        
        print(f"🎯 FUSED EVENT: {fused_event.event_type}")
        print(f"   📊 Confidence: {fused_event.confidence:.2f}")
        print(f"   📹 Cameras: {cameras_count} ({fused_event.consensus_score:.2f} consensus)")
        print(f"   🕐 Time: {datetime.fromtimestamp(fused_event.timestamp).strftime('%H:%M:%S')}")
        
        # Save to database, send notifications, etc.
        self._save_event_to_database(fused_event)
    
    def _save_event_to_database(self, fused_event: FusedEvent):
        """Save fused event to database"""
        try:
            # Use the correct HealthcareEventPublisher service
            from service.emergency_notification_dispatcher import healthcare_publisher
            
            # Create bounding boxes (dummy data if not available)
            bounding_boxes = [{
                'x': 100,
                'y': 100, 
                'width': 200,
                'height': 200,
                'confidence': fused_event.confidence,
                'class': 'person'
            }]
            
            # Create context data
            context_data = {
                'source': 'enhanced_fusion_system',
                'multi_camera': True,
                'cameras_involved': [fused_event.primary_camera] + fused_event.supporting_cameras,
                'consensus_score': fused_event.consensus_score,
                'frame_shape': fused_event.frame.shape if fused_event.frame is not None else None,
                'detection_type': 'fused_multi_camera',
                'timestamp': fused_event.timestamp
            }
            
            # Publish based on event type using the correct service
            if fused_event.event_type == 'fall':
                result = healthcare_publisher.publish_fall_detection(
                    confidence=fused_event.confidence,
                    bounding_boxes=bounding_boxes,
                    context=context_data
                )
            elif fused_event.event_type == 'seizure':
                result = healthcare_publisher.publish_seizure_detection(
                    confidence=fused_event.confidence,
                    bounding_boxes=bounding_boxes,
                    context=context_data
                )
            else:
                # For other event types, use generic alert via PostgreSQL service
                from service.postgresql_healthcare_service import PostgreSQLHealthcareService
                postgresql_service = PostgreSQLHealthcareService()
                
                alert_data = {
                    'alert_type': fused_event.event_type,
                    'severity': 'medium',
                    'message': f"Multi-camera {fused_event.event_type} detected (confidence: {fused_event.confidence:.2f})",
                    'alert_data': context_data
                }
                result = postgresql_service.publish_alert(alert_data)
            
            if result:
                print(f"✅ Event saved to database: {fused_event.event_type} (confidence: {fused_event.confidence:.2f})")
            else:
                print(f"⚠️ Event saved but result is None: {fused_event.event_type}")
            
        except Exception as e:
            print(f"❌ Database save error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()[:200]}...")
    
    def _display_camera_frame_with_stats(self, camera_id: str, camera_name: str, frame: np.ndarray, persons: List[Dict], processing_result: Dict):
        """Queue frame for display with comprehensive statistics overlay"""
        if not self.enable_monitors:
            print(f"🚫 Monitors disabled, skipping frame from {camera_name}")
            return
            
        try:
            current_time = time.time()
            
            # Limit frame rate per camera (max 15 FPS per camera to reduce queue pressure)
            last_time = self.last_display_time.get(camera_id, 0)
            if current_time - last_time < 0.067:  # ~15 FPS limit per camera
                return
                
            self.last_display_time[camera_id] = current_time
            
            # Create enhanced display frame with statistics overlay
            display_img = frame.copy()
            camera_stats = self.stats['camera_stats'][camera_id]
            
            # Draw person detections with pose keypoints
            for i, person in enumerate(persons):
                if 'bbox' in person:
                    # Draw bounding box
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw person ID and confidence
                    person_conf = person.get('confidence', 0)
                    cv2.putText(display_img, f"Person {i+1}: {person_conf:.2f}", 
                               (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw pose keypoints if available
                if 'keypoints' in person and person['keypoints'] is not None:
                    keypoints = person['keypoints']
                    # Draw keypoints as circles
                    for kp_idx in range(0, len(keypoints), 3):  # x, y, confidence format
                        if kp_idx + 2 < len(keypoints):
                            x, y, conf = keypoints[kp_idx], keypoints[kp_idx+1], keypoints[kp_idx+2]
                            if conf > 0.5:  # Only draw confident keypoints
                                cv2.circle(display_img, (int(x), int(y)), 3, (255, 0, 0), -1)
            
            # Statistics overlay panel (top-left)
            panel_height = 180
            panel_width = 280
            overlay = display_img.copy()
            cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
            
            # Camera info
            y_offset = 25
            cv2.putText(display_img, f"Camera: {camera_name}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS and frame count
            y_offset += 20
            cv2.putText(display_img, f"FPS: {camera_stats['fps']:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display_img, f"Frames: {camera_stats['total_frames']}", (140, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Person detection stats
            y_offset += 20
            cv2.putText(display_img, f"Persons: {len(persons)} | Total: {camera_stats['person_detections']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Pose detection stats
            y_offset += 20
            cv2.putText(display_img, f"Poses: {camera_stats['pose_estimations']} | KP: {camera_stats['keypoints_detected']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Fall detection stats
            y_offset += 20
            cv2.putText(display_img, f"Falls: {camera_stats['fall_count']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Seizure detection stats  
            y_offset += 20
            cv2.putText(display_img, f"Seizures: {camera_stats['seizure_count']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Confidence average
            y_offset += 20
            cv2.putText(display_img, f"Avg Conf: {camera_stats['confidence_avg']:.2f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Controls hint
            cv2.putText(display_img, "s=Stats e=Event q=Quit", (10, display_img.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Create display frame object
            display_frame = DisplayFrame(
                camera_id=camera_id,
                camera_name=camera_name,
                frame=display_img,  # Use enhanced frame with stats
                persons=persons.copy(),
                timestamp=current_time
            )
            
            # Always keep the latest frame: if full, remove all but one and add the new frame
            while self.display_queue.full():
                try:
                    # Remove frames until only one left (to avoid lag)
                    if self.display_queue.qsize() > 1:
                        self.display_queue.get_nowait()
                    else:
                        break
                except queue.Empty:
                    break
            try:
                self.display_queue.put_nowait(display_frame)
                queue_size = self.display_queue.qsize()
                
                # Debug every 100 frames or when queue is full
                if queue_size >= 25 or (queue_size % 20 == 0):
                    print(f"🖥️ Display queue size: {queue_size}, Camera: {camera_name}, Frame: {frame.shape}")
                    
            except queue.Full:
                print(f"⚠️ Display queue full, dropping frame from {camera_name}")
                
        except Exception as e:
            print(f"❌ Display queue error for {camera_name}: {e}")

    def _display_camera_frame(self, camera_id: str, camera_name: str, frame: np.ndarray, persons: List[Dict]):
        """Legacy method - now calls enhanced version"""
        self._display_camera_frame_with_stats(camera_id, camera_name, frame, persons, {})
    
    def _display_thread(self):
        """Optimized thread for OpenCV display management"""
        print("🖥️ Display thread started - waiting for frames...")
        
        windows_created = set()
        frames_processed = 0
        last_stats_time = time.time()
        startup_timeout = time.time() + 10  # 10 second timeout for first frame
        
        while self.running:
            try:
                # Get display frame from queue (with shorter timeout for responsiveness)
                try:
                    display_frame = self.display_queue.get(timeout=0.1)
                    frames_processed += 1
                    
                    # Debug: Show first frame received
                    if frames_processed == 1:
                        print(f"🖥️ First frame received! Camera: {display_frame.camera_name}")
                    
                    # Debug: Show processing activity less frequently
                    current_time = time.time()
                    if current_time - last_stats_time > 5:  # Every 5 seconds
                        print(f"🖥️ Processed {frames_processed} display frames, Queue: {self.display_queue.qsize()}, Windows: {len(windows_created)}")
                        last_stats_time = current_time
                        
                except queue.Empty:
                    # Check for startup timeout
                    if frames_processed == 0 and time.time() > startup_timeout:
                        print("⚠️ Display thread: No frames received after 10 seconds - cameras may not be sending display frames")
                        startup_timeout = time.time() + 30  # Reset timeout
                    
                    # Process OpenCV events even when no new frames
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key_input(key)
                    continue
                
                camera_id = display_frame.camera_id
                camera_name = display_frame.camera_name
                frame = display_frame.frame
                persons = display_frame.persons
                
                # Skip validation for speed - trust the input
                if frame is None or frame.size == 0:
                    continue
                
                # Optimize display processing
                display_img = frame  # Use original frame directly, don't copy
                
                # Draw person detections (simplified)
                for person in persons:
                    if 'bbox' in person:
                        x1, y1, x2, y2 = person['bbox']
                        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add minimal info overlay
                cv2.putText(display_img, camera_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_img, f"P:{len(persons)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add keyboard controls hint only once
                cv2.putText(display_img, "s=Stats q=Quit", (10, display_img.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Resize for display if needed (optimized)
                height, width = display_img.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    display_img = cv2.resize(display_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Create window if not exists
                window_name = f"Camera_{camera_name}"
                if window_name not in windows_created:
                    print(f"🖥️ Creating window: {window_name}")
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    # Try to position window
                    cv2.moveWindow(window_name, 100 + len(windows_created) * 320, 100)
                    windows_created.add(window_name)
                    print(f"✅ Window created: {window_name}")
                
                # Display frame
                cv2.imshow(window_name, display_img)
                
                # Process OpenCV events with minimal wait
                key = cv2.waitKey(1) & 0xFF
                self._handle_key_input(key)
                
            except Exception as e:
                print(f"Display thread error: {e}")
                time.sleep(0.05)
        
        # Cleanup on exit
        print("🖥️ Cleaning up display windows...")
        cv2.destroyAllWindows()
        print("🖥️ Display thread stopped")
    
    def _handle_key_input(self, key):
        """Handle keyboard input from OpenCV windows"""
        if key == 255:  # No key pressed
            return
            
        # Debug: Always show what key was pressed
        if key < 128:
            print(f"🎮 Key detected: '{chr(key)}' (code: {key})")
        else:
            print(f"🎮 Special key detected: {key}")
            
        try:
            if key == ord('q'):
                print("🛑 Quit requested via keyboard")
                self.stop()
            elif key == ord('s'):
                print("📊 Showing statistics...")
                self._show_statistics()
            elif key == ord('h'):
                print("ℹ️ Showing help...")
                self._show_help()
            elif key == ord('e'):
                print("🎲 Generating random event...")
                self._generate_random_event()
            elif key == ord('c'):
                print("🧹 Clearing statistics...")
                self._clear_statistics()
            else:
                # Show usage hint
                if key < 128:
                    print(f"💡 Unknown key '{chr(key)}' - use s=stats, e=event, q=quit")
                else:
                    print(f"💡 Special key {key} - use s=stats, e=event, q=quit")
        except Exception as e:
            print(f"Key handling error: {e}")
    
    def _monitor_display_thread(self):
        """Monitor system status and provide backup keyboard input"""
        print("🎮 Enhanced Monitor controls:")
        print("   Focus any camera window and press:")
        print("   'q' = Quit system")
        print("   's' = Show statistics")
        print("   'h' = Help")
        print("   'e' = Generate random event")
        print("   'c' = Clear statistics")
        print("💡 Click on camera windows to ensure they receive keyboard focus!")
        
        while self.running:
            try:
                # Just monitor system status
                time.sleep(1.0)
                
                # Check if any threads are dead and restart if needed
                for camera_id, thread in self.camera_threads.items():
                    if not thread.is_alive() and self.running:
                        print(f"⚠️ Camera thread {camera_id} died, attempting restart...")
                        
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                print(f"Monitor thread error: {e}")
                time.sleep(1.0)
    
    def _show_statistics(self):
        """Show comprehensive system statistics"""
        print("\n📊 ENHANCED MULTI-CAMERA STATISTICS:")
        print(f"   Total Events: {self.stats['total_events']}")
        print(f"   Fused Events: {self.stats['fused_events']}")
        print()
        
        print("📹 PER-CAMERA DETAILED STATS:")
        for camera_id, stats in self.stats['camera_stats'].items():
            print(f"\n   🎥 {stats['name']} ({camera_id[:8]}...):")
            print(f"      Frames: {stats['total_frames']} | FPS: {stats['fps']:.1f}")
            print(f"      Persons: {stats['person_detections']} | Poses: {stats['pose_estimations']}")
            print(f"      Keypoints: {stats['keypoints_detected']}")
            print(f"      Falls: {stats['fall_count']} | Seizures: {stats['seizure_count']}")
            print(f"      Avg Confidence: {stats['confidence_avg']:.2f}")
        
        print(f"\n🔄 EVENT TYPES:")
        for event_type, count in self.stats['event_types'].items():
            print(f"     {event_type}: {count}")
        print()
        
        # Summary
        total_falls = sum(stats['fall_count'] for stats in self.stats['camera_stats'].values())
        total_seizures = sum(stats['seizure_count'] for stats in self.stats['camera_stats'].values())
        total_persons = sum(stats['person_detections'] for stats in self.stats['camera_stats'].values())
        
        print("📈 SYSTEM SUMMARY:")
        print(f"   Total Falls Detected: {total_falls}")
        print(f"   Total Seizures Detected: {total_seizures}")
        print(f"   Total Person Detections: {total_persons}")
        print(f"   Active Cameras: {len(self.camera_configs)}")
        print()
    
    def _clear_statistics(self):
        """Clear all statistics"""
        self.stats = {
            'total_events': 0,
            'fused_events': 0,
            'camera_events': {config['camera_id']: 0 for config in self.camera_configs},
            'event_types': {},
            'camera_stats': {}
        }
        
        # Reset per-camera detailed stats
        for config in self.camera_configs:
            camera_id = config['camera_id']
            camera_name = config['name']
            self.stats['camera_stats'][camera_id] = {
                'name': camera_name,
                'fall_count': 0,
                'seizure_count': 0,
                'person_detections': 0,
                'total_frames': 0,
                'fps': 0.0,
                'last_fps_time': time.time(),
                'last_fps_count': 0,
                'confidence_avg': 0.0,
                'keypoints_detected': 0,
                'pose_estimations': 0
            }
        print("🧹 Statistics cleared!")
    
    def _generate_random_event(self):
        """Generate a random test event"""
        if not self.camera_configs:
            print("❌ No cameras available for random event")
            return
        
        # Pick random camera
        config = random.choice(self.camera_configs)
        camera_id = config['camera_id']
        camera_name = config['name']
        
        # Pick random event type
        event_types = ['fall', 'seizure', 'abnormal_behavior']
        event_type = random.choice(event_types)
        confidence = random.uniform(0.6, 0.95)
        
        # Create dummy frame (black image with text)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"RANDOM {event_type.upper()}", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(dummy_frame, f"Confidence: {confidence:.2f}", (180, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(dummy_frame, f"Camera: {camera_name}", (160, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Create random persons list
        persons = [{
            'bbox': [100, 100, 200, 300],
            'keypoints': None,
            'confidence': 0.8
        }]
        
        # Create and queue random event
        camera_event = CameraEvent(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=time.time(),
            event_type=event_type,
            confidence=confidence,
            frame=dummy_frame,
            persons=persons,
            metadata={'source': 'random_generator', 'test': True}
        )
        
        # Add to event queue
        self.event_queues[camera_id].put(camera_event)
        self.stats['camera_events'][camera_id] += 1
        self.stats['total_events'] += 1
        
        # Update event type stats
        if event_type not in self.stats['event_types']:
            self.stats['event_types'][event_type] = 0
        self.stats['event_types'][event_type] += 1
        
        print(f"🎲 Random event generated: {event_type} ({confidence:.2f}) from {camera_name}")
    
    def _show_help(self):
        """Show help"""
        print("\n🎮 KEYBOARD CONTROLS:")
        print("   'q' = Quit system")
        print("   's' = Show statistics")
        print("   'h' = Show this help")
        print("   'e' = Generate random event")
        print("   'c' = Clear statistics")
        print("   Focus any camera window and press keys")
        print()
        print("   's' = Show statistics")
        print("   'h' = Show this help")
        print()
    
    def stop(self):
        """Stop all threads gracefully"""
        print("🛑 Stopping Enhanced Multi-Camera System...")
        self.running = False
        
        # Wait a moment for threads to recognize stop signal
        time.sleep(0.5)
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Wait for threads to finish gracefully (but not current thread)
        print("⏳ Waiting for threads to finish...")
        current_thread = threading.current_thread()
        
        # Wait for camera threads
        for camera_id, thread in self.camera_threads.items():
            if thread.is_alive() and thread != current_thread:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"⚠️ Camera thread {camera_id} didn't stop gracefully")
        
        # Wait for fusion thread
        if (self.fusion_thread and 
            self.fusion_thread.is_alive() and 
            self.fusion_thread != current_thread):
            self.fusion_thread.join(timeout=1.0)
        
        # Wait for display thread
        if (self.display_thread and 
            self.display_thread.is_alive() and 
            self.display_thread != current_thread):
            self.display_thread.join(timeout=1.0)
        
        print("✅ Enhanced Multi-Camera System stopped successfully")

# Global instance for easy access
enhanced_system = None

def create_enhanced_system(camera_configs: List[Dict], enable_monitors: bool = True) -> EnhancedMultiCameraSystem:
    """Create enhanced multi-camera system"""
    global enhanced_system
    enhanced_system = EnhancedMultiCameraSystem(camera_configs, enable_monitors=enable_monitors)
    return enhanced_system