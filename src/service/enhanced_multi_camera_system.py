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
        
        # Display queue for frames
        self.display_queue = queue.Queue(maxsize=10)
        
        # Event fusion
        self.fusion_engine = EventFusionEngine()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'fused_events': 0,
            'camera_events': {},
            'event_types': {}
        }
        
        # Initialize per-camera stats
        for config in camera_configs:
            camera_id = config['camera_id']
            self.event_queues[camera_id] = queue.Queue()
            self.stats['camera_events'][camera_id] = 0
        
        print(f"🎥 Enhanced Multi-Camera System initialized for {len(camera_configs)} cameras")
        if enable_monitors:
            print("🖥️ Display monitors enabled")
        else:
            print("🚫 Display monitors disabled")
    
    def start(self):
        """Start all camera threads and fusion engine"""
        self.running = True
        
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
        
        return True
    
    def _camera_thread(self, camera_id: str, config: Dict):
        """Individual camera processing thread"""
        camera_name = config['name']
        print(f"📹 [{camera_name}] Thread started")
        
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
        
        camera = CameraService(camera_config)
        
        if not camera.connect():
            print(f"❌ [{camera_name}] Failed to connect")
            return
        
        print(f"✅ [{camera_name}] Connected and streaming")
        
        # Initialize detection services
        from service.video_processing_service import VideoProcessingService
        from service.fall_detection_service import FallDetectionService
        from service.seizure_detection_service import SeizureDetectionService
        
        video_processor = VideoProcessingService(120)
        fall_detector = FallDetectionService()
        seizure_detector = SeizureDetectionService()
        
        frame_count = 0
        last_event_time = 0
        
        while self.running:
            try:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Display frame ALWAYS (not just when processing)
                self._display_camera_frame(camera_id, camera_name, frame, [])
                
                # Process every 5th frame for efficiency
                if frame_count % 5 != 0:
                    continue
                
                # Process frame
                result = video_processor.process_frame(frame)
                if not result.get('processed', False):
                    continue
                
                persons = result.get('person_detections', [])
                
                # Update display with person detections if any
                if persons:
                    self._display_camera_frame(camera_id, camera_name, frame, persons)
                
                # Detect events
                events_detected = []
                
                # Fall detection
                try:
                    fall_result = fall_detector.detect_fall(frame, persons)
                    if fall_result.get('fall_detected', False):
                        confidence = fall_result.get('confidence', 0)
                        if confidence > 0.3:  # Threshold
                            events_detected.append({
                                'type': 'fall',
                                'confidence': confidence,
                                'metadata': fall_result
                            })
                except Exception as e:
                    pass  # Skip errors
                
                # Seizure detection
                try:
                    seizure_result = seizure_detector.detect_seizure(frame, persons)
                    if seizure_result.get('seizure_detected', False):
                        confidence = seizure_result.get('confidence', 0)
                        if confidence > 0.25:  # Lower threshold
                            events_detected.append({
                                'type': 'seizure',
                                'confidence': confidence,
                                'metadata': seizure_result
                            })
                except Exception as e:
                    pass  # Skip errors
                
                # Create events and queue them
                for event_data in events_detected:
                    # Debounce - avoid too frequent events
                    if current_time - last_event_time < 1.0:
                        continue
                    
                    camera_event = CameraEvent(
                        camera_id=camera_id,
                        camera_name=camera_name,
                        timestamp=current_time,
                        event_type=event_data['type'],
                        confidence=event_data['confidence'],
                        frame=frame.copy(),
                        persons=persons,
                        metadata=event_data['metadata']
                    )
                    
                    # Queue event for fusion
                    self.event_queues[camera_id].put(camera_event)
                    self.stats['camera_events'][camera_id] += 1
                    last_event_time = current_time
                    
                    print(f"🔔 [{camera_name}] Event: {event_data['type']} ({event_data['confidence']:.2f})")
                
                # Display frame (if monitors enabled)
                self._display_camera_frame(camera_id, camera_name, frame, persons)
                
            except Exception as e:
                print(f"❌ [{camera_name}] Error: {e}")
                time.sleep(1)
        
        camera.disconnect()
        print(f"📹 [{camera_name}] Thread stopped")
    
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
            # Use PostgreSQL healthcare service
            from service.postgresql_healthcare_service import PostgreSQLHealthcareService
            healthcare_service = PostgreSQLHealthcareService()
            
            # Create event data
            event_data = {
                'event_type': fused_event.event_type,
                'confidence': fused_event.confidence,
                'cameras_involved': [fused_event.primary_camera] + fused_event.supporting_cameras,
                'consensus_score': fused_event.consensus_score,
                'frame_shape': fused_event.frame.shape if fused_event.frame is not None else None,
                'multi_camera': True,
                'source': 'enhanced_fusion_system'
            }
            
            # Publish based on event type
            if fused_event.event_type == 'fall':
                healthcare_service.publish_fall_detection(
                    confidence=fused_event.confidence,
                    bounding_boxes=[],
                    context=event_data
                )
            elif fused_event.event_type == 'seizure':
                healthcare_service.publish_seizure_detection(
                    confidence=fused_event.confidence,
                    context=event_data
                )
            else:
                # For other event types, use generic alert
                healthcare_service.publish_alert(
                    alert_type=fused_event.event_type,
                    severity='medium',
                    message=f"Multi-camera {fused_event.event_type} detected",
                    metadata=event_data
                )
            
            print(f"✅ Event saved to database: {fused_event.event_type} (confidence: {fused_event.confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Database save error: {e}")
    
    def _display_camera_frame(self, camera_id: str, camera_name: str, frame: np.ndarray, persons: List[Dict]):
        """Queue frame for display (thread-safe)"""
        if not self.enable_monitors:
            return
            
        try:
            display_frame = DisplayFrame(
                camera_id=camera_id,
                camera_name=camera_name,
                frame=frame.copy(),
                persons=persons.copy(),
                timestamp=time.time()
            )
            
            # Add to display queue (non-blocking)
            try:
                self.display_queue.put_nowait(display_frame)
            except queue.Full:
                # Skip if queue is full (drop frames to avoid lag)
                pass
                
        except Exception as e:
            print(f"Display queue error for {camera_name}: {e}")
    
    def _display_thread(self):
        """Dedicated thread for OpenCV display management"""
        print("🖥️ Display thread started")
        
        windows_created = set()
        
        while self.running:
            try:
                # Get display frame from queue (with timeout)
                try:
                    display_frame = self.display_queue.get(timeout=0.1)
                except queue.Empty:
                    # Process OpenCV events even when no new frames
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key_input(key)
                    continue
                
                camera_id = display_frame.camera_id
                camera_name = display_frame.camera_name
                frame = display_frame.frame
                persons = display_frame.persons
                
                # Process frame for display
                display_img = frame.copy()
                
                # Draw person detections
                for person in persons:
                    if 'bbox' in person:
                        x1, y1, x2, y2 = person['bbox']
                        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add camera info and timestamp
                cv2.putText(display_img, f"{camera_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_img, f"Persons: {len(persons)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_img, f"Time: {time.strftime('%H:%M:%S')}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add keyboard controls hint
                cv2.putText(display_img, "Press: s=Stats, e=Event, q=Quit", (10, display_img.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Resize for display if needed
                height, width = display_img.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    display_img = cv2.resize(display_img, (new_width, new_height))
                
                # Create window if not exists
                window_name = f"Camera_{camera_name}"
                if window_name not in windows_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    windows_created.add(window_name)
                
                # Display frame
                cv2.imshow(window_name, display_img)
                
                # Process OpenCV events and handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_key_input(key)
                
            except Exception as e:
                print(f"Display thread error: {e}")
                time.sleep(0.1)
        
        # Cleanup on exit
        print("🖥️ Cleaning up display windows...")
        cv2.destroyAllWindows()
        print("🖥️ Display thread stopped")
    
    def _handle_key_input(self, key):
        """Handle keyboard input from OpenCV windows"""
        if key == 255:  # No key pressed
            return
            
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
                # Show what key was pressed for debugging
                if key < 128:
                    print(f"🎮 Key pressed: '{chr(key)}' (use s=stats, e=event, q=quit)")
                else:
                    print(f"🎮 Special key pressed: {key}")
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
        """Show system statistics"""
        print("\n📊 SYSTEM STATISTICS:")
        print(f"   Total Events: {self.stats['total_events']}")
        print(f"   Fused Events: {self.stats['fused_events']}")
        print("   Camera Events:")
        for camera_id, count in self.stats['camera_events'].items():
            # Get camera name from config
            camera_name = "Unknown"
            for config in self.camera_configs:
                if config['camera_id'] == camera_id:
                    camera_name = config['name']
                    break
            print(f"     {camera_name}: {count}")
        print("   Event Types:")
        for event_type, count in self.stats['event_types'].items():
            print(f"     {event_type}: {count}")
        print()
    
    def _clear_statistics(self):
        """Clear all statistics"""
        self.stats = {
            'total_events': 0,
            'fused_events': 0,
            'camera_events': {config['camera_id']: 0 for config in self.camera_configs},
            'event_types': {}
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