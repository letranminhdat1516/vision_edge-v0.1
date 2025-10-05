"""
Multi-Camera Manager for Healthcare Monitoring
Manages multiple cameras and selects best frame for detection
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from service.camera_service import CameraService

@dataclass
class CameraFrame:
    """Container for camera frame with metadata"""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    quality_score: float = 0.0
    motion_level: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0

@dataclass
class CameraConfig:
    """Configuration for individual camera"""
    camera_id: str
    name: str
    rtsp_url: str
    location: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    enabled: bool = True

class FrameQualityAnalyzer:
    """Analyzes frame quality for selection"""
    
    def __init__(self):
        self.prev_frames = {}  # Store previous frames for motion calculation
    
    def analyze_frame_quality(self, camera_id: str, frame: np.ndarray) -> Dict[str, float]:
        """Analyze frame quality metrics"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Brightness analysis
            brightness = gray.mean()  # Use numpy array method
            brightness_score = self._score_brightness(brightness)
            
            # 2. Sharpness analysis (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = self._score_sharpness(sharpness)
            
            # 3. Motion analysis
            motion_score = self._analyze_motion(camera_id, gray)
            
            # 4. Overall quality score
            quality_score = (
                brightness_score * 0.3 +
                sharpness_score * 0.4 +
                motion_score * 0.3
            )
            
            return {
                'brightness': brightness,
                'brightness_score': brightness_score,
                'sharpness': sharpness,
                'sharpness_score': sharpness_score,
                'motion_score': motion_score,
                'quality_score': quality_score
            }
            
        except Exception as e:
            print(f"❌ Frame quality analysis error: {e}")
            return {
                'brightness': 0, 'brightness_score': 0,
                'sharpness': 0, 'sharpness_score': 0,
                'motion_score': 0, 'quality_score': 0
            }
    
    def _score_brightness(self, brightness: float) -> float:
        """Score brightness (0-1, higher is better)"""
        # Optimal brightness range: 80-180
        if 80 <= brightness <= 180:
            return 1.0
        elif brightness < 50 or brightness > 200:
            return 0.2
        else:
            return 0.6
    
    def _score_sharpness(self, sharpness: float) -> float:
        """Score sharpness (0-1, higher is better)"""
        # Higher variance = sharper image
        if sharpness > 500:
            return 1.0
        elif sharpness > 100:
            return 0.7
        elif sharpness > 50:
            return 0.4
        else:
            return 0.1
    
    def _analyze_motion(self, camera_id: str, gray_frame: np.ndarray) -> float:
        """Analyze motion level (0-1, moderate motion is better)"""
        try:
            if camera_id not in self.prev_frames:
                self.prev_frames[camera_id] = gray_frame
                return 0.5  # Default score for first frame
            
            # Calculate frame difference
            diff = cv2.absdiff(self.prev_frames[camera_id], gray_frame)
            motion_pixels = cv2.countNonZero(diff)
            motion_ratio = motion_pixels / (gray_frame.shape[0] * gray_frame.shape[1])
            
            # Update previous frame
            self.prev_frames[camera_id] = gray_frame.copy()
            
            # Score motion (moderate motion is preferred for detection)
            if 0.02 <= motion_ratio <= 0.15:  # Good motion range
                return 1.0
            elif 0.01 <= motion_ratio <= 0.25:  # Acceptable motion
                return 0.7
            elif motion_ratio < 0.01:  # Too static
                return 0.3
            else:  # Too much motion
                return 0.2
                
        except Exception as e:
            return 0.5

class MultiCameraManager:
    """Manages multiple cameras and selects best frame"""
    
    def __init__(self, camera_configs: List[CameraConfig]):
        self.camera_configs = camera_configs
        self.cameras: Dict[str, CameraService] = {}
        self.frame_analyzer = FrameQualityAnalyzer()
        
        # Threading
        self.running = False
        self.capture_threads = {}
        self.frame_buffers = {}  # Latest frame from each camera
        self.frame_locks = {}
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'best_selections': {},  # Count per camera
            'quality_scores': {},   # Latest scores per camera
            'connection_status': {} # Connection status per camera
        }
        
        print(f"🎥 Multi-Camera Manager initialized with {len(camera_configs)} cameras")
    
    def initialize_cameras(self) -> bool:
        """Initialize all cameras"""
        success_count = 0
        
        for config in self.camera_configs:
            if not config.enabled:
                continue
                
            try:
                print(f"📹 Initializing {config.name} ({config.camera_id})...")
                
                # Create camera service
                camera_service_config = {
                    'url': config.rtsp_url,
                    'buffer_size': 1,
                    'fps': 15,
                    'resolution': (640, 480),
                    'auto_reconnect': True
                }
                
                camera = CameraService(camera_service_config)
                
                if camera.connect():
                    self.cameras[config.camera_id] = camera
                    self.frame_buffers[config.camera_id] = None
                    self.frame_locks[config.camera_id] = threading.Lock()
                    self.stats['best_selections'][config.camera_id] = 0
                    self.stats['quality_scores'][config.camera_id] = 0.0
                    self.stats['connection_status'][config.camera_id] = True
                    
                    print(f"✅ {config.name} connected successfully")
                    success_count += 1
                else:
                    print(f"❌ Failed to connect {config.name}")
                    self.stats['connection_status'][config.camera_id] = False
                    
            except Exception as e:
                print(f"❌ Error initializing {config.name}: {e}")
                self.stats['connection_status'][config.camera_id] = False
        
        print(f"📊 Cameras initialized: {success_count}/{len([c for c in self.camera_configs if c.enabled])}")
        return success_count > 0
    
    def start_capture(self):
        """Start capturing from all cameras"""
        if self.running:
            return
            
        self.running = True
        
        for camera_id, camera in self.cameras.items():
            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera_id, camera),
                daemon=True
            )
            thread.start()
            self.capture_threads[camera_id] = thread
            
        print("🎬 Multi-camera capture started")
    
    def _capture_loop(self, camera_id: str, camera: CameraService):
        """Capture loop for individual camera"""
        while self.running:
            try:
                frame = camera.get_frame()
                if frame is not None:
                    # Analyze frame quality
                    quality_metrics = self.frame_analyzer.analyze_frame_quality(camera_id, frame)
                    
                    # Create camera frame object
                    camera_frame = CameraFrame(
                        camera_id=camera_id,
                        frame=frame,
                        timestamp=time.time(),
                        quality_score=quality_metrics['quality_score'],
                        motion_level=quality_metrics['motion_score'],
                        brightness=quality_metrics['brightness'],
                        sharpness=quality_metrics['sharpness']
                    )
                    
                    # Update frame buffer
                    with self.frame_locks[camera_id]:
                        self.frame_buffers[camera_id] = camera_frame
                        self.stats['quality_scores'][camera_id] = quality_metrics['quality_score']
                        self.stats['connection_status'][camera_id] = True
                else:
                    self.stats['connection_status'][camera_id] = False
                    time.sleep(0.1)  # Wait before retry
                    
            except Exception as e:
                print(f"❌ Capture error for {camera_id}: {e}")
                self.stats['connection_status'][camera_id] = False
                time.sleep(1)
    
    def get_best_frame(self) -> Optional[CameraFrame]:
        """Select and return the best frame from all cameras"""
        try:
            available_frames = []
            
            # Collect available frames
            for camera_id in self.cameras.keys():
                with self.frame_locks[camera_id]:
                    if self.frame_buffers[camera_id] is not None:
                        # Check if frame is recent (within 1 second)
                        if time.time() - self.frame_buffers[camera_id].timestamp < 1.0:
                            available_frames.append(self.frame_buffers[camera_id])
            
            if not available_frames:
                return None
            
            # Select best frame based on multiple criteria
            best_frame = self._select_best_frame(available_frames)
            
            if best_frame:
                self.stats['best_selections'][best_frame.camera_id] += 1
                self.stats['total_frames'] += 1
            
            return best_frame
            
        except Exception as e:
            print(f"❌ Error getting best frame: {e}")
            return None
    
    def _select_best_frame(self, frames: List[CameraFrame]) -> Optional[CameraFrame]:
        """Select the best frame from available frames"""
        if not frames:
            return None
        
        # Sort by composite score
        def calculate_composite_score(frame: CameraFrame) -> float:
            # Get camera priority
            camera_priority = 1.0
            for config in self.camera_configs:
                if config.camera_id == frame.camera_id:
                    camera_priority = 1.0 / config.priority  # Higher priority = lower number
                    break
            
            # Composite score: quality + priority + recency
            recency_score = max(0, 1.0 - (time.time() - frame.timestamp))
            
            composite = (
                frame.quality_score * 0.6 +
                camera_priority * 0.3 +
                recency_score * 0.1
            )
            
            return composite
        
        # Select frame with highest composite score
        best_frame = max(frames, key=calculate_composite_score)
        return best_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-camera statistics"""
        return {
            'total_frames_processed': self.stats['total_frames'],
            'cameras_connected': sum(1 for status in self.stats['connection_status'].values() if status),
            'total_cameras': len(self.cameras),
            'best_selections_per_camera': dict(self.stats['best_selections']),
            'quality_scores_per_camera': dict(self.stats['quality_scores']),
            'connection_status': dict(self.stats['connection_status'])
        }
    
    def stop_capture(self):
        """Stop capturing from all cameras"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.capture_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)
        
        print("🛑 Multi-camera capture stopped")
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        self.stop_capture()
        
        for camera in self.cameras.values():
            camera.disconnect()
        
        print("🔌 All cameras disconnected")

# Helper function to create camera configs
def create_camera_configs() -> List[CameraConfig]:
    """Create camera configurations from config file"""
    # Load from config.json cameras section
    from service.config_loader import config_loader
    
    config = config_loader.load_system_config()
    cameras_config = config.get('database', {}).get('cameras', {})
    
    camera_configs = []
    for camera_key, camera_data in cameras_config.items():
        config = CameraConfig(
            camera_id=camera_data['camera_id'],
            name=camera_data['name'],
            rtsp_url=camera_data['rtsp_url'],
            location=camera_data['location'],
            priority=1 if 'living' in camera_data['location'].lower() else 2,  # Living room has priority
            enabled=True
        )
        camera_configs.append(config)
    
    return camera_configs

# Global instance (will be initialized when needed)
multi_camera_manager = None

def get_multi_camera_manager() -> MultiCameraManager:
    """Get or create multi-camera manager instance"""
    global multi_camera_manager
    
    if multi_camera_manager is None:
        camera_configs = create_camera_configs()
        multi_camera_manager = MultiCameraManager(camera_configs)
        
        if multi_camera_manager.initialize_cameras():
            multi_camera_manager.start_capture()
        else:
            print("❌ Failed to initialize any cameras")
            
    return multi_camera_manager
