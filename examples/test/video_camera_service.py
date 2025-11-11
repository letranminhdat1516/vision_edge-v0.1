"""
Video Camera Service - Simulates camera using video files for testing
Replaces CameraService to read from video files instead of RTSP streams
"""

import cv2
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VideoCameraService:
    """Simulates camera feed using video file"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize video camera service
        
        Args:
            config: Camera configuration with 'video_path' instead of 'url'
                - video_path: Path to video file
                - camera_id: Unique camera ID
                - camera_name: Display name
                - fps: Target FPS
                - resolution: Target resolution (width, height)
                - loop: Loop video when finished (default: False)
        """
        self.video_path = config.get('video_path') or config.get('url')
        self.camera_id = config.get('camera_id', 'test_camera_001')
        self.camera_name = config.get('camera_name', 'Test Video Camera')
        self.fps = config.get('fps', 30)
        self.resolution = config.get('resolution', (1920, 1080))
        self.loop = config.get('loop', False)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_count = 0
        self.total_frames = 0
        self.video_fps = 0
        
        logger.info(f"📹 Initialized VideoCameraService for {self.camera_name}")
        logger.info(f"   Video path: {self.video_path}")
    
    def connect(self) -> bool:
        """Connect to video file"""
        try:
            if not Path(self.video_path).exists():
                logger.error(f"❌ Video file not found: {self.video_path}")
                return False
            
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open video: {self.video_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.is_connected = True
            logger.info(f"✅ Video loaded successfully!")
            logger.info(f"   Total frames: {self.total_frames}")
            logger.info(f"   Video FPS: {self.video_fps:.2f}")
            logger.info(f"   Resolution: {video_width}x{video_height}")
            logger.info(f"   Duration: {self.total_frames / self.video_fps:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error connecting to video: {e}")
            return False
    
    def get_frame(self) -> Optional[Any]:
        """Get next frame from video"""
        if not self.is_connected or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop:
                # Restart video from beginning
                logger.info("📹 Video finished, restarting from beginning...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("❌ Failed to restart video")
                    return None
            else:
                logger.info("📹 Video finished")
                return None
        
        self.frame_count += 1
        
        # Resize if needed (only if resolution is specified)
        if frame is not None and self.resolution is not None:
            current_height, current_width = frame.shape[:2]
            if (current_width, current_height) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
        
        # Simulate camera FPS delay
        time.sleep(1.0 / self.fps)
        
        return frame
    
    def disconnect(self):
        """Disconnect from video"""
        if self.cap:
            self.cap.release()
            self.is_connected = False
            logger.info(f"✅ Video camera disconnected: {self.camera_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
        return {
            'connected': self.is_connected,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'video_path': self.video_path,
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'progress': f"{progress:.1f}%",
            'progress_percent': progress
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()
