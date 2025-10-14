#!/usr/bin/env python3
"""
Same Room Dual Detection Main Application
Sử dụng 2 camera trong cùng 1 phòng để phát hiện sự cố y tế không có điểm mù
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from service.dual_camera_surveillance_system import SameRoomDualDetection
from config.config_loader import ConfigLoader
from camera.config import IMOUCameraConfig
from camera.simple_camera import SimpleIMOUCamera
from service.video_processing_service import VideoProcessingService
from service.fall_detection_service import FallDetectionService
from service.seizure_detection_service import SeizureDetectionService
from service.notification_service import NotificationService
from service.supabase_service import SupabaseService
from service.fcm_service import FCMService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dual_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DualDetectionApp:
    def __init__(self, config_path="src/config/config.json"):
        """Initialize dual detection application"""
        self.config_path = config_path
        self.dual_detector = None
        self.running = False
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.dual_detector:
            self.dual_detector.stop()
            
    def _setup_cameras(self):
        """Setup camera configurations"""
        cameras = {}
        camera_configs = self.config.get('cameras', {})
        
        for camera_id, camera_config in camera_configs.items():
            try:
                # Create camera config
                config = IMOUCameraConfig(
                    rtsp_url=camera_config['rtsp_url'],
                    username=camera_config['username'],
                    password=camera_config['password'],
                    width=camera_config.get('width', 1920),
                    height=camera_config.get('height', 1080),
                    fps=camera_config.get('fps', 30)
                )
                
                # Create camera instance
                camera = SimpleIMOUCamera(config)
                cameras[camera_id] = {
                    'camera': camera,
                    'position': camera_config.get('position', 'unknown'),
                    'area': camera_config.get('area', 'unknown')
                }
                
                logger.info(f"Setup camera {camera_id} at position {camera_config.get('position')}")
                
            except Exception as e:
                logger.error(f"Failed to setup camera {camera_id}: {e}")
                
        return cameras
        
    def _setup_services(self):
        """Setup detection and notification services"""
        services = {}
        
        try:
            # Video processing service
            services['video_processor'] = VideoProcessingService()
            logger.info("Video processing service initialized")
            
            # Fall detection service
            fall_config = self.config.get('fall_detection', {})
            services['fall_detector'] = FallDetectionService(
                confidence_threshold=fall_config.get('confidence_threshold', 0.4)
            )
            logger.info("Fall detection service initialized")
            
            # Seizure detection service
            seizure_config = self.config.get('seizure_detection', {})
            services['seizure_detector'] = SeizureDetectionService(
                confidence_threshold=seizure_config.get('confidence_threshold', 0.7),
                temporal_window=seizure_config.get('temporal_window', 25),
                alert_threshold=seizure_config.get('alert_threshold', 0.7),
                warning_threshold=seizure_config.get('warning_threshold', 0.5)
            )
            logger.info("Seizure detection service initialized")
            
            # Notification services
            notification_config = self.config.get('notifications', {})
            if notification_config.get('enabled', False):
                services['notification'] = NotificationService()
                logger.info("Notification service initialized")
                
            # Supabase service
            supabase_config = self.config.get('supabase', {})
            if supabase_config.get('enabled', False):
                services['supabase'] = SupabaseService()
                logger.info("Supabase service initialized")
                
            # FCM service
            fcm_config = self.config.get('fcm', {})
            if fcm_config.get('enabled', False):
                services['fcm'] = FCMService()
                logger.info("FCM service initialized")
                
        except Exception as e:
            logger.error(f"Failed to setup services: {e}")
            
        return services
        
    def _setup_dual_detector(self, cameras, services):
        """Setup dual detection system"""
        try:
            # Create dual detector
            self.dual_detector = SameRoomDualDetection(
                cameras=cameras,
                video_processor=services.get('video_processor'),
                fall_detector=services.get('fall_detector'),
                seizure_detector=services.get('seizure_detector'),
                notification_service=services.get('notification'),
                supabase_service=services.get('supabase'),
                fcm_service=services.get('fcm')
            )
            
            logger.info("Same room dual detection system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup dual detector: {e}")
            return False
            
    def run(self):
        """Run the dual detection application"""
        logger.info("Starting Same Room Dual Detection Application")
        
        try:
            # Setup cameras
            logger.info("Setting up cameras...")
            cameras = self._setup_cameras()
            if not cameras:
                logger.error("No cameras configured, exiting")
                return
                
            # Setup services
            logger.info("Setting up services...")
            services = self._setup_services()
            
            # Setup dual detector
            logger.info("Setting up dual detection system...")
            if not self._setup_dual_detector(cameras, services):
                logger.error("Failed to setup dual detector, exiting")
                return
                
            # Start detection
            logger.info("Starting dual detection...")
            self.dual_detector.start()
            self.running = True
            
            # Main loop
            logger.info("Dual detection system running. Press Ctrl+C to stop.")
            while self.running:
                # Get latest statistics
                stats = self.dual_detector.get_statistics()
                
                # Log statistics every 30 seconds
                if int(time.time()) % 30 == 0:
                    logger.info(f"Detection Stats: {json.dumps(stats, indent=2)}")
                    
                # Check for alerts
                if self.dual_detector.has_recent_detections():
                    logger.info("Recent detections found, checking for alerts...")
                    
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self._cleanup()
            
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        if self.dual_detector:
            self.dual_detector.stop()
            
        logger.info("Cleanup completed")
        
    def test_cameras(self):
        """Test camera connections"""
        logger.info("Testing camera connections...")
        
        cameras = self._setup_cameras()
        
        for camera_id, camera_info in cameras.items():
            camera = camera_info['camera']
            position = camera_info['position']
            
            try:
                logger.info(f"Testing camera {camera_id} at {position}...")
                
                if camera.connect():
                    logger.info(f"Camera {camera_id} connected successfully")
                    
                    # Test frame capture
                    frame = camera.get_frame()
                    if frame is not None:
                        logger.info(f"Camera {camera_id} frame capture OK (shape: {frame.shape})")
                    else:
                        logger.warning(f"Camera {camera_id} failed to capture frame")
                        
                    camera.disconnect()
                else:
                    logger.error(f"Camera {camera_id} failed to connect")
                    
            except Exception as e:
                logger.error(f"Camera {camera_id} test failed: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Same Room Dual Detection System')
    parser.add_argument('--config', default='src/config/config.json',
                       help='Configuration file path')
    parser.add_argument('--test-cameras', action='store_true',
                       help='Test camera connections only')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create app
    app = DualDetectionApp(config_path=args.config)
    
    if args.test_cameras:
        app.test_cameras()
    else:
        app.run()

if __name__ == "__main__":
    main()
