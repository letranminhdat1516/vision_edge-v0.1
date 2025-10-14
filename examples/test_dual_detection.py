#!/usr/bin/env python3
"""
Test script cho Same Room Dual Detection System
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dual_detection():
    """Test dual detection system"""
    logger.info("Starting dual detection test...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent.parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Import modules
        from service.dual_camera_surveillance_system import SameRoomDualDetection, DetectionResult
        from camera.config import IMOUCameraConfig
        from camera.simple_camera import SimpleIMOUCamera
        
        logger.info("Modules imported successfully")
        
        # Test camera configuration
        camera_config = IMOUCameraConfig(
            rtsp_url="rtsp://test:test@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
            username="test",
            password="test",
            ip_address="192.168.1.100",
            port=554,
            frame_width=1920,
            frame_height=1080,
            fps=30
        )
        
        logger.info("Camera config created")
        
        # Test DetectionResult
        detection_result = DetectionResult(
            camera_id="test_camera",
            frame=None,
            timestamp=time.time(),
            persons_detected=[],
            fall_confidence=0.0,
            seizure_confidence=0.0,
            motion_level=0,
            coverage_area="test"
        )
        
        logger.info(f"Detection result created: {detection_result}")
        
        # Test dual detection initialization
        camera_configs = [
            {
                "camera_id": "camera_01",
                "name": "Left Camera",
                "position": "left",
                "area": "Living Room",
                "rtsp_url": "rtsp://test:test@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
            },
            {
                "camera_id": "camera_02", 
                "name": "Right Camera",
                "position": "right",
                "area": "Living Room",
                "rtsp_url": "rtsp://test:test@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0"
            }
        ]
        
        dual_detector = SameRoomDualDetection(camera_configs=camera_configs)
        logger.info("Dual detector created successfully")
        
        # Test statistics
        stats = dual_detector.get_statistics()
        logger.info(f"Initial statistics: {json.dumps(stats, indent=2)}")
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        config_path = Path(__file__).parent.parent / "src" / "config" / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            logger.info("Configuration loaded successfully")
            
            # Check cameras config
            cameras = config.get('cameras', {})
            logger.info(f"Found {len(cameras)} cameras configured:")
            
            for camera_id, camera_config in cameras.items():
                logger.info(f"  {camera_id}: {camera_config.get('position')} at {camera_config.get('area')}")
                
        else:
            logger.warning(f"Config file not found: {config_path}")
            
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")

def main():
    """Main test runner"""
    logger.info("=== Same Room Dual Detection Test Suite ===")
    
    # Test configuration
    test_configuration()
    
    # Test dual detection
    test_dual_detection()
    
    logger.info("=== Test Suite Completed ===")

if __name__ == "__main__":
    main()
