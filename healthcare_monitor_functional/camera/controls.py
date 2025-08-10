#!/usr/bin/env python3
"""
Camera Control Functions - Fall Detection Only
Function-based camera handling for RTSP stream
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


def initialize_camera(camera_config: Dict[str, Any]) -> Tuple[Optional[cv2.VideoCapture], Dict[str, Any]]:
    """
    Initialize RTSP camera with configuration settings
    
    Args:
        camera_config: Camera configuration dictionary
        
    Returns:
        Tuple of (camera_object, status_info)
    """
    status_info = {
        'connected': False,
        'error': None,
        'fps': 0,
        'resolution': None
    }
    
    try:
        rtsp_url = camera_config.get('rtsp_url', '')
        if not rtsp_url:
            raise ValueError("RTSP URL not provided in configuration")
        
        # Initialize RTSP capture
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
        
        # Set buffer size for low latency
        buffer_size = camera_config.get('buffer_size', 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        # Set camera properties
        width = camera_config.get('resolution', (640, 480))[0]
        height = camera_config.get('resolution', (640, 480))[1]
        fps = camera_config.get('fps', 30)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        status_info.update({
            'connected': True,
            'fps': actual_fps,
            'resolution': (actual_width, actual_height)
        })
        
        print(f"IMOU Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return cap, status_info
        
    except Exception as e:
        error_msg = f"Camera initialization failed: {str(e)}"
        print(error_msg)
        status_info['error'] = error_msg
        return None, status_info
