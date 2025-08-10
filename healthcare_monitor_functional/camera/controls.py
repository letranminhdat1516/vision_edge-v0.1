#!/usr/bin/env python3
"""
Camera Control Functions  
Function-based camera handling system
"""

import sys
import cv2
import time
import logging
import threading
import numpy as np
from queue import Queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Add src to path for imports  
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def initialize_camera(camera_config: Dict[str, Any]) -> Tuple[Optional[cv2.VideoCapture], Dict[str, Any]]:
    """
    Initialize camera with configuration settings
    
    Args:
        camera_config: Camera configuration dictionary
        
    Returns:
        Tuple of (camera_object, status_info)
    """
    status_info = {
        'connected': False,
        'error': None,
        'fps': 0,
        'resolution': None,
        'backend': None
    }
    
    try:
        # Determine camera source
        if camera_config.get('type') == 'rtsp':
            rtsp_url = camera_config.get('rtsp_url', '')
            if not rtsp_url:
                raise ValueError("RTSP URL not provided in configuration")
            camera_source = rtsp_url
            
        elif camera_config.get('type') == 'webcam':
            camera_source = camera_config.get('device_index', 0)
            
        else:
            # Try RTSP first, then fallback to webcam
            rtsp_url = camera_config.get('rtsp_url', '')
            if rtsp_url:
                camera_source = rtsp_url
            else:
                camera_source = 0  # Default webcam
        
        # Initialize capture
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {camera_source}")
        
        # Set camera properties
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        fps = camera_config.get('fps', 15)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.get(cv2.CAP_PROP_BACKEND)
        
        status_info.update({
            'connected': True,
            'fps': actual_fps,
            'resolution': (actual_width, actual_height),
            'backend': backend
        })
        
        logging.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return cap, status_info
        
    except Exception as e:
        error_msg = f"Camera initialization failed: {str(e)}"
        logging.error(error_msg)
        status_info['error'] = error_msg
        return None, status_info


def capture_frame(cap: cv2.VideoCapture, retry_attempts: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Capture a single frame from camera
    
    Args:
        cap: OpenCV VideoCapture object
        retry_attempts: Number of retry attempts on failure
        
    Returns:
        Tuple of (success, frame)
    """
    if cap is None:
        return False, None
    
    for attempt in range(retry_attempts):
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                return True, frame
            else:
                if attempt < retry_attempts - 1:
                    time.sleep(0.05)  # Brief pause before retry
                    
        except Exception as e:
            logging.warning(f"Frame capture attempt {attempt + 1} failed: {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(0.1)
    
    return False, None


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate frame quality and properties
    
    Args:
        frame: Input frame
        
    Returns:
        True if frame is valid, False otherwise
    """
    if frame is None:
        return False
    
    if not isinstance(frame, np.ndarray):
        return False
    
    # Check dimensions
    if len(frame.shape) != 3:
        return False
    
    height, width, channels = frame.shape
    if height < 100 or width < 100 or channels != 3:
        return False
    
    # Check if frame is not completely black or white
    mean_val = np.mean(frame)
    if mean_val < 5 or mean_val > 250:
        return False
    
    return True


def preprocess_frame(frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = False) -> np.ndarray:
    """
    Preprocess frame for detection
    
    Args:
        frame: Input frame
        target_size: Target size for resizing (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed frame
    """
    processed_frame = frame.copy()
    
    # Resize if target size specified
    if target_size is not None:
        processed_frame = cv2.resize(processed_frame, target_size)
    
    # Normalize if requested
    if normalize:
        processed_frame = processed_frame.astype(np.float32) / 255.0
    
    return processed_frame


class FrameBuffer:
    """Thread-safe frame buffer for camera frames"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = Queue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def put_frame(self, frame: np.ndarray) -> bool:
        """Add frame to buffer"""
        with self.lock:
            if self.buffer.full():
                # Remove oldest frame
                try:
                    self.buffer.get_nowait()
                except:
                    pass
            
            try:
                self.buffer.put_nowait(frame.copy())
                return True
            except:
                return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get frame from buffer"""
        with self.lock:
            try:
                return self.buffer.get_nowait()
            except:
                return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame, discarding older ones"""
        latest_frame = None
        with self.lock:
            while not self.buffer.empty():
                try:
                    latest_frame = self.buffer.get_nowait()
                except:
                    break
        return latest_frame
    
    def size(self) -> int:
        """Get current buffer size"""
        return self.buffer.qsize()


def start_camera_thread(cap: cv2.VideoCapture, frame_buffer: FrameBuffer,
                       running_flag: threading.Event, fps_limit: float = 15.0) -> threading.Thread:
    """
    Start camera capture thread
    
    Args:
        cap: Camera capture object
        frame_buffer: Buffer to store frames
        running_flag: Thread running control flag
        fps_limit: Maximum FPS for capture
        
    Returns:
        Camera thread object
    """
    def camera_worker():
        """Camera capture worker function"""
        frame_interval = 1.0 / fps_limit
        last_capture_time = 0
        
        while running_flag.is_set():
            current_time = time.time()
            
            # Respect FPS limit
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.01)
                continue
            
            # Capture frame
            success, frame = capture_frame(cap)
            
            if success and validate_frame(frame):
                frame_buffer.put_frame(frame)
                last_capture_time = current_time
            else:
                # If capture fails, wait briefly before retrying
                time.sleep(0.1)
    
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()
    return camera_thread


def get_camera_stats(cap: cv2.VideoCapture, frame_count: int, start_time: float) -> Dict[str, Any]:
    """
    Get camera performance statistics
    
    Args:
        cap: Camera capture object
        frame_count: Total frames processed
        start_time: Start time of monitoring
        
    Returns:
        Camera statistics dictionary
    """
    if cap is None:
        return {
            'connected': False,
            'fps': 0.0,
            'frames_processed': frame_count,
            'runtime': time.time() - start_time,
            'avg_fps': 0.0
        }
    
    current_time = time.time()
    runtime = current_time - start_time
    avg_fps = frame_count / max(runtime, 1)
    
    try:
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend = cap.get(cv2.CAP_PROP_BACKEND)
        
        return {
            'connected': cap.isOpened(),
            'fps': camera_fps,
            'resolution': (width, height),
            'backend': backend,
            'frames_processed': frame_count,
            'runtime': runtime,
            'avg_fps': avg_fps
        }
        
    except Exception as e:
        logging.warning(f"Error getting camera stats: {str(e)}")
        return {
            'connected': False,
            'fps': 0.0,
            'frames_processed': frame_count,
            'runtime': runtime,
            'avg_fps': avg_fps,
            'error': str(e)
        }


def release_camera(cap: cv2.VideoCapture, camera_thread: Optional[threading.Thread] = None,
                  running_flag: Optional[threading.Event] = None) -> bool:
    """
    Properly release camera resources
    
    Args:
        cap: Camera capture object
        camera_thread: Camera thread to stop
        running_flag: Thread running control flag
        
    Returns:
        True if successfully released
    """
    try:
        # Stop camera thread if running
        if running_flag is not None:
            running_flag.clear()
        
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)
        
        # Release camera
        if cap is not None:
            cap.release()
        
        logging.info("Camera resources released successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error releasing camera resources: {str(e)}")
        return False


def test_camera_connection(camera_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test camera connection and capabilities
    
    Args:
        camera_config: Camera configuration
        
    Returns:
        Test result dictionary
    """
    test_result = {
        'success': False,
        'error': None,
        'fps': 0,
        'resolution': None,
        'frame_captured': False,
        'latency_ms': 0
    }
    
    cap = None
    try:
        start_time = time.time()
        
        # Initialize camera
        cap, status_info = initialize_camera(camera_config)
        
        if not status_info['connected']:
            test_result['error'] = status_info['error']
            return test_result
        
        # Test frame capture
        success, frame = capture_frame(cap)
        capture_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if success and validate_frame(frame):
            test_result.update({
                'success': True,
                'fps': status_info['fps'],
                'resolution': status_info['resolution'],
                'frame_captured': True,
                'latency_ms': capture_time
            })
        else:
            test_result['error'] = "Failed to capture valid frame"
        
    except Exception as e:
        test_result['error'] = str(e)
    
    finally:
        if cap is not None:
            cap.release()
    
    return test_result
