#!/usr/bin/env python3
"""
Detection Controller
Function-based fall detection control logic
"""

import cv2
import time
from typing import Dict, Any, Tuple, Optional
from detection.fall_functional.fall_detection_functional import run_movenet_inference, detect_fall_from_keypoints, load_movenet_model, preprocess_frame


def initialize_fall_detection_system(model_path: str, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Khởi tạo hệ thống fall detection
    
    Args:
        model_path: Đường dẫn đến model
        threshold: Ngưỡng detect fall
        
    Returns:
        Dictionary chứa components của fall detection
    """
    try:
        interpreter = load_movenet_model(model_path)
        print(f"Fall detection model loaded: {model_path}")
        
        return {
            'interpreter': interpreter,
            'threshold': threshold,
            'initialized': True,
            'error': None
        }
    except Exception as e:
        print(f"Failed to initialize fall detection: {str(e)}")
        return {
            'interpreter': None,
            'threshold': threshold,
            'initialized': False,
            'error': str(e)
        }


def process_frame_for_fall_detection(frame, fall_system: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """
    Xử lý frame cho fall detection
    
    Args:
        frame: Frame đầu vào từ camera
        fall_system: Hệ thống fall detection
        
    Returns:
        Tuple of (fall_result, keypoints)
    """
    if not fall_system['initialized']:
        return {
            'fall_detected': False,
            'confidence': 0.0,
            'error': fall_system['error']
        }, None
    
    try:
        # Preprocess frame
        input_frame = preprocess_frame(frame)
        
        # Run inference
        keypoints = run_movenet_inference(fall_system['interpreter'], input_frame)
        
        # Detect fall
        fall_result = detect_fall_from_keypoints(keypoints, fall_system['threshold'])
        
        return fall_result, keypoints
        
    except Exception as e:
        return {
            'fall_detected': False,
            'confidence': 0.0,
            'error': str(e)
        }, None


def check_user_input() -> str:
    """
    Kiểm tra input từ user
    
    Returns:
        String command ('quit', 'reset', 'continue')
    """
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        return 'quit'
    elif key == ord('r'):
        return 'reset'
    else:
        return 'continue'


def handle_frame_processing_error(frame, error_message: str) -> bool:
    """
    Xử lý lỗi trong quá trình xử lý frame
    
    Args:
        frame: Frame gây lỗi
        error_message: Thông báo lỗi
        
    Returns:
        True nếu có thể tiếp tục, False nếu cần dừng
    """
    print(f"Frame processing error: {error_message}")
    
    # Sleep một chút để tránh loop lỗi liên tục
    time.sleep(0.1)
    
    # Có thể tiếp tục với frame tiếp theo
    return True
