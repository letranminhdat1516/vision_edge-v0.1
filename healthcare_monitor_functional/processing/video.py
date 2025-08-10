#!/usr/bin/env python3
"""
Video Processing Functions  
Function-based video processing and analysis system
"""

import cv2
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def detect_objects_yolo(frame: np.ndarray, model: Any, confidence_threshold: float = 0.5,
                       target_classes: List[int] = None) -> List[Dict[str, Any]]:
    """
    Detect objects using YOLO model
    
    Args:
        frame: Input frame
        model: YOLO model instance  
        confidence_threshold: Confidence threshold for detections
        target_classes: List of class IDs to detect (None for all)
        
    Returns:
        List of detection dictionaries
    """
    if model is None or frame is None:
        return []
    
    try:
        results = model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        
        if results is not None and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get confidence and class
                    conf = float(boxes.conf[i].cpu() if hasattr(boxes.conf[i], 'cpu') else boxes.conf[i])
                    cls_id = int(boxes.cls[i].cpu() if hasattr(boxes.cls[i], 'cpu') else boxes.cls[i])
                    
                    # Filter by target classes if specified
                    if target_classes is not None and cls_id not in target_classes:
                        continue
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': model.names.get(cls_id, f"class_{cls_id}") if hasattr(model, 'names') else f"class_{cls_id}"
                    }
                    
                    detections.append(detection)
        
        return detections
        
    except Exception as e:
        logging.error(f"YOLO detection error: {str(e)}")
        return []


def find_largest_person(detections: List[Dict[str, Any]], person_class_id: int = 0) -> Optional[Dict[str, Any]]:
    """
    Find the largest person detection in the frame
    
    Args:
        detections: List of detection dictionaries
        person_class_id: Class ID for person (default: 0 for COCO)
        
    Returns:
        Largest person detection or None
    """
    person_detections = [det for det in detections if det['class_id'] == person_class_id]
    
    if not person_detections:
        return None
    
    # Find person with largest area
    largest_person = max(person_detections, key=lambda det: 
                        (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1]))
    
    return largest_person


def extract_person_crop(frame: np.ndarray, bbox: List[int], padding: int = 10) -> Optional[np.ndarray]:
    """
    Extract person crop from frame with padding
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        padding: Padding around bounding box
        
    Returns:
        Cropped person image or None
    """
    if frame is None or not bbox:
        return None
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding and ensure within frame bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Extract crop
    person_crop = frame[y1:y2, x1:x2]
    
    if person_crop.size == 0:
        return None
    
    return person_crop


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize frame to target size
    
    Args:
        frame: Input frame
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if frame is None:
        return frame
    
    target_width, target_height = target_size
    
    if maintain_aspect:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        if target_width / target_height > aspect_ratio:
            # Height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            # Width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        
        # Resize with aspect ratio
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
        
        # Center the resized frame
        start_y = (target_height - new_height) // 2
        start_x = (target_width - new_width) // 2
        canvas[start_y:start_y + new_height, start_x:start_x + new_width] = resized
        
        return canvas
    else:
        return cv2.resize(frame, target_size)


def apply_frame_enhancements(frame: np.ndarray, brightness: float = 0.0, 
                           contrast: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """
    Apply enhancement filters to frame
    
    Args:
        frame: Input frame
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        gamma: Gamma correction (0.1 to 3.0)
        
    Returns:
        Enhanced frame
    """
    if frame is None:
        return frame
    
    enhanced = frame.astype(np.float32)
    
    # Apply brightness
    enhanced = enhanced + brightness
    
    # Apply contrast
    enhanced = enhanced * contrast
    
    # Apply gamma correction
    if gamma != 1.0:
        enhanced = enhanced / 255.0
        enhanced = np.power(enhanced, gamma)
        enhanced = enhanced * 255.0
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray, 
                             threshold: int = 30) -> Tuple[float, np.ndarray]:
    """
    Calculate difference between two frames
    
    Args:
        frame1: First frame
        frame2: Second frame  
        threshold: Threshold for binary difference
        
    Returns:
        Tuple of (difference_percentage, difference_mask)
    """
    if frame1 is None or frame2 is None:
        return 0.0, np.array([])
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    total_pixels = binary_diff.shape[0] * binary_diff.shape[1]
    changed_pixels = np.count_nonzero(binary_diff)
    difference_percentage = (changed_pixels / total_pixels) * 100
    
    return difference_percentage, binary_diff


def detect_motion_regions(frame1: np.ndarray, frame2: np.ndarray,
                         min_area: int = 500) -> List[Tuple[int, int, int, int]]:
    """
    Detect motion regions between two frames
    
    Args:
        frame1: Previous frame
        frame2: Current frame
        min_area: Minimum area for motion regions
        
    Returns:
        List of motion region bounding boxes (x, y, w, h)
    """
    if frame1 is None or frame2 is None:
        return []
    
    # Calculate frame difference
    _, diff_mask = calculate_frame_difference(frame1, frame2)
    
    if diff_mask.size == 0:
        return []
    
    # Find contours
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))
    
    return motion_regions


def stabilize_frame(frame: np.ndarray, reference_frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply basic frame stabilization
    
    Args:
        frame: Current frame to stabilize
        reference_frame: Reference frame for stabilization
        
    Returns:
        Stabilized frame or None if stabilization fails
    """
    if frame is None or reference_frame is None:
        return frame
    
    try:
        # Convert to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        detector = cv2.ORB_create()
        kp1, des1 = detector.detectAndCompute(gray_reference, None)
        kp2, des2 = detector.detectAndCompute(gray_current, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return frame
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        if len(matches) < 10:
            return frame
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Calculate transformation matrix
        M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        
        if M is None:
            return frame
        
        # Apply transformation
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, M, (w, h))
        
        return stabilized
        
    except Exception as e:
        logging.warning(f"Frame stabilization failed: {str(e)}")
        return frame


def extract_keyframe_features(frame: np.ndarray, method: str = 'orb') -> Optional[Tuple[Any, Any]]:
    """
    Extract keypoint features from frame
    
    Args:
        frame: Input frame
        method: Feature extraction method ('orb', 'sift', 'surf')
        
    Returns:
        Tuple of (keypoints, descriptors) or None
    """
    if frame is None:
        return None
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if method.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=1000)
        elif method.lower() == 'sift':
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create(nfeatures=1000)  # Default to ORB
        
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors
        
    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}")
        return None


def is_frame_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if frame is blurry using Laplacian variance
    
    Args:
        frame: Input frame
        threshold: Blur threshold (lower values indicate more blur)
        
    Returns:
        True if frame is blurry, False otherwise
    """
    if frame is None:
        return True
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    except:
        return True


def calculate_frame_quality(frame: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive frame quality metrics
    
    Args:
        frame: Input frame
        
    Returns:
        Dictionary of quality metrics
    """
    if frame is None:
        return {
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'noise_level': 1.0,
            'overall_quality': 0.0
        }
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Noise level (estimated from high-frequency content)
        blur_kernel = np.ones((5,5)) / 25
        smooth = cv2.filter2D(gray, -1, blur_kernel)
        noise_level = np.mean(np.abs(gray.astype(float) - smooth.astype(float)))
        
        # Overall quality score (0-1, higher is better)
        quality_score = min(1.0, (sharpness / 500.0) * (contrast / 50.0) * (1.0 - noise_level / 50.0))
        
        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast, 
            'noise_level': noise_level,
            'overall_quality': quality_score
        }
        
    except Exception as e:
        logging.error(f"Frame quality calculation error: {str(e)}")
        return {
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'noise_level': 1.0,
            'overall_quality': 0.0
        }
