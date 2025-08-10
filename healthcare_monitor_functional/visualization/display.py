#!/usr/bin/env python3
"""
Display Functions - Fall Detection Only
Function-based display system for dual screen and statistics
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.04) -> np.ndarray:
    """
    Vẽ keypoints lên frame để thấy pose detection
    
    Args:
        frame: Input frame
        keypoints: MoveNet keypoints (17 points)
        confidence_threshold: Ngưỡng confidence để hiển thị keypoint
        
    Returns:
        Frame với keypoints được vẽ
    """
    display_frame = frame.copy()
    
    # Check if keypoints is valid
    if keypoints is None or keypoints.size == 0:
        print("Warning: keypoints is None or empty")
        return display_frame
    
    # Debug: print keypoints shape and some values (reduced logging)
    # print(f"DEBUG: keypoints shape = {keypoints.shape}")
    # if keypoints.size > 0:
    #     print(f"DEBUG: keypoints min/max = {np.min(keypoints):.3f}/{np.max(keypoints):.3f}")
    
    # Handle different keypoints shapes from MoveNet
    if len(keypoints.shape) == 4 and keypoints.shape[0] == 1 and keypoints.shape[1] == 1:
        # Shape (1, 1, 17, 3) - extract to (17, 3)
        kp = keypoints[0, 0]
        # print("DEBUG: Converted from (1,1,17,3) to (17,3)")
    elif len(keypoints.shape) == 2 and keypoints.shape[0] == 17 and keypoints.shape[1] == 3:
        # Already (17, 3)
        kp = keypoints
        # print("DEBUG: Already (17,3) shape")
    else:
        print(f"Warning: Unexpected keypoints shape: {keypoints.shape}")
        return display_frame
    
    height, width = frame.shape[:2]
    
    # MoveNet keypoint connections (skeleton)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Body
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    visible_points = 0
    confidence_values = []
    try:
        for i in range(17):
            y_norm = float(kp[i, 0])  # MoveNet: [y, x, confidence]
            x_norm = float(kp[i, 1])
            confidence = float(kp[i, 2])
            confidence_values.append(confidence)
            
            # Convert to pixel coordinates
            y = int(y_norm * height)
            x = int(x_norm * width)
            
            if confidence > confidence_threshold:
                points.append((x, y))
                # Draw keypoint with confidence color - make it bigger and more visible
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.02 else (0, 0, 255)
                cv2.circle(display_frame, (x, y), 6, color, -1)  # Bigger circles
                cv2.circle(display_frame, (x, y), 8, (255, 255, 255), 2)  # White border
                # Add keypoint index for debugging
                cv2.putText(display_frame, str(i), (x-15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                visible_points += 1
            else:
                points.append(None)
        
        # Debug confidence values (reduced logging)
        max_confidence = max(confidence_values) if confidence_values else 0
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        print(f"Drew {visible_points}/17 keypoints with confidence > {confidence_threshold} (max: {max_confidence:.3f})")
        
        # Draw skeleton connections
        connections_drawn = 0
        for start_idx, end_idx in connections:
            if points[start_idx] is not None and points[end_idx] is not None:
                # Draw thicker, more visible skeleton lines
                cv2.line(display_frame, points[start_idx], points[end_idx], (255, 0, 0), 3)  # Thicker blue lines
                cv2.line(display_frame, points[start_idx], points[end_idx], (255, 255, 255), 1)  # White outline
                connections_drawn += 1
        
        print(f"Drew {connections_drawn} skeleton connections")
    
    except Exception as e:
        print(f"Error drawing keypoints: {str(e)}")
        return display_frame
    
    return display_frame


def create_statistics_panel(stats: Dict[str, Any], panel_size: Tuple[int, int] = (400, 300)) -> np.ndarray:
    """
    Tạo panel thống kê cho màn hình thứ 2
    
    Args:
        stats: Dictionary chứa thống kê
        panel_size: Kích thước panel (width, height)
        
    Returns:
        Panel statistics như một numpy array
    """
    width, height = panel_size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background color
    panel[:] = (40, 40, 40)
    
    # Title
    cv2.putText(panel, 'FALL DETECTION STATS', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Horizontal line
    cv2.line(panel, (20, 50), (width-20, 50), (255, 255, 255), 1)
    
    y_pos = 80
    line_height = 30
    
    # Display statistics
    stat_items = [
        f"Runtime: {stats.get('runtime', '0.0')}s",
        f"Total Frames: {stats.get('total_frames', 0)}",
        f"Current FPS: {stats.get('current_fps', 0.0):.1f}",
        f"Fall Detections: {stats.get('fall_count', 0)}",
        f"Last Detection: {stats.get('last_fall_time', 'Never')}",
        f"Detection Rate: {stats.get('detection_rate', 0.0):.2f}%",
        f"Model Confidence: {stats.get('last_confidence', 0.0):.3f}",
        f"Status: {stats.get('status', 'Running')}"
    ]
    
    for item in stat_items:
        cv2.putText(panel, item, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
    
    return panel


def display_dual_windows(main_frame: np.ndarray, keypoints_frame: np.ndarray, 
                        stats: Dict[str, Any], fall_detected: bool = False) -> None:
    """
    Hiển thị 2 màn hình: chính (RTSP stream) và phụ (keypoints + stats)
    
    Args:
        main_frame: Frame chính từ RTSP
        keypoints_frame: Frame với keypoints được vẽ
        stats: Thống kê để hiển thị
        fall_detected: Có detect fall không
    """
    # Main window - RTSP stream with fall alert
    display_main = main_frame.copy()
    
    if fall_detected:
        # Highlight fall detection
        cv2.rectangle(display_main, (0, 0), (display_main.shape[1], display_main.shape[0]), 
                     (0, 0, 255), 10)
        cv2.putText(display_main, 'FALL DETECTED!', (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    # Add basic info to main frame
    cv2.putText(display_main, f"FPS: {stats.get('current_fps', 0.0):.1f}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_main, f"Falls: {stats.get('fall_count', 0)}", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Secondary window - Keypoints + Statistics
    stats_panel = create_statistics_panel(stats)
    
    # Resize keypoints frame to match stats panel width if needed
    if keypoints_frame.shape[1] != stats_panel.shape[1]:
        aspect_ratio = keypoints_frame.shape[0] / keypoints_frame.shape[1]
        new_width = stats_panel.shape[1]
        new_height = int(new_width * aspect_ratio)
        keypoints_frame = cv2.resize(keypoints_frame, (new_width, new_height))
    
    # Combine keypoints frame and stats panel vertically
    combined_secondary = np.vstack([keypoints_frame, stats_panel])
    
    # Display windows
    cv2.imshow('IMOU RTSP Stream - Fall Detection', display_main)
    cv2.imshow('Pose Detection & Statistics', combined_secondary)


def calculate_fps(frame_count: int, start_time: float) -> float:
    """
    Tính FPS hiện tại
    
    Args:
        frame_count: Số frame đã xử lý
        start_time: Thời gian bắt đầu
        
    Returns:
        FPS hiện tại
    """
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0


def format_time(seconds: float) -> str:
    """
    Format thời gian thành chuỗi dễ đọc
    
    Args:
        seconds: Thời gian tính bằng giây
        
    Returns:
        Chuỗi thời gian formatted
    """
    if seconds < 60:
        return f"{seconds:.1f}"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}"


def create_fall_statistics(fall_count: int, total_frames: int, runtime: float) -> Dict[str, Any]:
    """
    Tạo thống kê fall detection
    
    Args:
        fall_count: Số lần detect fall
        total_frames: Tổng số frame
        runtime: Thời gian chạy
        
    Returns:
        Dictionary thống kê
    """
    detection_rate = (fall_count / max(total_frames, 1)) * 100
    avg_fps = total_frames / max(runtime, 1)
    
    return {
        'fall_count': fall_count,
        'total_frames': total_frames,
        'runtime': format_time(runtime),
        'detection_rate': detection_rate,
        'avg_fps': avg_fps,
        'status': 'Active'
    }
