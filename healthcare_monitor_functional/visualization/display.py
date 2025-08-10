#!/usr/bin/env python3
"""
Visualization Functions  
Function-based visualization and display system
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union


def draw_bounding_box(frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2, label: str = None, confidence: float = None) -> np.ndarray:
    """
    Draw bounding box on frame with optional label and confidence
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        color: Box color (BGR)
        thickness: Line thickness
        label: Optional label text
        confidence: Optional confidence value
        
    Returns:
        Frame with bounding box drawn
    """
    if frame is None or not bbox:
        return frame
    
    frame_copy = frame.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Add label with confidence if provided
    if label is not None:
        label_text = label
        if confidence is not None:
            label_text = f"{label}: {confidence:.2f}"
        
        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw label background
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(frame_copy, 
                     (x1, label_y - text_height - 5), 
                     (x1 + text_width + 5, label_y + 5), 
                     color, -1)
        
        # Draw label text
        cv2.putText(frame_copy, label_text, (x1 + 3, label_y - 3), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return frame_copy


def draw_keypoints(frame: np.ndarray, keypoints: List[Tuple[int, int]], 
                  color: Tuple[int, int, int] = (255, 0, 0), radius: int = 3) -> np.ndarray:
    """
    Draw keypoints on frame
    
    Args:
        frame: Input frame
        keypoints: List of keypoint coordinates (x, y)
        color: Point color (BGR)
        radius: Point radius
        
    Returns:
        Frame with keypoints drawn
    """
    if frame is None or not keypoints:
        return frame
    
    frame_copy = frame.copy()
    
    for keypoint in keypoints:
        if len(keypoint) >= 2:
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(frame_copy, (x, y), radius, color, -1)
    
    return frame_copy


def draw_skeleton(frame: np.ndarray, keypoints: List[Tuple[int, int]], 
                 connections: List[Tuple[int, int]], 
                 point_color: Tuple[int, int, int] = (255, 0, 0),
                 line_color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2) -> np.ndarray:
    """
    Draw skeleton connections on frame
    
    Args:
        frame: Input frame
        keypoints: List of keypoint coordinates
        connections: List of keypoint connection pairs
        point_color: Keypoint color (BGR)
        line_color: Connection line color (BGR)
        thickness: Line thickness
        
    Returns:
        Frame with skeleton drawn
    """
    if frame is None or not keypoints or not connections:
        return frame
    
    frame_copy = frame.copy()
    
    # Draw connections
    for connection in connections:
        if len(connection) >= 2:
            start_idx, end_idx = connection[:2]
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx] is not None and keypoints[end_idx] is not None):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                cv2.line(frame_copy, start_point, end_point, line_color, thickness)
    
    # Draw keypoints on top
    frame_copy = draw_keypoints(frame_copy, keypoints, point_color)
    
    return frame_copy


def draw_motion_indicators(frame: np.ndarray, motion_level: float, position: str = "top-left") -> np.ndarray:
    """
    Draw motion level indicators on frame
    
    Args:
        frame: Input frame
        motion_level: Motion level (0-1)
        position: Position for indicators ("top-left", "top-right", "bottom-left", "bottom-right")
        
    Returns:
        Frame with motion indicators
    """
    if frame is None:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Determine position
    if position == "top-right":
        base_x, base_y = w - 150, 30
    elif position == "bottom-left":
        base_x, base_y = 20, h - 50
    elif position == "bottom-right":
        base_x, base_y = w - 150, h - 50
    else:  # top-left (default)
        base_x, base_y = 20, 30
    
    # Draw motion bar
    bar_width = 100
    bar_height = 10
    fill_width = int(bar_width * motion_level)
    
    # Background bar
    cv2.rectangle(frame_copy, (base_x, base_y), (base_x + bar_width, base_y + bar_height), 
                 (100, 100, 100), -1)
    
    # Motion level bar
    if motion_level > 0.7:
        color = (0, 0, 255)  # Red for high motion
    elif motion_level > 0.4:
        color = (0, 255, 255)  # Yellow for medium motion
    else:
        color = (0, 255, 0)  # Green for low motion
    
    cv2.rectangle(frame_copy, (base_x, base_y), (base_x + fill_width, base_y + bar_height), 
                 color, -1)
    
    # Motion level text
    motion_text = f"Motion: {motion_level:.2f}"
    cv2.putText(frame_copy, motion_text, (base_x, base_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame_copy


def draw_detection_alerts(frame: np.ndarray, alerts: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw detection alerts on frame
    
    Args:
        frame: Input frame
        alerts: List of alert dictionaries
        
    Returns:
        Frame with alerts drawn
    """
    if frame is None or not alerts:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    alert_y = 50
    for alert in alerts:
        alert_type = alert.get('type', 'unknown')
        message = alert.get('message', 'Alert')
        confidence = alert.get('confidence', 0.0)
        
        # Choose color based on alert type
        if alert_type == 'fall':
            color = (0, 0, 255)  # Red
        elif alert_type == 'seizure':
            color = (255, 0, 255)  # Magenta
        else:
            color = (0, 255, 255)  # Yellow
        
        # Draw alert background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        alert_text = f"{message} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(alert_text, font, font_scale, font_thickness)
        
        cv2.rectangle(frame_copy, 
                     (10, alert_y - text_height - 10), 
                     (20 + text_width, alert_y + 5), 
                     color, -1)
        
        # Draw alert text
        cv2.putText(frame_copy, alert_text, (15, alert_y - 5), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        alert_y += text_height + 20
    
    return frame_copy


def draw_statistics_overlay(frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """
    Draw statistics overlay on frame
    
    Args:
        frame: Input frame
        stats: Statistics dictionary
        
    Returns:
        Frame with statistics overlay
    """
    if frame is None or not stats:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (w - 250, 10), (w - 10, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw statistics text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_color = (255, 255, 255)
    
    y_offset = 30
    line_height = 15
    
    for key, value in stats.items():
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        elif isinstance(value, int):
            text = f"{key}: {value}"
        else:
            text = f"{key}: {str(value)}"
        
        cv2.putText(frame_copy, text, (w - 240, y_offset), 
                   font, font_scale, text_color, font_thickness)
        y_offset += line_height
    
    return frame_copy


def draw_confidence_graph(frame: np.ndarray, confidence_history: List[float], 
                         graph_position: Tuple[int, int] = None,
                         graph_size: Tuple[int, int] = (200, 100)) -> np.ndarray:
    """
    Draw confidence history graph on frame
    
    Args:
        frame: Input frame
        confidence_history: List of confidence values
        graph_position: Graph position (x, y), None for automatic
        graph_size: Graph size (width, height)
        
    Returns:
        Frame with confidence graph
    """
    if frame is None or not confidence_history:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    graph_width, graph_height = graph_size
    
    if graph_position is None:
        graph_x = w - graph_width - 20
        graph_y = h - graph_height - 20
    else:
        graph_x, graph_y = graph_position
    
    # Draw graph background
    cv2.rectangle(frame_copy, 
                 (graph_x, graph_y), 
                 (graph_x + graph_width, graph_y + graph_height), 
                 (50, 50, 50), -1)
    
    # Draw grid lines
    for i in range(0, graph_width, 40):
        cv2.line(frame_copy, 
                (graph_x + i, graph_y), 
                (graph_x + i, graph_y + graph_height), 
                (100, 100, 100), 1)
    
    for i in range(0, graph_height, 20):
        cv2.line(frame_copy, 
                (graph_x, graph_y + i), 
                (graph_x + graph_width, graph_y + i), 
                (100, 100, 100), 1)
    
    # Draw confidence curve
    if len(confidence_history) > 1:
        points = []
        for i, conf in enumerate(confidence_history[-graph_width:]):  # Only show recent history
            x = graph_x + int((i / max(len(confidence_history) - 1, 1)) * graph_width)
            y = graph_y + graph_height - int(conf * graph_height)
            points.append((x, y))
        
        # Draw lines between points
        for i in range(len(points) - 1):
            cv2.line(frame_copy, points[i], points[i + 1], (0, 255, 0), 2)
        
        # Draw current confidence point
        if points:
            cv2.circle(frame_copy, points[-1], 3, (0, 0, 255), -1)
    
    # Draw labels
    cv2.putText(frame_copy, "1.0", (graph_x - 20, graph_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(frame_copy, "0.0", (graph_x - 20, graph_y + graph_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame_copy


def create_status_display(frame_size: Tuple[int, int], status_info: Dict[str, Any]) -> np.ndarray:
    """
    Create status display frame
    
    Args:
        frame_size: Size of status frame (width, height)
        status_info: Status information dictionary
        
    Returns:
        Status display frame
    """
    width, height = frame_size
    status_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw title
    cv2.putText(status_frame, "Healthcare Monitor Status", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw status information
    y_offset = 70
    line_height = 25
    
    for key, value in status_info.items():
        if isinstance(value, bool):
            color = (0, 255, 0) if value else (0, 0, 255)
            status_text = "ON" if value else "OFF"
            text = f"{key}: {status_text}"
        elif isinstance(value, float):
            color = (255, 255, 255)
            text = f"{key}: {value:.2f}"
        else:
            color = (255, 255, 255)
            text = f"{key}: {str(value)}"
        
        cv2.putText(status_frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_offset += line_height
    
    return status_frame


def apply_visual_effects(frame: np.ndarray, effect: str = "none", 
                        intensity: float = 0.5) -> np.ndarray:
    """
    Apply visual effects to frame
    
    Args:
        frame: Input frame
        effect: Effect type ("none", "blur", "sharpen", "edge", "sepia")
        intensity: Effect intensity (0-1)
        
    Returns:
        Frame with applied effect
    """
    if frame is None or effect == "none":
        return frame
    
    frame_copy = frame.copy()
    
    try:
        if effect == "blur":
            kernel_size = int(5 + intensity * 10)
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame_copy = cv2.GaussianBlur(frame_copy, (kernel_size, kernel_size), 0)
            
        elif effect == "sharpen":
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * intensity
            frame_copy = cv2.filter2D(frame_copy, -1, kernel)
            
        elif effect == "edge":
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame_copy = cv2.addWeighted(frame_copy, 1 - intensity, edges_colored, intensity, 0)
            
        elif effect == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            sepia_frame = cv2.transform(frame_copy, kernel)
            frame_copy = cv2.addWeighted(frame_copy, 1 - intensity, sepia_frame, intensity, 0)
    
    except Exception as e:
        print(f"Visual effect error: {str(e)}")
        return frame
    
    return frame_copy
