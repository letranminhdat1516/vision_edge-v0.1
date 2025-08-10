#!/usr/bin/env python3
"""
Display Controller
Function-based display control logic
"""

from typing import Dict, Any
from visualization.display import draw_keypoints, display_dual_windows, create_fall_statistics


def create_display_system() -> Dict[str, Any]:
    """
    Khởi tạo hệ thống display
    
    Returns:
        Dictionary chứa cấu hình display
    """
    return {
        'dual_screen_enabled': True,
        'show_keypoints': True,
        'show_statistics': True,
        'initialized': True
    }


def render_display_frames(frame, keypoints, stats_state: Dict[str, Any], 
                         fall_detected: bool, display_system: Dict[str, Any]) -> None:
    """
    Render và hiển thị các frame
    
    Args:
        frame: Frame gốc từ camera
        keypoints: Keypoints từ MoveNet
        stats_state: Trạng thái thống kê
        fall_detected: Có detect fall không
        display_system: Hệ thống display
    """
    if not display_system['initialized']:
        return
    
    try:
        # Create keypoints frame if enabled
        if display_system['show_keypoints'] and keypoints is not None:
            keypoints_frame = draw_keypoints(frame, keypoints)
        else:
            keypoints_frame = frame.copy()
        
        # Get formatted statistics
        if display_system['show_statistics']:
            from logic.statistics import get_formatted_statistics
            formatted_stats = get_formatted_statistics(stats_state)
        else:
            formatted_stats = {}
        
        # Display frames
        if display_system['dual_screen_enabled']:
            display_dual_windows(frame, keypoints_frame, formatted_stats, fall_detected)
        else:
            # Single screen fallback
            import cv2
            display_frame = keypoints_frame if display_system['show_keypoints'] else frame
            
            if fall_detected:
                cv2.putText(display_frame, 'FALL DETECTED!', (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            cv2.imshow('Fall Detection', display_frame)
            
    except Exception as e:
        print(f"Display error: {str(e)}")


def cleanup_display_system() -> None:
    """
    Cleanup display system
    """
    import cv2
    cv2.destroyAllWindows()


def print_system_instructions() -> None:
    """
    In hướng dẫn sử dụng hệ thống
    """
    print("Fall Detection System Started")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset statistics")
    print("Display:")
    print("  - Window 1: RTSP Stream with Fall Alerts")
    print("  - Window 2: Pose Detection & Statistics")
