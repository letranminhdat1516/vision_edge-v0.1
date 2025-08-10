#!/usr/bin/env python3
"""
Statistics Manager
Function-based statistics tracking for fall detection
"""

import time
from typing import Dict, Any


def create_statistics_state() -> Dict[str, Any]:
    """
    Tạo trạng thái thống kê ban đầu
    
    Returns:
        Dictionary chứa trạng thái thống kê
    """
    return {
        'start_time': time.time(),
        'frame_count': 0,
        'fall_count': 0,
        'last_fall_time': "Never",
        'last_confidence': 0.0,
        'fps_update_time': time.time(),
        'current_fps': 0.0
    }


def update_frame_statistics(stats_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cập nhật thống kê frame
    
    Args:
        stats_state: Trạng thái thống kê hiện tại
        
    Returns:
        Trạng thái thống kê đã cập nhật
    """
    updated_stats = stats_state.copy()
    updated_stats['frame_count'] += 1
    
    # Update FPS every second
    current_time = time.time()
    if current_time - updated_stats['fps_update_time'] >= 1.0:
        elapsed = current_time - updated_stats['start_time']
        updated_stats['current_fps'] = updated_stats['frame_count'] / max(elapsed, 1)
        updated_stats['fps_update_time'] = current_time
    
    return updated_stats


def update_fall_statistics(stats_state: Dict[str, Any], confidence: float) -> Dict[str, Any]:
    """
    Cập nhật thống kê khi detect fall
    
    Args:
        stats_state: Trạng thái thống kê hiện tại
        confidence: Confidence của fall detection
        
    Returns:
        Trạng thái thống kê đã cập nhật
    """
    updated_stats = stats_state.copy()
    updated_stats['fall_count'] += 1
    updated_stats['last_fall_time'] = time.strftime("%H:%M:%S")
    updated_stats['last_confidence'] = confidence
    
    return updated_stats


def reset_statistics(stats_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset thống kê về trạng thái ban đầu
    
    Args:
        stats_state: Trạng thái thống kê hiện tại
        
    Returns:
        Trạng thái thống kê đã reset
    """
    return create_statistics_state()


def get_formatted_statistics(stats_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lấy thống kê đã format để hiển thị
    
    Args:
        stats_state: Trạng thái thống kê hiện tại
        
    Returns:
        Dictionary thống kê formatted
    """
    current_time = time.time()
    runtime = current_time - stats_state['start_time']
    
    # Calculate detection rate
    detection_rate = 0.0
    if stats_state['frame_count'] > 0:
        detection_rate = (stats_state['fall_count'] / stats_state['frame_count']) * 100
    
    # Format runtime
    if runtime < 60:
        runtime_str = f"{runtime:.1f}s"
    elif runtime < 3600:
        minutes = int(runtime // 60)
        seconds = runtime % 60
        runtime_str = f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        runtime_str = f"{hours}h {minutes}m"
    
    return {
        'fall_count': stats_state['fall_count'],
        'total_frames': stats_state['frame_count'],
        'runtime': runtime_str,
        'current_fps': stats_state['current_fps'],
        'last_fall_time': stats_state['last_fall_time'],
        'last_confidence': stats_state['last_confidence'],
        'detection_rate': detection_rate,
        'status': 'Active'
    }


def print_fall_detection_alert(stats_state: Dict[str, Any]) -> None:
    """
    In thông báo khi detect fall
    
    Args:
        stats_state: Trạng thái thống kê hiện tại
    """
    print(f"FALL DETECTED! Count: {stats_state['fall_count']} | "
          f"Time: {stats_state['last_fall_time']} | "
          f"Confidence: {stats_state['last_confidence']:.3f}")


def print_final_statistics(stats_state: Dict[str, Any]) -> None:
    """
    In thống kê cuối cùng khi thoát
    
    Args:
        stats_state: Trạng thái thống kê cuối cùng
    """
    final_stats = get_formatted_statistics(stats_state)
    
    print(f"\nFinal Statistics:")
    print(f"Runtime: {final_stats['runtime']}")
    print(f"Total Falls Detected: {final_stats['fall_count']}")
    print(f"Total Frames: {final_stats['total_frames']}")
    print(f"Average FPS: {final_stats['current_fps']:.2f}")
    print(f"Detection Rate: {final_stats['detection_rate']:.2f}%")
