#!/usr/bin/env python3
"""
Healthcare Monitor Main Application - Pure Functional Programming
Function-based healthcare monitoring system entry point
"""

import sys
import cv2
import time
import threading
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import function-based modules
from core.config import load_config
from core.logger import setup_logging
from core.utils import calculate_runtime_stats
from camera.controls import initialize_camera, FrameBuffer, start_camera_thread, release_camera
from processing.video import detect_objects_yolo, find_largest_person, calculate_frame_quality, extract_keyframe_features
from detection.motion import calculate_motion_level, analyze_motion_patterns
from detection.fall import process_fall_detection, initialize_fall_detector
from detection.seizure import process_seizure_detection, initialize_seizure_detector
from visualization.display import (draw_bounding_box, draw_motion_indicators, 
                                  draw_detection_alerts, draw_statistics_overlay)
from alerts.management import AlertManager, format_alert_message


# Pure data structures using NamedTuple (immutable)
class MonitoringState(NamedTuple):
    """Immutable state structure for monitoring system"""
    frame_count: int = 0
    start_time: float = 0.0
    motion_history: List[float] = []
    fall_confidence_history: List[float] = []
    seizure_confidence_history: List[float] = []
    fall_confirmation_frames: int = 0
    seizure_confirmation_frames: int = 0
    stats: Dict[str, Any] = {}
    previous_frame: Optional[np.ndarray] = None


class SystemComponents(NamedTuple):
    """Immutable system components"""
    config: Dict[str, Any]
    camera: Any = None
    frame_buffer: Any = None
    running: Any = None
    camera_thread: Any = None
    yolo_model: Any = None
    fall_detector: Any = None
    fall_predictor: Any = None
    seizure_detector: Any = None
    seizure_predictor: Any = None
    alert_manager: Any = None


class DisplayConfig(NamedTuple):
    """Immutable display configuration"""
    show_original: bool = True
    show_processed: bool = True
    display_enabled: bool = True
    frame_skip: int = 1


# Pure functions for system initialization
def create_initial_state(start_time: Optional[float] = None) -> MonitoringState:
    """Create initial monitoring state"""
    if start_time is None:
        start_time = time.time()
    
    initial_stats = {
        'frames_processed': 0,
        'detections': 0,
        'fall_alerts': 0,
        'seizure_alerts': 0,
        'motion_alerts': 0,
        'avg_fps': 0.0,
        'avg_processing_time': 0.0
    }
    
    return MonitoringState(
        frame_count=0,
        start_time=start_time,
        motion_history=[],
        fall_confidence_history=[],
        seizure_confidence_history=[],
        fall_confirmation_frames=0,
        seizure_confirmation_frames=0,
        stats=initial_stats,
        previous_frame=None
    )


def create_display_config(config: Dict[str, Any], args: Any = None) -> DisplayConfig:
    """Create display configuration from config and args"""
    display_conf = config.get('display', {})
    performance_conf = config.get('performance', {})
    
    show_original = display_conf.get('show_original', True)
    show_processed = display_conf.get('show_processed', True)
    frame_skip = performance_conf.get('frame_skip', 1)
    
    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'no_display') and args.no_display:
            show_original = False
            show_processed = False
        elif hasattr(args, 'original_only') and args.original_only:
            show_original = True
            show_processed = False
        elif hasattr(args, 'processed_only') and args.processed_only:
            show_original = False
            show_processed = True
        
        if hasattr(args, 'high_fps') and args.high_fps:
            frame_skip = 2
        elif hasattr(args, 'frame_skip') and args.frame_skip > 1:
            frame_skip = args.frame_skip
    
    display_enabled = show_original or show_processed
    
    return DisplayConfig(
        show_original=show_original,
        show_processed=show_processed,
        display_enabled=display_enabled,
        frame_skip=frame_skip
    )


def initialize_models(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Initialize detection models - pure function"""
    models: Dict[str, Any] = {
        'yolo_model': None,
        'fall_detector': None,
        'fall_predictor': None,
        'seizure_detector': None,
        'seizure_predictor': None
    }
    
    try:
        # Initialize YOLO model
        try:
            from ultralytics import YOLO
            model_path = config.get('detection', {}).get('yolo_model', 'yolov8s.pt')
            models['yolo_model'] = YOLO(model_path)
            logging.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            return False, models
        
        # Initialize fall detection
        fall_detector, fall_predictor = initialize_fall_detector(
            config.get('detection', {}).get('fall_threshold', 0.7)
        )
        models['fall_detector'] = fall_detector
        models['fall_predictor'] = fall_predictor
        logging.info("Fall detection initialized")
        
        # Initialize seizure detection
        seizure_detector, seizure_predictor = initialize_seizure_detector(
            config.get('detection', {}).get('seizure_threshold', 0.7)
        )
        models['seizure_detector'] = seizure_detector
        models['seizure_predictor'] = seizure_predictor
        logging.info("Seizure detection initialized")
        
        return True, models
        
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        return False, models


def initialize_camera_system(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Initialize camera system - pure function"""
    camera_components = {
        'camera': None,
        'frame_buffer': None,
        'running': None,
        'camera_thread': None
    }
    
    try:
        camera_config = config.get('camera', {})
        
        # Initialize camera
        camera, status_info = initialize_camera(camera_config)
        
        if not status_info['connected']:
            logging.error(f"Camera initialization failed: {status_info['error']}")
            return False, camera_components
        
        # Initialize frame buffer and threading
        frame_buffer = FrameBuffer(max_size=2)
        running = threading.Event()
        running.set()
        
        # Start camera thread
        fps_limit = camera_config.get('fps', 15)
        if camera is not None:
            camera_thread = start_camera_thread(camera, frame_buffer, running, fps_limit)
            
            camera_components = {
                'camera': camera,
                'frame_buffer': frame_buffer,
                'running': running,
                'camera_thread': camera_thread
            }
        
        logging.info(f"Camera system initialized: {status_info['resolution']} @ {status_info['fps']} FPS")
        return True, camera_components
        
    except Exception as e:
        logging.error(f"Camera system initialization failed: {str(e)}")
        return False, camera_components


def create_system_components(config: Dict[str, Any]) -> SystemComponents:
    """Create system components with initialized models and camera - pure function"""
    # Initialize models
    models_success, models = initialize_models(config)
    if not models_success:
        raise RuntimeError("Failed to initialize detection models")
    
    # Initialize camera
    camera_success, camera_components = initialize_camera_system(config)
    if not camera_success:
        raise RuntimeError("Failed to initialize camera system")
    
    # Initialize alert manager
    alert_manager = AlertManager(config.get('alerts', {}))
    
    return SystemComponents(
        config=config,
        camera=camera_components['camera'],
        frame_buffer=camera_components['frame_buffer'],
        running=camera_components['running'],
        camera_thread=camera_components['camera_thread'],
        yolo_model=models['yolo_model'],
        fall_detector=models['fall_detector'],
        fall_predictor=models['fall_predictor'],
        seizure_detector=models['seizure_detector'],
        seizure_predictor=models['seizure_predictor'],
        alert_manager=alert_manager
    )


# Pure processing functions
def process_frame(frame: np.ndarray, 
                 components: SystemComponents,
                 state: MonitoringState) -> Tuple[Dict[str, Any], MonitoringState]:
    """Process a single frame through the detection pipeline - pure function"""
    processing_start = time.time()
    
    result = {
        'detections': [],
        'motion_level': 0.0,
        'fall_detected': False,
        'seizure_detected': False,
        'alerts': [],
        'processing_time': 0.0
    }
    
    try:
        # Object detection
        detections = detect_objects_yolo(
            frame, components.yolo_model, 
            confidence_threshold=components.config.get('detection', {}).get('confidence_threshold', 0.5)
        )
        result['detections'] = detections
        
        # Find largest person
        person_detection = find_largest_person(detections, person_class_id=0)
        
        # Create new state with updated motion history (immutable updates)
        new_motion_history = list(state.motion_history)
        new_fall_history = list(state.fall_confidence_history)
        new_seizure_history = list(state.seizure_confidence_history)
        new_fall_frames = state.fall_confirmation_frames
        new_seizure_frames = state.seizure_confirmation_frames
        new_stats = dict(state.stats)
        
        motion_level = 0.0
        if person_detection:
            person_bbox = person_detection['bbox']
            
            # Calculate motion level
            if state.previous_frame is not None and person_detection:
                # Simple motion calculation using optical flow or frame difference
                gray_prev = cv2.cvtColor(state.previous_frame, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(gray_prev, gray_curr)
                
                # Focus on person region
                x1, y1, x2, y2 = map(int, person_bbox)
                person_diff = diff[max(0, y1):min(diff.shape[0], y2), 
                                 max(0, x1):min(diff.shape[1], x2)]
                
                # Calculate motion level as normalized mean difference
                if person_diff.size > 0:
                    motion_level = float(person_diff.mean()) / 255.0
                result['motion_level'] = motion_level
                
                # Update motion history (immutable)
                new_motion_history.append(motion_level)
                if len(new_motion_history) > 30:  # Keep last 30 frames
                    new_motion_history = new_motion_history[-30:]
            
            # Fall detection
            if components.fall_detector is not None:
                # Get person detections for fall detection
                person_detections = [{'bbox': person_bbox}] if person_bbox is not None else []
                
                fall_result = process_fall_detection(
                    components.fall_detector, frame, person_detections,
                    new_fall_history, new_fall_frames, 
                    motion_level, components.config.get('detection', {})
                )
                
                result['fall_detected'] = fall_result['fall_detected']
                new_fall_history = fall_result['confidence_history']
                new_fall_frames = fall_result['confirmation_frames']
                
                # Create fall alert if detected
                if fall_result['fall_detected'] and components.alert_manager:
                    alert_message = format_alert_message('fall', fall_result['smoothed_confidence'])
                    alert = components.alert_manager.add_alert(
                        'fall', alert_message, fall_result['smoothed_confidence'],
                        {'bbox': person_bbox, 'motion_level': motion_level}
                    )
                    if alert:
                        result['alerts'].append(alert)
                        new_stats['fall_alerts'] += 1
            
            # Seizure detection
            if components.seizure_detector is not None:
                seizure_result = process_seizure_detection(
                    components.seizure_detector, components.seizure_predictor, frame, person_bbox,
                    new_seizure_history, new_seizure_frames,
                    motion_level, components.config.get('detection', {})
                )
                
                result['seizure_detected'] = seizure_result['seizure_detected']
                new_seizure_history = seizure_result['confidence_history']
                new_seizure_frames = seizure_result['confirmation_frames']
                
                # Create seizure alert if detected
                if seizure_result['seizure_detected'] and components.alert_manager:
                    alert_message = format_alert_message('seizure', seizure_result['smoothed_confidence'])
                    alert = components.alert_manager.add_alert(
                        'seizure', alert_message, seizure_result['smoothed_confidence'],
                        {'bbox': person_bbox, 'motion_level': motion_level}
                    )
                    if alert:
                        result['alerts'].append(alert)
                        new_stats['seizure_alerts'] += 1
            
            # Motion alert for very high motion
            if motion_level > 0.9 and components.alert_manager:
                alert_message = format_alert_message('motion', motion_level, {'motion_level': motion_level})
                alert = components.alert_manager.add_alert(
                    'motion', alert_message, motion_level,
                    {'bbox': person_bbox, 'motion_level': motion_level}
                )
                if alert:
                    result['alerts'].append(alert)
                    new_stats['motion_alerts'] += 1
        
        # Create new state with updated values (immutable)
        new_state = MonitoringState(
            frame_count=state.frame_count,
            start_time=state.start_time,
            motion_history=new_motion_history,
            fall_confidence_history=new_fall_history,
            seizure_confidence_history=new_seizure_history,
            fall_confirmation_frames=new_fall_frames,
            seizure_confirmation_frames=new_seizure_frames,
            stats=new_stats,
            previous_frame=frame.copy()
        )
        
    except Exception as e:
        logging.error(f"Frame processing error: {str(e)}")
        result['error'] = str(e)
        new_state = state  # Return unchanged state on error
    
    result['processing_time'] = time.time() - processing_start
    return result, new_state


def create_display_frame(frame: np.ndarray, 
                        processing_result: Dict[str, Any], 
                        state: MonitoringState) -> np.ndarray:
    """Create display frame with visualizations - pure function"""
    display_frame = frame.copy()
    
    try:
        # Draw person detection
        for detection in processing_result.get('detections', []):
            if detection['class_id'] == 0:  # Person class
                display_frame = draw_bounding_box(
                    display_frame, detection['bbox'],
                    color=(0, 255, 0), thickness=2,
                    label="Person", confidence=detection['confidence']
                )
        
        # Draw motion indicators
        motion_level = processing_result.get('motion_level', 0.0)
        display_frame = draw_motion_indicators(display_frame, motion_level, "top-right")
        
        # Draw alerts
        alerts = processing_result.get('alerts', [])
        if alerts:
            display_frame = draw_detection_alerts(display_frame, alerts)
        
        # Draw statistics
        current_stats = get_current_statistics(state)
        display_frame = draw_statistics_overlay(display_frame, current_stats)
        
    except Exception as e:
        logging.error(f"Display creation error: {str(e)}")
    
    return display_frame


def display_windows(original_frame: np.ndarray, 
                   processed_frame: np.ndarray, 
                   display_config: DisplayConfig,
                   frame_count: int) -> bool:
    """Display frames in separate windows - pure function (except for cv2 side effects)"""
    if not display_config.display_enabled:
        return True
    
    try:
        # Window 1: Original RTSP feed
        if display_config.show_original:
            # Add basic info overlay to original frame
            info_frame = original_frame.copy()
            
            # Add timestamp and frame counter
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(info_frame, f"RTSP Feed - Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, timestamp, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Healthcare Monitor - Original Feed", info_frame)
        
        # Window 2: Processed feed with detections
        if display_config.show_processed:
            # Add title to processed frame
            title_frame = processed_frame.copy()
            cv2.putText(title_frame, "Healthcare Monitor - Detection Results", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Healthcare Monitor - Processed Feed", title_frame)
        
        # Handle window events
        key = cv2.waitKey(1) & 0xFF
        
        # Keyboard controls
        if key == ord('q'):
            logging.info("Quit requested by user")
            return False  # Signal to stop monitoring
        elif key == ord('h'):
            print_help()
            
        return True
        
    except Exception as e:
        logging.error(f"Display window error: {str(e)}")
        return True


def print_help():
    """Print keyboard controls help - pure function"""
    help_text = """
========== Healthcare Monitor Controls ==========
q - Quit application
h - Show this help message
===============================================
    """
    print(help_text)
    logging.info("Control help displayed")


def setup_display_windows(display_config: DisplayConfig):
    """Setup OpenCV display windows - impure function"""
    if not display_config.display_enabled:
        return
        
    try:
        if display_config.show_original:
            cv2.namedWindow("Healthcare Monitor - Original Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Healthcare Monitor - Original Feed", 640, 480)
            
        if display_config.show_processed:
            cv2.namedWindow("Healthcare Monitor - Processed Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Healthcare Monitor - Processed Feed", 640, 480)
        
        # Position windows side by side
        if display_config.show_original and display_config.show_processed:
            cv2.moveWindow("Healthcare Monitor - Original Feed", 50, 50)
            cv2.moveWindow("Healthcare Monitor - Processed Feed", 720, 50)
        
        logging.info("Display windows initialized")
        print_help()
        
    except Exception as e:
        logging.error(f"Display window setup error: {str(e)}")


def cleanup_display():
    """Clean up display windows - impure function"""
    try:
        cv2.destroyAllWindows()
        logging.info("Display windows closed")
    except Exception as e:
        logging.error(f"Display cleanup error: {str(e)}")


def get_current_statistics(state: MonitoringState) -> Dict[str, Any]:
    """Get current monitoring statistics - pure function"""
    runtime_stats = calculate_runtime_stats(state.start_time, state.frame_count)
    
    current_stats = {
        'Runtime': f"{runtime_stats['runtime_minutes']:.1f}m",
        'FPS': f"{runtime_stats['avg_fps']:.1f}",
        'Frames': state.frame_count,
        'Fall Alerts': state.stats['fall_alerts'],
        'Seizure Alerts': state.stats['seizure_alerts'],
        'Motion Alerts': state.stats['motion_alerts']
    }
    
    return current_stats


def update_frame_count(state: MonitoringState) -> MonitoringState:
    """Update frame count - pure function"""
    return state._replace(
        frame_count=state.frame_count + 1,
        stats={**state.stats, 'frames_processed': state.frame_count + 1}
    )


def run_monitoring_loop(components: SystemComponents, 
                       display_config: DisplayConfig,
                       initial_state: MonitoringState) -> MonitoringState:
    """Main monitoring loop - functional approach"""
    logging.info("Starting healthcare monitoring...")
    
    # Setup display windows
    setup_display_windows(display_config)
    
    state = initial_state
    
    try:
        while components.running and components.running.is_set():
            # Get frame from buffer
            frame = components.frame_buffer.get_latest_frame() if components.frame_buffer else None
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Update frame count first
            state = update_frame_count(state)
            
            # Skip frames for performance if configured
            if display_config.frame_skip > 1 and state.frame_count % display_config.frame_skip != 0:
                # Still show the frame but skip processing
                continue_monitoring = display_windows(frame, frame, display_config, state.frame_count)
                if not continue_monitoring:
                    break
                time.sleep(0.005)
                continue
            
            # Process frame (only every Nth frame if frame_skip > 1)
            processing_result, new_state = process_frame(frame, components, state)
            state = new_state
            
            # Create display frame with overlays
            display_frame = create_display_frame(frame, processing_result, state)
            
            # Show both windows if enabled
            continue_monitoring = display_windows(frame, display_frame, display_config, state.frame_count)
            if not continue_monitoring:
                break
            
            # Log processing results for significant events
            if processing_result.get('alerts'):
                logging.info(f"Frame {state.frame_count}: {len(processing_result['alerts'])} alerts generated")
            
            # Print real-time stats every 300 frames (~10 seconds at 30 FPS)
            if state.frame_count % 300 == 0:
                current_stats = get_current_statistics(state)
                logging.info(f"Current stats: {current_stats}")
            
            # Clean up old alerts periodically
            if state.frame_count % 600 == 0 and components.alert_manager:  # Every 600 frames (~20 seconds at 30 FPS)
                cleared_count = components.alert_manager.clear_old_alerts()
                active_alerts = components.alert_manager.get_active_alerts()
                logging.info(f"Cleared old alerts, {len(active_alerts)} active alerts remaining")
            
            # Reduce pause time for higher FPS
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Monitoring loop error: {str(e)}")
    finally:
        cleanup_resources(components)
    
    return state


def cleanup_resources(components: SystemComponents):
    """Clean up system resources - impure function"""
    logging.info("Cleaning up resources...")
    
    # Cleanup display windows
    cleanup_display()
    
    # Stop camera
    if components.camera is not None:
        release_camera(components.camera, components.camera_thread, components.running)
    
    logging.info("Healthcare monitor shutdown complete")


def main():
    """Main entry point - functional approach"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Healthcare Monitor - Pure Functional Architecture')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-display', action='store_true', help='Disable display windows (headless mode)')
    parser.add_argument('--original-only', action='store_true', help='Show only original RTSP feed')
    parser.add_argument('--processed-only', action='store_true', help='Show only processed detection feed')
    parser.add_argument('--high-fps', action='store_true', help='Enable high FPS mode (skip some processing)')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame (1=process all, 2=skip every other)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(config.get('logging', {}))
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug logging enabled")
        
        # Create display configuration
        display_config = create_display_config(config, args)
        
        # Log display mode
        if not display_config.display_enabled:
            logging.info("Running in headless mode (no display)")
        elif display_config.show_original and not display_config.show_processed:
            logging.info("Showing original feed only")
        elif display_config.show_processed and not display_config.show_original:
            logging.info("Showing processed feed only")
        
        # Log performance settings
        if display_config.frame_skip > 1:
            logging.info(f"Frame skip set to {display_config.frame_skip} (processing every {display_config.frame_skip}th frame)")
        
        # Create system components
        components = create_system_components(config)
        
        # Create initial state
        initial_state = create_initial_state()
        
        # Run monitoring loop
        final_state = run_monitoring_loop(components, display_config, initial_state)
        
        # Log final statistics
        final_stats = get_current_statistics(final_state)
        logging.info(f"Final statistics: {final_stats}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
