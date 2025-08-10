import cv2
import time
from core.config import get_camera_config, get_fall_detection_config
from camera.controls import initialize_camera
from logic.statistics import (
    create_statistics_state, update_frame_statistics, update_fall_statistics,
    reset_statistics, print_fall_detection_alert, print_final_statistics
)
from logic.detection_controller import (
    initialize_fall_detection_system, process_frame_for_fall_detection,
    check_user_input, handle_frame_processing_error
)
from logic.display_controller import (
    create_display_system, render_display_frames, cleanup_display_system,
    print_system_instructions
)

def main():
    # Initialize camera
    camera_config = get_camera_config()
    cap, status = initialize_camera(camera_config)
    if not status['connected']:
        print(f"Camera error: {status['error']}")
        return
    
    # Initialize fall detection system
    fall_config = get_fall_detection_config()
    fall_system = initialize_fall_detection_system(
        fall_config['model_path'], 
        fall_config['threshold']
    )
    
    if not fall_system['initialized']:
        print(f"Fall detection initialization failed: {fall_system['error']}")
        return
    
    # Initialize display system
    display_system = create_display_system()
    
    # Initialize statistics
    stats_state = create_statistics_state()
    
    # Print instructions
    print_system_instructions()
    
    # Main processing loop
    while True:
        # Check camera connection
        if cap is None:
            break
            
        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        # Update frame statistics
        stats_state = update_frame_statistics(stats_state)
        
        # Process frame for fall detection
        fall_result, keypoints = process_frame_for_fall_detection(frame, fall_system)
        
        # Handle detection result
        fall_detected = fall_result.get('fall_detected', False)
        confidence = fall_result.get('confidence', 0.0)
        
        # Debug: In ra thông tin detection
        if stats_state['frame_count'] % 30 == 0:  # In mỗi 30 frame
            fall_ratio = fall_result.get('fall_ratio', 0.0)
            keypoints_shape = keypoints.shape if keypoints is not None else None
            print(f"Frame {stats_state['frame_count']}: Fall={fall_detected}, Confidence={confidence:.3f}, Ratio={fall_ratio:.3f}")
            print(f"Keypoints shape: {keypoints_shape}")
        
        if fall_detected:
            old_count = stats_state['fall_count']
            stats_state = update_fall_statistics(stats_state, confidence)
            # Chỉ in nếu thực sự tăng count (không bị cooldown)
            if stats_state['fall_count'] > old_count:
                print_fall_detection_alert(stats_state)
        
        # Handle any processing errors
        if fall_result.get('error'):
            if not handle_frame_processing_error(frame, fall_result['error']):
                break
        
        # Render display
        render_display_frames(frame, keypoints, stats_state, fall_detected, display_system)
        
        # Check user input
        user_command = check_user_input()
        
        if user_command == 'quit':
            break
        elif user_command == 'reset':
            stats_state = reset_statistics(stats_state)
            print("Statistics reset!")
    
    # Cleanup and final statistics
    if cap is not None:
        cap.release()
    cleanup_display_system()
    
    # Print final statistics
    print_final_statistics(stats_state)

if __name__ == "__main__":
    main()
