"""
Vision AI System - Main Entry Point
"""
import cv2
import time
from vision_ai_system.container import container

def main():
    # Get use cases from container
    start_monitoring = container.get_start_monitoring_use_case()
    process_frame = container.get_process_frame_use_case()
    stop_monitoring = container.get_stop_monitoring_use_case()
    
    # Camera config
    camera_configs = [
        {
            "camera_id": "main_camera",
            "source": 0,
            "fps": 30
        }
    ]
    
    print("Starting camera monitoring...")
    
    # Start monitoring
    result = start_monitoring.execute(camera_configs)
    if not result['success']:
        print(f"Failed to start: {result.get('error')}")
        return
    
    print("Camera monitoring started. Processing frames...")
    
    try:
        # Process frames
        frame_count = 0
        while True:
            frame_result = process_frame.execute_single_camera("main_camera")
            
            if frame_result['success'] and frame_result['frame_ready']:
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Processed {frame_count} frames")
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop monitoring
        stop_result = stop_monitoring.execute()
        print(f"Stopped: {stop_result['message']}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()