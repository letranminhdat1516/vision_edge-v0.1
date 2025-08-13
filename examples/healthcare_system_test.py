"""
Test Full Healthcare System with RTSP Camera
- Connects to real RTSP camera
- Runs fall and seizure detection
- Publishes events to Supabase
- Simulates mobile app receiving notifications
"""

import cv2
import sys
import os
import threading
import time
import signal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_mobile_app():
    """Run mobile app simulation in separate thread"""
    try:
        from examples.mobile_healthcare_app import MobileHealthcareApp
        print("📱 Starting mobile app simulation...")
        app = MobileHealthcareApp()
        app.start()
    except Exception as e:
        print(f"❌ Error in mobile app: {e}")

def run_healthcare_pipeline():
    """Run main healthcare pipeline with RTSP camera"""
    try:
        print("🏥 Starting Healthcare Pipeline...")
        print("📹 Connecting to RTSP camera...")
        
        # Import services
        from service.camera_service import CameraService
        from service.video_processing_service import VideoProcessingService
        from service.fall_detection_service import FallDetectionService
        from service.seizure_detection_service import SeizureDetectionService
        from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
        from seizure_detection.seizure_predictor import SeizurePredictor
        
        # Camera configuration for RTSP
        camera_config = {
            'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
            'buffer_size': 1,
            'fps': 15,
            'resolution': (640, 480),
            'auto_reconnect': True
        }
        
        # Initialize services
        camera = CameraService(camera_config)
        camera.connect()
        
        video_processor = VideoProcessingService(120)
        fall_detector = FallDetectionService()
        seizure_detector = SeizureDetectionService()
        seizure_predictor = SeizurePredictor(temporal_window=25, alert_threshold=0.7, warning_threshold=0.5)
        
        alerts_folder = "examples/data/saved_frames/alerts"
        pipeline = AdvancedHealthcarePipeline(
            camera, video_processor, fall_detector, 
            seizure_detector, seizure_predictor, alerts_folder
        )
        
        print("✅ Healthcare Pipeline initialized")
        print("🔥 Real-time processing started...")
        print("🎯 Fall detection: ACTIVE")
        print("⚡ Seizure detection: ACTIVE") 
        print("📡 Supabase events: ACTIVE")
        print("📱 Mobile notifications: ACTIVE")
        print("🛑 Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("⚠️ No frame received, retrying...")
                time.sleep(1)
                continue
                
            # Process frame through pipeline
            result = pipeline.process_frame(frame)
            
            # Display results
            cv2.imshow("🏥 Healthcare Monitor - Normal View", result["normal_window"])
            
            # Create analysis view with detection overlays
            detection_result = result["detection_result"]
            person_detections = result["person_detections"]
            analysis_view = pipeline.visualize_dual_detection(frame, detection_result, person_detections)
            analysis_view = pipeline.draw_statistics_overlay(analysis_view, pipeline.stats)
            
            cv2.imshow("🔬 Healthcare Analysis - Detection View", analysis_view)
            
            # Show statistics every 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Processing: {frame_count} frames | FPS: {fps:.1f} | Falls: {pipeline.stats['fall_detections']} | Seizures: {pipeline.stats['seizure_detections']}")
            
            # Handle events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping healthcare pipeline...")
                break
                
        # Cleanup
        camera.disconnect()
        cv2.destroyAllWindows()
        print("✅ Healthcare pipeline stopped")
        
    except Exception as e:
        print(f"❌ Error in healthcare pipeline: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run full healthcare system"""
    print("🚀 VISION EDGE HEALTHCARE SYSTEM")
    print("=" * 60)
    print("🏥 Real-time Healthcare Monitoring")
    print("📡 Supabase Realtime Integration") 
    print("📱 Mobile App Notifications")
    print("🎥 RTSP Camera Processing")
    print("=" * 60)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down healthcare system...")
        os._exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start mobile app in separate thread
        mobile_thread = threading.Thread(target=run_mobile_app, daemon=True)
        mobile_thread.start()
        
        # Give mobile app time to start
        time.sleep(2)
        
        # Run main healthcare pipeline
        run_healthcare_pipeline()
        
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Healthcare system terminated")

if __name__ == "__main__":
    main()
