"""
Multi-Camera Healthcare Monitoring System
Uses 2 cameras and selects best frame for detection
"""

import cv2
import time
import threading
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.camera_network_coordinator import get_multi_camera_manager
from service.video_processing_service import VideoProcessingService
from service.fall_detection_service import FallDetectionService
from service.seizure_detection_service import SeizureDetectionService
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
from seizure_detection.seizure_predictor import SeizurePredictor

def main():
    """Main function for multi-camera healthcare system"""
    
    print("🎥 Multi-Camera Healthcare Monitoring System")
    print("="*60)
    print("📹 Initializing 2-camera system with best frame selection...")
    
    try:
        # Initialize multi-camera manager
        camera_manager = get_multi_camera_manager()
        
        # Wait a moment for cameras to start capturing
        time.sleep(2)
        
        # Check camera status
        stats = camera_manager.get_stats()
        connected_cameras = stats['cameras_connected']
        
        if connected_cameras == 0:
            print("❌ No cameras connected! Please check camera configurations.")
            return
        elif connected_cameras == 1:
            print("⚠️ Only 1 camera connected. System will work but redundancy is reduced.")
        else:
            print(f"✅ {connected_cameras} cameras connected successfully!")
        
        # Initialize processing services
        print("🔧 Initializing healthcare services...")
        video_processor = VideoProcessingService(120)
        fall_detector = FallDetectionService()
        seizure_detector = SeizureDetectionService()
        seizure_predictor = SeizurePredictor(temporal_window=25, alert_threshold=0.7, warning_threshold=0.5)
        
        # Create alerts folder
        alerts_folder = "examples/data/saved_frames/alerts"
        Path(alerts_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize healthcare pipeline with multi-camera support
        class MultiCameraHealthcarePipeline(AdvancedHealthcarePipeline):
            """Extended pipeline for multi-camera support"""
            
            def __init__(self, camera_manager, *args, **kwargs):
                # Initialize with dummy camera (we'll override get_frame)
                super().__init__(None, *args, **kwargs)
                self.camera_manager = camera_manager
                self.multi_camera_stats = {
                    'frames_from_camera_1': 0,
                    'frames_from_camera_2': 0,
                    'total_selections': 0
                }
            
            def get_frame(self):
                """Get best frame from multi-camera manager"""
                best_frame = self.camera_manager.get_best_frame()
                if best_frame:
                    # Update selection statistics
                    camera_id = best_frame.camera_id
                    if 'camera_01' in camera_id or '201' in camera_id:
                        self.multi_camera_stats['frames_from_camera_1'] += 1
                    elif 'camera_02' in camera_id or '202' in camera_id:
                        self.multi_camera_stats['frames_from_camera_2'] += 1
                    
                    self.multi_camera_stats['total_selections'] += 1
                    return best_frame.frame
                return None
            
            def get_multi_camera_stats(self):
                """Get multi-camera specific statistics"""
                base_stats = self.camera_manager.get_stats()
                base_stats.update(self.multi_camera_stats)
                return base_stats
        
        # Create multi-camera pipeline
        pipeline = MultiCameraHealthcarePipeline(
            camera_manager=camera_manager,
            video_processor=video_processor,
            fall_detector=fall_detector,
            seizure_detector=seizure_detector,
            seizure_predictor=seizure_predictor,
            alerts_folder=alerts_folder
        )
        
        print("✅ Multi-camera healthcare pipeline initialized!")
        print("🎯 System will automatically select the best camera frame for detection")
        print("📊 Frame selection based on: Quality + Motion + Brightness + Sharpness")
        print("🚨 All healthcare features enabled: Fall + Seizure detection")
        print("📱 Emergency notifications: ACTIVE")
        print()
        print("Controls:")
        print("  'q' - Quit system")
        print("  's' - Show statistics")
        print("  'c' - Show camera selection stats")
        print("="*60)
        
        # Main processing loop
        frame_count = 0
        last_stats_time = time.time()
        
        while True:
            # Get best frame from multi-camera system
            frame = pipeline.get_frame()
            
            if frame is None:
                print("⚠️ No frame available from any camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process frame through healthcare pipeline
            result = pipeline.process_frame(frame)
            
            # Display frames
            if result:
                # Show normal camera view
                normal_window = result.get("normal_window")
                if normal_window is not None:
                    cv2.imshow("Multi-Camera Healthcare Monitor", normal_window)
                
                # Show AI processing view
                ai_window = result.get("ai_window")
                if ai_window is not None:
                    cv2.imshow("AI Detection View", ai_window)
                
                # Show detection results
                detection_result = result.get("detection_result", {})
                if detection_result.get('fall_detected'):
                    print(f"🚨 FALL DETECTED! Confidence: {detection_result.get('fall_confidence', 0):.2f}")
                
                if detection_result.get('seizure_detected'):
                    print(f"⚡ SEIZURE DETECTED! Confidence: {detection_result.get('seizure_confidence', 0):.2f}")
            
            # Show periodic statistics
            if time.time() - last_stats_time > 10:  # Every 10 seconds
                multi_stats = pipeline.get_multi_camera_stats()
                print(f"📊 Processed: {frame_count} frames | "
                      f"Cam1: {multi_stats.get('frames_from_camera_1', 0)} | "
                      f"Cam2: {multi_stats.get('frames_from_camera_2', 0)} | "
                      f"Connected: {multi_stats.get('cameras_connected', 0)}")
                last_stats_time = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping multi-camera system...")
                break
            elif key == ord('s'):
                # Show detailed statistics
                stats = pipeline.get_stats()
                multi_stats = pipeline.get_multi_camera_stats()
                print("\n📊 DETAILED STATISTICS:")
                print(f"Total frames processed: {frame_count}")
                print(f"Fall detections: {stats.get('fall_detections', 0)}")
                print(f"Seizure detections: {stats.get('seizure_detections', 0)}")
                print(f"Cameras connected: {multi_stats.get('cameras_connected', 0)}")
                print(f"Camera selection quality scores: {multi_stats.get('quality_scores_per_camera', {})}")
                print()
            elif key == ord('c'):
                # Show camera selection statistics
                multi_stats = pipeline.get_multi_camera_stats()
                total = multi_stats.get('total_selections', 1)
                cam1_pct = (multi_stats.get('frames_from_camera_1', 0) / total) * 100
                cam2_pct = (multi_stats.get('frames_from_camera_2', 0) / total) * 100
                
                print("\n🎥 CAMERA SELECTION STATISTICS:")
                print(f"Camera 1 selected: {multi_stats.get('frames_from_camera_1', 0)} times ({cam1_pct:.1f}%)")
                print(f"Camera 2 selected: {multi_stats.get('frames_from_camera_2', 0)} times ({cam2_pct:.1f}%)")
                print(f"Total selections: {total}")
                print(f"Selection criteria working: {'✅' if min(cam1_pct, cam2_pct) > 10 else '⚠️'}")
                print()
        
        # Cleanup
        cv2.destroyAllWindows()
        camera_manager.disconnect_all()
        print("✅ Multi-camera healthcare system stopped cleanly")
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        cv2.destroyAllWindows()
        try:
            camera_manager = get_multi_camera_manager()
            camera_manager.disconnect_all()
        except:
            pass

if __name__ == "__main__":
    main()
