#!/usr/bin/env python3
"""
Single Video Tester - Test từng video riêng lẻ để điều chỉnh tham số
Usage: python test_single_video.py <video_number>
Example: python test_single_video.py 1
"""

import os
import sys
import cv2
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import video camera service
from video_camera_service import VideoCameraService

# Import tất cả services từ main.py
from service.video_processing_service import VideoProcessingService
from service.fall_detection_service import FallDetectionService
from service.seizure_detection_service import SeizureDetectionService
from seizure_detection.seizure_predictor import SeizurePredictor
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import intelligent action generation
try:
    from service.ai_vision_description_service import get_professional_caption_pipeline
    INTELLIGENT_ACTIONS_AVAILABLE = True
except ImportError:
    INTELLIGENT_ACTIONS_AVAILABLE = False


class SingleVideoTester:
    """Test single video với khả năng điều chỉnh tham số"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        self.output_base = self.script_dir / "test_results"
        self.user_id = os.getenv('DEFAULT_USER_ID', 'test_user_001')
        
        # Tạo output folders
        self.output_folders = {
            'alerts': self.output_base / 'alerts',
            'keypoints': self.output_base / 'keypoints',
            'reports': self.output_base / 'reports'
        }
        
        for folder in self.output_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def find_video(self, video_number: int) -> Path:
        """Tìm video theo số"""
        # Thử cả .mp4 và .MP4
        video_path_lower = self.resource_folder / f"{video_number}.mp4"
        video_path_upper = self.resource_folder / f"{video_number}.MP4"
        
        if video_path_lower.exists():
            return video_path_lower
        elif video_path_upper.exists():
            return video_path_upper
        else:
            raise FileNotFoundError(f"Video {video_number} not found in {self.resource_folder}")
    
    def test_video(self, video_number: int, show_display: bool = True, save_keypoints: bool = True):
        """
        Test 1 video với các tùy chọn
        
        Args:
            video_number: Số thứ tự video (1, 2, 3, ...)
            show_display: Hiển thị window realtime (True/False)
            save_keypoints: Lưu keypoint images (True/False)
        """
        video_path = self.find_video(video_number)
        video_name = video_path.stem
        
        print("\n" + "="*100)
        print(f"🎬 TESTING VIDEO #{video_number}")
        print("="*100)
        print(f"📹 File: {video_path.name}")
        print(f"📂 Path: {video_path}")
        print(f"👁️  Display: {'ON' if show_display else 'OFF'}")
        print(f"💾 Save Keypoints: {'ON' if save_keypoints else 'OFF'}")
        print("="*100 + "\n")
        
        # ==================== SETUP ====================
        
        camera_config = {
            'video_path': str(video_path),
            'buffer_size': 1,
            'fps': 30,
            'resolution': None,  # Giữ nguyên resolution gốc của video, không resize
            'camera_id': f"test_video_{video_number}",
            'camera_name': video_name,
            'loop': False
        }
        
        alerts_folder = str(self.output_folders['alerts'] / f"video_{video_number}")
        Path(alerts_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        print("🔧 Initializing services...")
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print(f"❌ Failed to load video")
            return None
        
        # IMPORTANT: Set buffer_size = 1 to process EVERY frame (no keyframe skipping)
        video_processor = VideoProcessingService(config=1)  # buffer_size = 1
        
        # ĐIỀU CHỈNH THRESHOLD TÉ NGÃ - Giảm threshold = tăng độ nhạy
        # Video 1 là té ngã đột ngột, cần rất nhạy
        fall_detector = FallDetectionService(confidence_threshold=0.05)  # Giảm từ 0.15 -> 0.05 (cực kỳ nhạy)
        
        seizure_detector = SeizureDetectionService()
        
        # Điều chỉnh threshold cho co giật - TĂNG LÊN để giảm false positive
        seizure_predictor = SeizurePredictor(
            temporal_window=5,          # Tăng từ 3 -> 5 (cần nhiều frame hơn mới detect)
            alert_threshold=0.02,       # Tăng từ 0.005 -> 0.02 (ít nhạy hơn)
            warning_threshold=0.01      # Tăng từ 0.002 -> 0.01
        )
        
        print(f"🏥 Initializing Healthcare Pipeline...")
        pipeline = AdvancedHealthcarePipeline(
            camera=camera,
            video_processor=video_processor,
            fall_detector=fall_detector,
            seizure_detector=seizure_detector,
            seizure_predictor=seizure_predictor,
            alerts_folder=alerts_folder,
            camera_id=camera_config['camera_id'],
            user_id=self.user_id
        )
        
        # Initialize intelligent actions
        caption_pipeline = None
        if INTELLIGENT_ACTIONS_AVAILABLE:
            try:
                caption_pipeline = get_professional_caption_pipeline()
                print("🤖 Intelligent actions: ENABLED")
            except Exception as e:
                print(f"⚠️ Intelligent actions: DISABLED ({e})")
        
        print("\n✅ All systems ready!")
        print("="*100)
        
        # ==================== PROCESSING ====================
        
        print(f"\n🎥 Processing video #{video_number}...")
        if show_display:
            print("🎮 Controls:")
            print("   'q' = Quit")
            print("   's' = Show statistics")
            print("   'p' = Pause/Resume")
            print("   'SPACE' = Step frame (when paused)")
        print("="*100 + "\n")
        
        detected_events = []
        frame_count = 0
        start_time = time.time()
        paused = False
        
        while True:
            if not paused:
                frame = camera.get_frame()
                if frame is None:
                    print(f"\n✅ Video completed!")
                    break
                
                frame_count += 1
                
                # Process frame
                result = pipeline.process_frame(frame)
                detection_result = result["detection_result"]
                person_detections = result["person_detections"]
                
                # Save keypoints
                if save_keypoints and (frame_count % 30 == 0 or detection_result.get('alert_level') in ['critical', 'high', 'warning']):
                    try:
                        keypoint_folder = self.output_folders['keypoints'] / f"video_{video_number}"
                        keypoint_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Vẽ pose skeleton và detection results
                        keypoint_frame = pipeline.visualize_dual_detection(frame.copy(), detection_result, person_detections)
                        
                        # Vẽ thêm pose keypoints chi tiết từ YOLO Pose
                        if person_detections:
                            for person in person_detections:
                                if hasattr(person, 'keypoints') and person.keypoints is not None:
                                    # Vẽ skeleton connections
                                    keypoints = person.keypoints.xy[0].cpu().numpy() if hasattr(person.keypoints, 'xy') else []
                                    if len(keypoints) > 0:
                                        # COCO pose connections (17 keypoints)
                                        connections = [
                                            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                                            (5, 11), (6, 12), (11, 12),  # Torso
                                            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                                        ]
                                        
                                        # Vẽ skeleton lines
                                        for start, end in connections:
                                            if start < len(keypoints) and end < len(keypoints):
                                                pt1 = tuple(map(int, keypoints[start]))
                                                pt2 = tuple(map(int, keypoints[end]))
                                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                                    cv2.line(keypoint_frame, pt1, pt2, (0, 255, 255), 2)
                                        
                                        # Vẽ keypoint circles
                                        for i, kpt in enumerate(keypoints):
                                            x, y = int(kpt[0]), int(kpt[1])
                                            if x > 0 and y > 0:
                                                cv2.circle(keypoint_frame, (x, y), 4, (0, 0, 255), -1)
                                                cv2.putText(keypoint_frame, str(i), (x+5, y-5),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Add detailed frame info
                        info_y = 30
                        info_texts = [
                            f"Video #{video_number} | Frame: {frame_count}/{camera.total_frames}",
                            f"Persons: {len(person_detections)}",
                            f"Alert: {detection_result.get('alert_level', 'normal')}",
                        ]
                        
                        # Add confidence if detection exists
                        if detection_result.get('alert_level') in ['critical', 'high', 'warning']:
                            emergency_type = detection_result.get('emergency_type', 'unknown')
                            if 'fall' in emergency_type:
                                conf = detection_result.get('fall_confidence', 0)
                                info_texts.append(f"Fall Confidence: {conf:.2%}")
                            else:
                                conf = detection_result.get('seizure_confidence', 0)
                                info_texts.append(f"Seizure Confidence: {conf:.2%}")
                        
                        for text in info_texts:
                            cv2.putText(keypoint_frame, text, (10, info_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            info_y += 30
                        
                        keypoint_path = keypoint_folder / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(keypoint_path), keypoint_frame)
                    except Exception as e:
                        logger.error(f"Error saving keypoint: {e}")
                
                # Detect events
                if detection_result.get('alert_level') in ['critical', 'high', 'warning']:
                    emergency_type = detection_result.get('emergency_type', 'unknown')
                    confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
                    
                    # Find alert image
                    alert_image_path = None
                    try:
                        alerts_dir = Path(alerts_folder)
                        if alerts_dir.exists():
                            alert_files = sorted(alerts_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                            if alert_files:
                                alert_image_path = str(alert_files[0])
                    except Exception as e:
                        logger.error(f"Error finding alert image: {e}")
                    
                    # Generate captions
                    intelligent_action = "Standard alert"
                    vietnamese_caption = ""
                    
                    if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline and alert_image_path:
                        try:
                            action_result = caption_pipeline(alert_image_path)
                            if action_result:
                                intelligent_action = action_result.get('recommended_action', intelligent_action)
                                vietnamese_caption = action_result.get('vietnamese_caption', '')
                        except Exception as e:
                            logger.error(f"Error generating caption: {e}")
                    else:
                        if 'fall' in emergency_type:
                            intelligent_action = "Kiểm tra người bệnh ngay lập tức. Gọi hỗ trợ y tế nếu cần."
                            vietnamese_caption = "Phát hiện ngã - Cần hỗ trợ khẩn cấp"
                        else:
                            intelligent_action = "Quan sát người bệnh. Chuẩn bị hỗ trợ y tế."
                            vietnamese_caption = "Phát hiện hành vi bất thường - Cần theo dõi"
                    
                    event = {
                        'frame': frame_count,
                        'type': emergency_type,
                        'confidence': confidence,
                        'alert_level': detection_result.get('alert_level'),
                        'timestamp': time.time() - start_time,
                        'intelligent_action': intelligent_action,
                        'vietnamese_caption': vietnamese_caption,
                        'alert_image': alert_image_path
                    }
                    detected_events.append(event)
                    
                    print(f"\n{'='*100}")
                    print(f"🚨 ALERT #{len(detected_events)} DETECTED")
                    print(f"{'='*100}")
                    print(f"   Frame: {frame_count}/{camera.total_frames} ({camera.get_status()['progress']})")
                    print(f"   Type: {emergency_type}")
                    print(f"   Confidence: {confidence:.2%}")
                    print(f"   Level: {detection_result.get('alert_level')}")
                    print(f"   📝 Action: {intelligent_action}")
                    print(f"   🇻🇳 Caption: {vietnamese_caption}")
                    if alert_image_path:
                        print(f"   📸 Image: {alert_image_path}")
                    print(f"{'='*100}\n")
                
                # Show display
                if show_display:
                    # Normal view
                    cv2.imshow(f"Video #{video_number} - Normal View", result["normal_window"])
                    
                    # Analysis view
                    analysis_view = pipeline.visualize_dual_detection(frame, detection_result, person_detections)
                    analysis_view = pipeline.draw_statistics_overlay(analysis_view, pipeline.stats)
                    
                    # Add progress
                    camera_status = camera.get_status()
                    progress_text = f"Progress: {camera_status['progress']} ({frame_count}/{camera.total_frames})"
                    cv2.putText(analysis_view, progress_text, (10, analysis_view.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Pause indicator
                    if paused:
                        cv2.putText(analysis_view, "PAUSED - Press 'p' to resume, SPACE for next frame", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imshow(f"Video #{video_number} - Analysis View", analysis_view)
                
                # Progress log
                if frame_count % 100 == 0:
                    camera_status = camera.get_status()
                    print(f"📊 Progress: {camera_status['progress']} - Frame {frame_count}/{camera.total_frames} - Events: {len(detected_events)}")
            
            # Handle keyboard
            if show_display:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('s'):
                    print(f"\n📊 Current Statistics:")
                    for k, v in pipeline.stats.items():
                        print(f"   {k}: {v}")
                    print()
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️ PAUSED' if paused else '▶️ RESUMED'}")
                elif key == ord(' ') and paused:
                    # Step one frame
                    paused = False
                    continue
        
        # ==================== RESULTS ====================
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n{'='*100}")
        print(f"✅ VIDEO #{video_number} TEST COMPLETED")
        print(f"{'='*100}")
        print(f"📹 Video: {video_name}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"🎞️  Total Frames: {frame_count}")
        print(f"⚡ FPS: {frame_count / processing_time:.2f}")
        print(f"🚨 Total Events: {len(detected_events)}")
        
        # Event breakdown
        falls = sum(1 for e in detected_events if 'fall' in e['type'])
        seizures = sum(1 for e in detected_events if 'seizure' in e['type'] or 'abnormal' in e['type'])
        print(f"   - Falls: {falls}")
        print(f"   - Seizures/Abnormal: {seizures}")
        
        print(f"\n📊 Statistics:")
        for k, v in pipeline.stats.items():
            print(f"   {k}: {v}")
        
        print(f"\n📁 Output Files:")
        print(f"   Alerts: {alerts_folder}")
        print(f"   Keypoints: {self.output_folders['keypoints'] / f'video_{video_number}'}")
        
        print(f"{'='*100}\n")
        
        # Generate single video report
        self._generate_report(video_number, video_name, video_path, processing_time, 
                             frame_count, detected_events, pipeline.stats)
        
        # Cleanup
        camera.disconnect()
        if show_display:
            cv2.destroyAllWindows()
        
        return {
            'video_number': video_number,
            'video_name': video_name,
            'processing_time': processing_time,
            'total_frames': frame_count,
            'fps': frame_count / processing_time,
            'detected_events': detected_events,
            'statistics': pipeline.stats.copy()
        }
    
    def _generate_report(self, video_number, video_name, video_path, processing_time, 
                         frame_count, detected_events, statistics):
        """Generate Excel report for single video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_folders['reports'] / f"video_{video_number}_report_{timestamp}.xlsx"
        
        # Summary
        summary_data = [{
            'Video Number': video_number,
            'Video Name': video_name,
            'Video Path': str(video_path),
            'Processing Time (s)': processing_time,
            'Total Frames': frame_count,
            'FPS': frame_count / processing_time if processing_time > 0 else 0,
            'Total Events': len(detected_events),
            'Falls': sum(1 for e in detected_events if 'fall' in e['type']),
            'Seizures': sum(1 for e in detected_events if 'seizure' in e['type'] or 'abnormal' in e['type'])
        }]
        
        # Events
        events_data = []
        for event in detected_events:
            events_data.append({
                'Frame': event['frame'],
                'Event Type': event['type'],
                'Confidence': event['confidence'],
                'Alert Level': event['alert_level'],
                'Timestamp (s)': event['timestamp'],
                'Action': event.get('intelligent_action', ''),
                'Caption (VN)': event.get('vietnamese_caption', ''),
                'Alert Image': event.get('alert_image', '')
            })
        
        # Statistics
        stats_data = [statistics]
        
        # Write Excel (try-catch for missing openpyxl)
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                if events_data:
                    pd.DataFrame(events_data).to_excel(writer, sheet_name='Events', index=False)
                pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
            
            print(f"📄 Excel report saved: {output_path}")
        except ImportError:
            # Fallback to CSV if openpyxl not available
            csv_path = output_path.with_suffix('.csv')
            pd.DataFrame(summary_data).to_csv(csv_path, index=False)
            print(f"⚠️ openpyxl not installed, saved CSV instead: {csv_path}")
            print(f"💡 Install: pip install openpyxl")


def main():
    if len(sys.argv) < 2:
        print("="*100)
        print("🎬 Single Video Tester")
        print("="*100)
        print("\nUsage:")
        print("  python test_single_video.py <video_number> [options]")
        print("\nExamples:")
        print("  python test_single_video.py 1              # Test video 1 with display")
        print("  python test_single_video.py 1 --no-display # Test without display")
        print("  python test_single_video.py 1 --no-save    # Don't save keypoints")
        print("\nOptions:")
        print("  --no-display    Disable realtime display (faster)")
        print("  --no-save       Don't save keypoint images")
        print("="*100)
        sys.exit(1)
    
    video_number = int(sys.argv[1])
    show_display = '--no-display' not in sys.argv
    save_keypoints = '--no-save' not in sys.argv
    
    print("="*100)
    print("🏥 Vision Edge Healthcare System - Single Video Test")
    print("="*100)
    print(f"🎬 Testing video #{video_number}")
    print("="*100 + "\n")
    
    tester = SingleVideoTester()
    result = tester.test_video(video_number, show_display=show_display, save_keypoints=save_keypoints)
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Events detected: {len(result['detected_events'])}")


if __name__ == "__main__":
    main()
