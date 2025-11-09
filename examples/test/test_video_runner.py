#!/usr/bin/env python3
"""
Healthcare System Video Test Runner
Tự động quét và test tất cả video .mp4 trong folder resource/
Sử dụng toàn bộ logic từ main.py, chỉ thay camera bằng video
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
from typing import List, Dict, Any
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

# Import intelligent action generation (giống main.py)
try:
    from service.ai_vision_description_service import get_professional_caption_pipeline
    INTELLIGENT_ACTIONS_AVAILABLE = True
    print("🤖 Intelligent Action Generation: AVAILABLE")
except ImportError:
    INTELLIGENT_ACTIONS_AVAILABLE = False
    print("📝 Intelligent Action Generation: Using static messages")


class VideoTestRunner:
    """Test runner tự động quét và test tất cả video .mp4"""
    
    def __init__(self, resource_folder: str = ""):
        # Get script directory
        script_dir = Path(__file__).parent
        self.resource_folder = Path(resource_folder) if resource_folder else script_dir / "resource"
        self.output_base = script_dir / "test_results"
        
        # Create output folders
        self.output_folders = {
            'reports': self.output_base / 'reports',
            'alerts': self.output_base / 'alerts',
            'keypoints': self.output_base / 'keypoints',
            'statistics': self.output_base / 'statistics',
            'logs': self.output_base / 'logs'
        }
        
        for folder in self.output_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # Load user ID từ environment (giống main.py)
        self.user_id = os.getenv('DEFAULT_USER_ID', 'test_user_001')
        logger.info(f"✅ Using user ID: {self.user_id}")
        
        # Results storage
        self.all_results = []
        self.case_number = 1
    
    def find_all_videos(self) -> List[Path]:
        """Tự động tìm tất cả video .mp4/.MP4 trong resource folder"""
        # Tìm cả .mp4 và .MP4
        videos_lower = list(self.resource_folder.glob("*.mp4"))
        videos_upper = list(self.resource_folder.glob("*.MP4"))
        videos = sorted(videos_lower + videos_upper, key=lambda x: x.name.lower())
        
        logger.info(f"📹 Found {len(videos)} video files in {self.resource_folder}")
        for video in videos:
            logger.info(f"   - {video.name}")
        return videos
    
    def run_single_video_test(self, video_path: Path, case_number: int) -> Dict[str, Any]:
        """
        Test 1 video - SỬ DỤNG TOÀN BỘ LOGIC TỪ MAIN.PY
        """
        video_name = video_path.stem
        
        print("\n" + "="*100)
        print(f"🎬 CASE #{case_number}: Testing Video '{video_name}'")
        print(f"📹 Path: {video_path}")
        print("="*100 + "\n")
        
        # ==================== SETUP GIỐNG MAIN.PY ====================
        
        # 1. Configure camera từ video
        camera_config = {
            'video_path': str(video_path),
            'buffer_size': 1,
            'fps': 30,
            'resolution': (1920, 1080),
            'auto_reconnect': True,
            'camera_id': f"test_case_{case_number}",
            'camera_name': video_name,
            'loop': False
        }
        
        processor_config = 120
        alerts_folder = str(self.output_folders['alerts'] / f"case_{case_number}")
        Path(alerts_folder).mkdir(parents=True, exist_ok=True)
        
        # 2. Initialize services (GIỐNG MAIN.PY)
        print("🔧 Initializing services...")
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            return {
                'case_number': case_number,
                'video_name': video_name,
                'video_path': str(video_path),
                'status': 'failed',
                'error': 'Failed to load video'
            }
        
        video_processor = VideoProcessingService(processor_config)
        fall_detector = FallDetectionService()
        seizure_detector = SeizureDetectionService()
        seizure_predictor = SeizurePredictor(
            temporal_window=3,
            alert_threshold=0.01,
            warning_threshold=0.005
        )
        
        # 3. Initialize Healthcare Pipeline (GIỐNG MAIN.PY)
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
        
        # 4. Initialize intelligent action pipeline (GIỐNG MAIN.PY)
        caption_pipeline = None
        if INTELLIGENT_ACTIONS_AVAILABLE:
            try:
                caption_pipeline = get_professional_caption_pipeline()
                print("🤖 Intelligent action pipeline initialized")
            except Exception as e:
                print(f"⚠️ Could not load intelligent actions: {e}")
        
        print("\n✅ All systems initialized!")
        print("="*100)
        
        # ==================== PROCESSING LOOP (GIỐNG MAIN.PY) ====================
        
        print(f"\n🎥 Starting video processing for CASE #{case_number}...")
        print("="*100 + "\n")
        
        # Test tracking variables
        detected_events = []
        frame_count = 0
        start_time = time.time()
        last_alert_image_path = None
        saved_keypoint_images = []
        
        # Statistics
        total_persons_detected = 0
        total_falls = 0
        total_seizures = 0
        
        # Main processing loop (GIỐNG MAIN.PY)
        while True:
            frame = camera.get_frame()
            if frame is None:
                print(f"\n✅ Video processing completed for CASE #{case_number}")
                break
            
            frame_count += 1
            
            # Process frame (GIỐNG MAIN.PY)
            result = pipeline.process_frame(frame)
            detection_result = result["detection_result"]
            person_detections = result["person_detections"]
            
            # Count persons
            if person_detections:
                total_persons_detected += len(person_detections)
            
            # ===== SAVE KEYPOINT IMAGES =====
            # Save every 30 frames or when detection occurs
            if frame_count % 30 == 0 or detection_result.get('alert_level') in ['critical', 'high', 'warning']:
                try:
                    keypoint_folder = self.output_folders['keypoints'] / f"case_{case_number}"
                    keypoint_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Draw keypoints on frame
                    keypoint_frame = pipeline.visualize_dual_detection(frame.copy(), detection_result, person_detections)
                    
                    # Add frame info
                    info_text = f"Frame: {frame_count} | Case: {case_number}"
                    cv2.putText(keypoint_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    keypoint_path = keypoint_folder / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(keypoint_path), keypoint_frame)
                    saved_keypoint_images.append(str(keypoint_path))
                except Exception as e:
                    logger.error(f"Error saving keypoint image: {e}")
            
            # ===== DETECT EVENTS (GIỐNG MAIN.PY) =====
            if detection_result.get('alert_level') in ['critical', 'high', 'warning']:
                emergency_type = detection_result.get('emergency_type', 'unknown')
                confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
                
                # Count events
                if 'fall' in emergency_type:
                    total_falls += 1
                elif 'seizure' in emergency_type or 'abnormal' in emergency_type:
                    total_seizures += 1
                
                # Try to find alert image
                alert_image_path = None
                try:
                    alerts_dir = Path(alerts_folder)
                    if alerts_dir.exists():
                        alert_files = sorted(alerts_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                        if alert_files:
                            alert_image_path = str(alert_files[0])
                            last_alert_image_path = alert_image_path
                except Exception as e:
                    logger.error(f"Error finding alert image: {e}")
                
                # Generate intelligent action / Vietnamese caption (GIỐNG MAIN.PY)
                intelligent_action = "Standard alert message"
                vietnamese_caption = ""
                
                if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline and alert_image_path:
                    try:
                        action_result = caption_pipeline(alert_image_path)
                        if action_result and 'recommended_action' in action_result:
                            intelligent_action = action_result['recommended_action']
                        if action_result and 'vietnamese_caption' in action_result:
                            vietnamese_caption = action_result['vietnamese_caption']
                    except Exception as e:
                        logger.error(f"Error generating intelligent action: {e}")
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
                    'alert_image': alert_image_path,
                    'intelligent_action': intelligent_action,
                    'vietnamese_caption': vietnamese_caption
                }
                detected_events.append(event)
                
                print(f"\n{'='*100}")
                print(f"🚨 ALERT DETECTED - CASE #{case_number}")
                print(f"{'='*100}")
                print(f"   Frame: {frame_count}/{camera.total_frames}")
                print(f"   Event Type: {emergency_type}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Alert Level: {detection_result.get('alert_level')}")
                print(f"   📝 Action: {intelligent_action}")
                print(f"   🇻🇳 Caption: {vietnamese_caption}")
                if alert_image_path:
                    print(f"   📸 Alert Image: {alert_image_path}")
                print(f"{'='*100}\n")
            
            # Show progress every 100 frames
            if frame_count % 100 == 0:
                camera_status = camera.get_status()
                print(f"📊 Progress: {camera_status['progress']} - Frame {frame_count}/{camera.total_frames}")
        
        # ==================== CLEANUP & RESULTS ====================
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n{'='*100}")
        print(f"✅ CASE #{case_number} COMPLETED: {video_name}")
        print(f"{'='*100}")
        print(f"   Processing Time: {processing_time:.2f}s")
        print(f"   Total Frames: {frame_count}")
        print(f"   FPS: {frame_count / processing_time:.2f}")
        print(f"   Detected Events: {len(detected_events)}")
        print(f"   - Falls: {total_falls}")
        print(f"   - Seizures: {total_seizures}")
        print(f"   Saved Keypoint Images: {len(saved_keypoint_images)}")
        print(f"{'='*100}\n")
        
        # Prepare result
        result = {
            'case_number': case_number,
            'video_name': video_name,
            'video_path': str(video_path),
            'status': 'completed',
            'processing_time': processing_time,
            'total_frames': frame_count,
            'fps': frame_count / processing_time if processing_time > 0 else 0,
            'detected_events': detected_events,
            'statistics': {
                **pipeline.stats.copy(),
                'total_persons_detected': total_persons_detected,
                'total_falls': total_falls,
                'total_seizures': total_seizures,
                'keypoint_images_saved': len(saved_keypoint_images)
            },
            'intelligent_actions_used': INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline is not None,
            'saved_keypoint_images': saved_keypoint_images[:10]  # First 10 for report
        }
        
        # Cleanup
        camera.disconnect()
        
        return result
    
    def generate_excel_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive Excel report"""
        if not results:
            logger.warning("No results to generate report")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_folders['reports'] / f"test_report_{timestamp}.xlsx"
        
        print(f"\n📊 Generating Excel report...")
        
        # ===== SHEET 1: SUMMARY =====
        summary_data = []
        for result in results:
            summary_row = {
                'Case Number': result['case_number'],
                'Video Name': result['video_name'],
                'Video Path': result['video_path'],
                'Status': result['status'],
                'Processing Time (s)': result.get('processing_time', 0),
                'Total Frames': result.get('total_frames', 0),
                'FPS': result.get('fps', 0),
                'Total Events': len(result.get('detected_events', [])),
                'Falls Detected': result.get('statistics', {}).get('total_falls', 0),
                'Seizures Detected': result.get('statistics', {}).get('total_seizures', 0),
                'Keypoint Images Saved': result.get('statistics', {}).get('keypoint_images_saved', 0),
                'Intelligent Actions': 'Yes' if result.get('intelligent_actions_used') else 'No'
            }
            summary_data.append(summary_row)
        
        # ===== SHEET 2: DETECTED EVENTS =====
        events_data = []
        for result in results:
            case_num = result['case_number']
            video_name = result['video_name']
            
            for event in result.get('detected_events', []):
                event_row = {
                    'Case Number': case_num,
                    'Video Name': video_name,
                    'Frame': event['frame'],
                    'Event Type': event['type'],
                    'Confidence': event['confidence'],
                    'Alert Level': event['alert_level'],
                    'Timestamp (s)': event['timestamp'],
                    'Intelligent Action': event.get('intelligent_action', ''),
                    'Vietnamese Caption': event.get('vietnamese_caption', ''),
                    'Alert Image Path': event.get('alert_image', '')
                }
                events_data.append(event_row)
        
        # ===== SHEET 3: STATISTICS =====
        stats_data = []
        for result in results:
            if 'statistics' in result:
                stats_row = {
                    'Case Number': result['case_number'],
                    'Video Name': result['video_name'],
                    **result['statistics']
                }
                stats_data.append(stats_row)
        
        # ===== WRITE EXCEL =====
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Events sheet
            if events_data:
                df_events = pd.DataFrame(events_data)
                df_events.to_excel(writer, sheet_name='Detected Events', index=False)
            
            # Statistics sheet
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"✅ Excel report generated: {output_path}")
        return output_path
    
    def run_all_tests(self):
        """Tự động chạy test tất cả video trong resource folder"""
        videos = self.find_all_videos()
        
        if not videos:
            print(f"❌ No .mp4 videos found in {self.resource_folder}")
            print(f"💡 Please add video files to {self.resource_folder}")
            return
        
        print("\n" + "="*100)
        print(f"🧪 HEALTHCARE SYSTEM VIDEO TEST SUITE")
        print("="*100)
        print(f"📹 Total Videos: {len(videos)}")
        print(f"📂 Resource Folder: {self.resource_folder}")
        print(f"📊 Output Folder: {self.output_base}")
        print(f"👤 User ID: {self.user_id}")
        print(f"🤖 Intelligent Actions: {'ENABLED' if INTELLIGENT_ACTIONS_AVAILABLE else 'DISABLED'}")
        print("="*100 + "\n")
        
        results = []
        
        for video_path in videos:
            result = self.run_single_video_test(video_path, self.case_number)
            results.append(result)
            self.all_results.append(result)
            self.case_number += 1
            
            # Note completion
            print(f"\n✅ Completed video: {video_path.name}")
            print(f"   Status: {result['status']}")
            if result['status'] == 'completed':
                print(f"   Events detected: {len(result.get('detected_events', []))}")
            print()
        
        # Generate final report
        print("\n" + "="*100)
        print("📊 GENERATING FINAL REPORT")
        print("="*100)
        
        report_path = self.generate_excel_report(results)
        
        print("\n" + "="*100)
        print("✅ ALL TESTS COMPLETED!")
        print("="*100)
        print(f"📊 Total Cases: {len(results)}")
        print(f"📁 Results saved to: {self.output_base}")
        print(f"📄 Excel Report: {report_path}")
        print(f"📸 Alert Images: {self.output_folders['alerts']}")
        print(f"🎯 Keypoint Images: {self.output_folders['keypoints']}")
        print("="*100 + "\n")


if __name__ == "__main__":
    print("="*100)
    print("🏥 Vision Edge Healthcare System - Video Test Mode")
    print("="*100 + "\n")
    
    runner = VideoTestRunner()
    runner.run_all_tests()
