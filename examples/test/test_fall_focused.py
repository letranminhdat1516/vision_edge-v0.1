#!/usr/bin/env python3
"""
Single Video Tester - OPTIMIZED FOR FALL DETECTION
Phiên bản tối ưu cho phát hiện té ngã
"""

import os
import sys
import cv2
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import video camera service
from video_camera_service import VideoCameraService

# Import services - SỬ DỤNG TRỰC TIẾP THAY VÌ QUA PIPELINE
from service.video_processing_service import VideoProcessingService
from service.fall_detection_service import FallDetectionService
from service.seizure_detection_service import SeizureDetectionService
from seizure_detection.seizure_predictor import SeizurePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FallFocusedTester:
    """Test tập trung vào fall detection, bỏ qua pipeline phức tạp"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.resource_folder = self.script_dir / "resource"
        self.output_base = self.script_dir / "test_results"
        self.user_id = os.getenv('DEFAULT_USER_ID', 'test_user_001')
        
        self.output_folders = {
            'alerts': self.output_base / 'alerts',
            'keypoints': self.output_base / 'keypoints',
            'reports': self.output_base / 'reports'
        }
        
        for folder in self.output_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def find_video(self, video_number: int) -> Path:
        """Tìm video theo số"""
        video_path_lower = self.resource_folder / f"{video_number}.mp4"
        video_path_upper = self.resource_folder / f"{video_number}.MP4"
        
        if video_path_lower.exists():
            return video_path_lower
        elif video_path_upper.exists():
            return video_path_upper
        else:
            raise FileNotFoundError(f"Video {video_number} not found")
    
    def test_video(self, video_number: int, show_display: bool = True):
        """Test video với focus vào fall detection"""
        
        video_path = self.find_video(video_number)
        video_name = video_path.stem
        
        print("\n" + "="*100)
        print(f"🎬 FALL-FOCUSED TEST - VIDEO #{video_number}")
        print("="*100)
        print(f"📹 File: {video_path.name}")
        print(f"🎯 Mode: FALL DETECTION PRIORITY")
        print("="*100 + "\n")
        
        # Setup camera
        camera_config = {
            'video_path': str(video_path),
            'fps': 30,
            'resolution': None,
            'camera_id': f"test_video_{video_number}",
            'camera_name': video_name,
            'loop': False
        }
        
        alerts_folder = self.output_folders['alerts'] / f"video_{video_number}"
        keypoint_folder = self.output_folders['keypoints'] / f"video_{video_number}"
        alerts_folder.mkdir(parents=True, exist_ok=True)
        keypoint_folder.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Initializing services...")
        camera = VideoCameraService(camera_config)
        if not camera.connect():
            print(f"❌ Failed to load video")
            return None
        
        # Initialize DIRECT detection services - BỎ QUA VideoProcessingService
        # Gọi trực tiếp IntegratedVideoProcessor với YOLO confidence thấp
        from video_processing.simple_processing import IntegratedVideoProcessor
        
        video_processor = IntegratedVideoProcessor(
            motion_threshold=1,      # Process mọi frame
            keyframe_threshold=0.1,  # Keyframe threshold thấp
            yolo_confidence=0.15,    # GIẢM từ 0.5 -> 0.15 (cực thấp để detect người)
            save_frames=False
        )
        print(f"✅ Video processor initialized with YOLO confidence=0.15")
        
        # FALL DETECTOR - CỰC KỲ NHẠY
        fall_detector = FallDetectionService(confidence_threshold=0.05)
        
        print("✅ Services ready!")
        print("="*100)
        
        # Processing
        print(f"\n🎥 Processing video #{video_number} - DIRECT FALL DETECTION...")
        if show_display:
            print("🎮 Press 'q' to quit, 's' for stats")
        print("="*100 + "\n")
        
        detected_events = []
        frame_count = 0
        start_time = time.time()
        
        # Statistics
        stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'fall_detections': 0,
            'high_confidence_falls': 0
        }
        
        while True:
            frame = camera.get_frame()
            if frame is None:
                print(f"\n✅ Video completed!")
                break
            
            frame_count += 1
            stats['total_frames'] += 1
            
            # Process frame to get person detections
            result = video_processor.process_frame(frame)
            
            # IntegratedVideoProcessor trả về 'detections', không phải 'person_detections'
            all_detections = result.get('detections', [])
            person_detections = [d for d in all_detections if d.get('class_name') == 'person']
            
            # DEBUG: Log detection status
            if frame_count % 30 == 0:  # Every 30 frames
                processed = result.get('processed', False)
                is_keyframe = result.get('is_keyframe', False)
                print(f"🔍 Frame {frame_count}: Processed={processed}, Keyframe={is_keyframe}, "
                      f"Persons={len(person_detections)}, All objects={len(all_detections)}")
                if not processed:
                    print(f"   ⚠️ Frame not processed: {result.get('reason', 'unknown')}")
                if not person_detections and all_detections:
                    print(f"   ℹ️ Detected objects but no persons: {[d['class_name'] for d in all_detections[:3]]}")
            
            if person_detections:
                stats['frames_with_person'] += 1
                
                # Test EVERY person for fall
                for person in person_detections:
                    try:
                        # DIRECT FALL DETECTION - Không qua pipeline
                        fall_result = fall_detector.detect_fall(frame, person)
                        confidence = fall_result.get('confidence', 0)
                        fall_detected = fall_result.get('fall_detected', False)
                        
                        # DEBUG: Lưu MỌI lần gọi fall detector
                        if frame_count % 30 == 0:  # Log every 30 frames
                            print(f"   🩺 Fall detector result: confidence={confidence:.3f}, detected={fall_detected}")
                            print(f"      Method: {fall_result.get('method', 'unknown')}")
                            if 'error' in fall_result:
                                print(f"      ❌ Error: {fall_result['error']}")
                        
                        # Lưu mọi confidence > 0 để phân tích
                        if confidence > 0:
                            print(f"📊 Frame {frame_count}: Fall confidence = {confidence:.3f}")
                        
                        # THRESHOLD CỰC THẤP - Bắt mọi khả năng té ngã
                        if confidence > 0.05 or fall_detected:  # Threshold = 0.05
                            stats['fall_detections'] += 1
                            
                            event = {
                                'frame': frame_count,
                                'type': 'fall',
                                'confidence': confidence,
                                'timestamp': time.time() - start_time,
                                'method': 'direct_detection'
                            }
                            detected_events.append(event)
                            
                            # Save alert image
                            alert_path = alerts_folder / f"fall_frame_{frame_count:06d}_conf_{confidence:.2f}.jpg"
                            
                            # Vẽ keypoints và save
                            alert_frame = frame.copy()
                            
                            # Vẽ bounding box
                            if hasattr(person, 'boxes'):
                                boxes = person.boxes
                                if hasattr(boxes, 'xyxy'):
                                    bbox = boxes.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = map(int, bbox)
                                    cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Vẽ skeleton nếu có
                            if hasattr(person, 'keypoints') and person.keypoints is not None:
                                keypoints = person.keypoints.xy[0].cpu().numpy()
                                connections = [
                                    (0, 1), (0, 2), (1, 3), (2, 4),
                                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                                    (5, 11), (6, 12), (11, 12),
                                    (11, 13), (13, 15), (12, 14), (14, 16)
                                ]
                                
                                for start, end in connections:
                                    if start < len(keypoints) and end < len(keypoints):
                                        pt1 = tuple(map(int, keypoints[start]))
                                        pt2 = tuple(map(int, keypoints[end]))
                                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                            cv2.line(alert_frame, pt1, pt2, (0, 255, 255), 2)
                                
                                for kpt in keypoints:
                                    x, y = int(kpt[0]), int(kpt[1])
                                    if x > 0 and y > 0:
                                        cv2.circle(alert_frame, (x, y), 4, (0, 0, 255), -1)
                            
                            # Add text info
                            cv2.putText(alert_frame, f"FALL DETECTED!", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(alert_frame, f"Confidence: {confidence:.2%}", (10, 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(alert_frame, f"Frame: {frame_count}", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                            
                            cv2.imwrite(str(alert_path), alert_frame)
                            
                            if confidence > 0.5:
                                stats['high_confidence_falls'] += 1
                            
                            print(f"\n{'='*100}")
                            print(f"🚨 FALL ALERT #{stats['fall_detections']}")
                            print(f"{'='*100}")
                            print(f"   Frame: {frame_count}/{camera.total_frames}")
                            print(f"   Confidence: {confidence:.2%}")
                            print(f"   Timestamp: {event['timestamp']:.2f}s")
                            print(f"   Image: {alert_path.name}")
                            print(f"{'='*100}\n")
                    
                    except Exception as e:
                        logger.error(f"Error in fall detection: {e}")
            
            # Display
            if show_display:
                display_frame = frame.copy()
                
                # Draw all persons
                for person in person_detections:
                    if hasattr(person, 'boxes'):
                        boxes = person.boxes
                        if hasattr(boxes, 'xyxy'):
                            bbox = boxes.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add info
                cv2.putText(display_frame, f"Frame: {frame_count}/{camera.total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Persons: {len(person_detections)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Falls: {stats['fall_detections']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(f"Fall Detection - Video #{video_number}", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Quitting...")
                    break
                elif key == ord('s'):
                    print(f"\n📊 Statistics:")
                    for k, v in stats.items():
                        print(f"   {k}: {v}")
                    print()
            
            if frame_count % 100 == 0:
                print(f"📊 Progress: {frame_count}/{camera.total_frames} - Falls: {stats['fall_detections']}")
        
        # Results
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n{'='*100}")
        print(f"✅ TEST COMPLETED - VIDEO #{video_number}")
        print(f"{'='*100}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"🎞️  Total Frames: {frame_count}")
        print(f"👤 Frames with Person: {stats['frames_with_person']}")
        print(f"🚨 Total Fall Detections: {stats['fall_detections']}")
        print(f"⚠️  High Confidence Falls (>50%): {stats['high_confidence_falls']}")
        print(f"⚡ FPS: {frame_count / processing_time:.2f}")
        print(f"{'='*100}\n")
        
        # Generate report
        self._generate_report(video_number, video_name, video_path, 
                             processing_time, frame_count, detected_events, stats)
        
        camera.disconnect()
        if show_display:
            cv2.destroyAllWindows()
        
        return {
            'video_number': video_number,
            'processing_time': processing_time,
            'detected_events': detected_events,
            'statistics': stats
        }
    
    def _generate_report(self, video_number, video_name, video_path,
                         processing_time, frame_count, detected_events, statistics):
        """Generate CSV report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_folders['reports'] / f"video_{video_number}_fall_focused_{timestamp}.csv"
        
        summary_data = [{
            'Video Number': video_number,
            'Video Name': video_name,
            'Video Path': str(video_path),
            'Processing Time (s)': processing_time,
            'Total Frames': frame_count,
            'FPS': frame_count / processing_time if processing_time > 0 else 0,
            'Total Fall Detections': len(detected_events),
            'High Confidence Falls': statistics.get('high_confidence_falls', 0),
            'Frames with Person': statistics.get('frames_with_person', 0)
        }]
        
        events_data = []
        for event in detected_events:
            events_data.append({
                'Frame': event['frame'],
                'Type': event['type'],
                'Confidence': event['confidence'],
                'Timestamp (s)': event['timestamp'],
                'Method': event.get('method', 'direct')
            })
        
        # Write CSV
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_path, index=False)
        
        if events_data:
            events_path = output_path.with_name(f"video_{video_number}_events_{timestamp}.csv")
            df_events = pd.DataFrame(events_data)
            df_events.to_csv(events_path, index=False)
            print(f"📄 Events saved: {events_path}")
        
        print(f"📄 Report saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fall_focused.py <video_number>")
        print("Example: python test_fall_focused.py 1")
        sys.exit(1)
    
    video_number = int(sys.argv[1])
    show_display = '--no-display' not in sys.argv
    
    print("="*100)
    print("🏥 FALL-FOCUSED VIDEO TESTER")
    print("="*100)
    print("🎯 Optimized for Fall Detection")
    print("🚫 Bypasses complex pipeline")
    print("="*100 + "\n")
    
    tester = FallFocusedTester()
    result = tester.test_video(video_number, show_display=show_display)
    
    if result:
        print(f"\n✅ Test completed!")
        print(f"🚨 Fall events detected: {len(result['detected_events'])}")


if __name__ == "__main__":
    main()
