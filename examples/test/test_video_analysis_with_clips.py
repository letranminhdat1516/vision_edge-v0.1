"""
Video Analysis Test with Event Clip Extraction
Phân tích video: đếm số lần ngã, co giật, số người, và cắt clip 5s khi có sự kiện
"""

import cv2
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator
from fall_detection.simple_fall_detector import SimpleFallDetector
from seizure_detection.seizure_predictor import SeizurePredictor
from seizure_detection.vsvig_detector import VSViGSeizureDetector


class VideoEventAnalyzer:
    """Phân tích video và trích xuất các đoạn clip có sự kiện"""
    
    def __init__(self, output_base_dir="test_results/video_analysis"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detection services
        print("🔧 Initializing detection services...")
        
        # YOLO Pose for person detection - Increased confidence for better accuracy
        self.pose_estimator = YOLOv8PoseEstimator(model_size='n')
        
        # Fall detector
        self.fall_detector = SimpleFallDetector()
        
        # Seizure detector with predictor
        self.seizure_detector = VSViGSeizureDetector()
        self.seizure_predictor = SeizurePredictor(
            temporal_window=3,
            alert_threshold=0.70,
            warning_threshold=0.55
        )
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'people_detected': 0,
            'fall_events': [],
            'seizure_events': [],
            'max_people_in_frame': 0
        }
        
        # Event detection state - Reduced cooldown for more sensitive detection
        self.fall_cooldown_time = None
        self.seizure_cooldown_time = None
        self.fall_cooldown_duration = 3.0  # Reduced from 5s to 3s for faster re-detection
        self.seizure_cooldown_duration = 5.0
        
        # Video writer for clips
        self.current_clip_writer = None
        self.clip_frames_buffer = []
        self.clip_buffer_size = 90  # Pre-buffer 3s at 30fps (3 * 30) - Increased for better context
        
    def draw_detection_info(self, frame, person_count, fall_conf, seizure_conf, fall_detected, seizure_detected, total_falls, total_seizures):
        """Vẽ thông tin detection lên frame với số đếm tích lũy"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Draw background rectangle for text (larger for event counts)
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        y_offset = 40
        x_offset = 20
        
        # Person count
        color = (0, 255, 0) if person_count > 0 else (128, 128, 128)
        cv2.putText(frame, f"Person: {person_count}", 
                   (x_offset, y_offset), font, font_scale, color, thickness)
        
        # Fall detection with count
        y_offset += 35
        if fall_detected:
            color = (0, 0, 255)  # Red
            text = f"Fall: DETECTED (conf: {fall_conf:.2f}) [Total: {total_falls}]"
        else:
            color = (128, 128, 128)  # Gray
            text = f"Fall: NO (conf: {fall_conf:.2f}) [Total: {total_falls}]"
        cv2.putText(frame, text, (x_offset, y_offset), font, 0.6, color, thickness)
        
        # Seizure detection with count
        y_offset += 35
        if seizure_detected:
            color = (255, 0, 255)  # Magenta
            text = f"Seizure: DETECTED (conf: {seizure_conf:.2f}) [Total: {total_seizures}]"
        else:
            color = (128, 128, 128)  # Gray
            text = f"Seizure: NO (conf: {seizure_conf:.2f}) [Total: {total_seizures}]"
        cv2.putText(frame, text, (x_offset, y_offset), font, 0.6, color, thickness)
        
        # Event summary
        y_offset += 35
        summary_color = (255, 255, 0)  # Yellow
        cv2.putText(frame, f"Summary: {total_falls} Falls | {total_seizures} Seizures",
                   (x_offset, y_offset), font, 0.65, summary_color, 2)
        
        return frame
    
    def start_clip_recording(self, video_name, event_type, frame_number, fps, frame_size):
        """Bắt đầu ghi clip 5 giây"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_dir = self.output_base_dir / video_name / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        clip_filename = f"{event_type}_frame{frame_number}_{timestamp}.mp4"
        clip_path = clip_dir / clip_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_clip_writer = cv2.VideoWriter(
            str(clip_path), fourcc, fps, frame_size
        )
        
        # Write buffered frames (before event)
        for buffered_frame in self.clip_frames_buffer:
            self.current_clip_writer.write(buffered_frame)
        
        return clip_path
    
    def update_clip_buffer(self, frame):
        """Cập nhật buffer frames để có video trước khi sự kiện xảy ra"""
        self.clip_frames_buffer.append(frame.copy())
        if len(self.clip_frames_buffer) > self.clip_buffer_size:
            self.clip_frames_buffer.pop(0)
    
    def analyze_video(self, video_path, video_name=None):
        """Phân tích một video và trích xuất các clip có sự kiện"""
        if video_name is None:
            video_name = Path(video_path).stem
        
        print(f"\n{'='*60}")
        print(f"📹 Analyzing video: {video_name}")
        print(f"{'='*60}")
        
        # Reset statistics
        self.stats = {
            'total_frames': 0,
            'people_detected': 0,
            'fall_events': [],
            'seizure_events': [],
            'max_people_in_frame': 0,
            'video_name': video_name
        }
        self.fall_cooldown_time = None
        self.seizure_cooldown_time = None
        self.clip_frames_buffer = []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        
        print(f"📊 Video properties:")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Resolution: {width}x{height}")
        
        # Create output directory for this video
        video_output_dir = self.output_base_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        clip_recording = False
        clip_frames_remaining = 0
        clip_total_frames = int(fps * 6)  # 6 seconds - Extended clip duration for better analysis
        
        print(f"\n🎬 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.stats['total_frames'] = frame_count
            current_time = frame_count / fps
            
            # Person detection using YOLO Pose
            keypoints = self.pose_estimator.extract_keypoints(frame, confidence_threshold=0.25)
            
            # Count if person detected
            person_count = 1 if keypoints is not None else 0
            
            if person_count > 0:
                self.stats['people_detected'] += 1
                self.stats['max_people_in_frame'] = max(
                    self.stats['max_people_in_frame'], 
                    person_count
                )
            
            # Initialize detection values
            fall_confidence = 0.0
            seizure_confidence = 0.0
            fall_detected = False
            seizure_detected = False
            
            # Fall detection (only if person detected)
            if person_count > 0 and keypoints is not None:
                # Extract bounding box from keypoints
                keypoints_xy = keypoints[:, :2]  # Get x, y coordinates
                valid_points = keypoints_xy[keypoints[:, 2] > 0.3]  # Points with confidence > 0.3
                
                if len(valid_points) > 0:
                    x_min, y_min = valid_points.min(axis=0)
                    x_max, y_max = valid_points.max(axis=0)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, int(x_min) - padding)
                    y_min = max(0, int(y_min) - padding)
                    x_max = min(frame.shape[1], int(x_max) + padding)
                    y_max = min(frame.shape[0], int(y_max) + padding)
                    
                    person_bbox = [x_min, y_min, x_max, y_max]
                    bbox_xywh = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    # Check fall cooldown
                    if self.fall_cooldown_time is None or (current_time - self.fall_cooldown_time) > self.fall_cooldown_duration:
                        # Prepare person dict for fall detector
                        # Convert keypoints (17, 3) to flat list [x1, y1, conf1, x2, y2, conf2, ...]
                        keypoints_flat = keypoints.flatten().tolist() if keypoints is not None else []
                        
                        person_dict = {
                            'bbox': bbox_xywh,  # [x, y, w, h]
                            'keypoints': keypoints_flat
                        }
                        
                        fall_result = self.fall_detector.detect_fall(
                            current_frame=frame,
                            timestamp=current_time,
                            person_bbox=person_bbox
                        )
                        fall_confidence = fall_result.get('confidence', 0.0)
                        
                        # Fall detection threshold - LOWERED for higher sensitivity
                        if fall_confidence >= 0.15:  # Reduced from 0.20 to 0.15
                            fall_detected = True
                            self.fall_cooldown_time = current_time
                            
                            # Record event
                            event_info = {
                                'frame': frame_count,
                                'time': current_time,
                                'confidence': fall_confidence,
                                'method': fall_result.get('method', 'unknown')
                            }
                            self.stats['fall_events'].append(event_info)
                            
                            # Start clip recording
                            if not clip_recording:
                                clip_path = self.start_clip_recording(
                                    video_name, 'fall', frame_count, fps, frame_size
                                )
                                clip_recording = True
                                clip_frames_remaining = clip_total_frames
                                print(f"🎬 Recording fall clip: {clip_path.name}")
                    
                    # Seizure detection
                    if self.seizure_cooldown_time is None or (current_time - self.seizure_cooldown_time) > self.seizure_cooldown_duration:
                        # Analyze for seizure using VSViG detector
                        seizure_result = self.seizure_detector.detect_seizure(
                            frame=frame,
                            person_bbox=person_bbox
                        )
                        
                        if seizure_result and seizure_result.get('temporal_ready', False):
                            base_confidence = seizure_result.get('confidence', 0.0)
                            pred_result = self.seizure_predictor.update_prediction(base_confidence)
                            seizure_confidence = pred_result['smoothed_confidence']
                            
                            # Seizure detection threshold
                            if seizure_confidence >= 0.70:
                                seizure_detected = True
                                self.seizure_cooldown_time = current_time
                                
                                # Record event
                                event_info = {
                                    'frame': frame_count,
                                    'time': current_time,
                                    'confidence': seizure_confidence
                                }
                                self.stats['seizure_events'].append(event_info)
                                
                                # Start clip recording
                                if not clip_recording:
                                    clip_path = self.start_clip_recording(
                                        video_name, 'seizure', frame_count, fps, frame_size
                                    )
                                    clip_recording = True
                                    clip_frames_remaining = clip_total_frames
                                    print(f"🎬 Recording seizure clip: {clip_path.name}")
            
            # Draw detection info on frame with cumulative counts
            display_frame = self.draw_detection_info(
                frame.copy(),
                person_count,
                fall_confidence,
                seizure_confidence,
                fall_detected,
                seizure_detected,
                len(self.stats['fall_events']),
                len(self.stats['seizure_events'])
            )
            
            # Update clip buffer
            self.update_clip_buffer(display_frame)
            
            # Write to clip if recording
            if clip_recording and self.current_clip_writer:
                self.current_clip_writer.write(display_frame)
                clip_frames_remaining -= 1
                
                if clip_frames_remaining <= 0:
                    self.current_clip_writer.release()
                    self.current_clip_writer = None
                    clip_recording = False
                    print(f"✅ Clip saved")
            
            # Progress display with debug info
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                      f"Falls: {len(self.stats['fall_events'])} | Seizures: {len(self.stats['seizure_events'])} | "
                      f"fall_conf={fall_confidence:.3f}, seiz_conf={seizure_confidence:.3f}")
        
        # Release video
        cap.release()
        
        # Close any remaining clip
        if self.current_clip_writer:
            self.current_clip_writer.release()
        
        # Save statistics
        self.save_statistics(video_name)
        
        return self.stats
    
    def save_statistics(self, video_name):
        """Lưu thống kê ra file JSON"""
        stats_dir = self.output_base_dir / video_name
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = stats_dir / "statistics.json"
        
        # Calculate summary
        summary = {
            'video_name': video_name,
            'total_frames': self.stats['total_frames'],
            'total_people_detected_frames': self.stats['people_detected'],
            'max_people_in_frame': self.stats['max_people_in_frame'],
            'fall_count': len(self.stats['fall_events']),
            'seizure_count': len(self.stats['seizure_events']),
            'fall_events': self.stats['fall_events'],
            'seizure_events': self.stats['seizure_events'],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Statistics saved: {stats_file}")
    
    def print_summary(self):
        """In ra tóm tắt kết quả"""
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Video: {self.stats['video_name']}")
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Frames with people: {self.stats['people_detected']}")
        print(f"Max people in frame: {self.stats['max_people_in_frame']}")
        print(f"\n🚨 Fall Events: {len(self.stats['fall_events'])}")
        for i, event in enumerate(self.stats['fall_events'], 1):
            print(f"   {i}. Frame {event['frame']} ({event['time']:.2f}s) - "
                  f"Confidence: {event['confidence']:.2f} - Method: {event['method']}")
        
        print(f"\n🧠 Seizure Events: {len(self.stats['seizure_events'])}")
        for i, event in enumerate(self.stats['seizure_events'], 1):
            print(f"   {i}. Frame {event['frame']} ({event['time']:.2f}s) - "
                  f"Confidence: {event['confidence']:.2f}")
        
        print(f"{'='*60}\n")


def main():
    """Main function để test"""
    print("="*60)
    print("🎬 VIDEO EVENT ANALYSIS WITH CLIP EXTRACTION")
    print("="*60)
    
    # Check for command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Phân tích video và cắt clip sự kiện')
    parser.add_argument('video_paths', nargs='*', help='Đường dẫn đến video cần phân tích')
    parser.add_argument('--dir', '-d', help='Thư mục chứa video để phân tích hàng loạt')
    parser.add_argument('--output', '-o', default='test_results/video_analysis', 
                       help='Thư mục output (mặc định: test_results/video_analysis)')
    args = parser.parse_args()
    
    video_files = []
    
    # Nếu có video paths từ command line
    if args.video_paths:
        for path_str in args.video_paths:
            path = Path(path_str)
            if path.exists():
                video_files.append(path)
            else:
                print(f"⚠️  Video not found: {path_str}")
    
    # Nếu có thư mục chỉ định
    elif args.dir:
        video_dir = Path(args.dir)
        if video_dir.exists():
            video_files = (list(video_dir.glob("*.mp4")) + 
                          list(video_dir.glob("*.MP4")) + 
                          list(video_dir.glob("*.avi")) + 
                          list(video_dir.glob("*.AVI")) +
                          list(video_dir.glob("*.mov")) +
                          list(video_dir.glob("*.MOV")))
        else:
            print(f"❌ Directory not found: {video_dir}")
            return
    
    # Mặc định tìm trong resource folder
    else:
        video_dir = Path("resource")
        if not video_dir.exists():
            video_dir = Path("examples/test/resource")
        
        video_files = (list(video_dir.glob("*.mp4")) + 
                      list(video_dir.glob("*.MP4")) + 
                      list(video_dir.glob("*.avi")) + 
                      list(video_dir.glob("*.AVI")) +
                      list(video_dir.glob("*.mov")) +
                      list(video_dir.glob("*.MOV")))
    
    if not video_files:
        print(f"\n❌ No video files found!")
        print(f"\n📖 Usage:")
        print(f"   1. Analyze specific video(s):")
        print(f"      python test_video_analysis_with_clips.py video1.mp4 video2.mp4")
        print(f"\n   2. Analyze all videos in a directory:")
        print(f"      python test_video_analysis_with_clips.py --dir path/to/videos")
        print(f"\n   3. Add videos to default resource folder:")
        print(f"      Copy videos to: examples/test/resource/")
        print(f"      Then run: python test_video_analysis_with_clips.py")
        print(f"\n   Supported formats: .mp4, .avi, .mov")
        return
    
    print(f"\n📁 Found {len(video_files)} video(s) to analyze:")
    for i, video_file in enumerate(video_files, 1):
        print(f"   {i}. {video_file.name}")
    
    # Create analyzer
    analyzer = VideoEventAnalyzer(output_base_dir=args.output)
    
    # Analyze each video
    all_results = []
    
    for video_file in video_files:
        try:
            result = analyzer.analyze_video(video_file)
            if result:
                analyzer.print_summary()
                all_results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {video_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall summary
    print("\n" + "="*60)
    print("📊 OVERALL SUMMARY")
    print("="*60)
    print(f"Total videos analyzed: {len(all_results)}")
    total_falls = sum(len(r['fall_events']) for r in all_results)
    total_seizures = sum(len(r['seizure_events']) for r in all_results)
    print(f"Total fall events: {total_falls}")
    print(f"Total seizure events: {total_seizures}")
    print("="*60)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: test_results/video_analysis/")
    print(f"   - Video clips with events")
    print(f"   - Statistics JSON files")


if __name__ == "__main__":
    main()
