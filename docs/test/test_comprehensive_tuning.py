"""
Comprehensive Testing & Tuning for False Positive Reduction
Test kỹ càng từng video để điều chỉnh threshold giảm false positive
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


class ComprehensiveTuner:
    """Test và tune detection thresholds để giảm false positive"""
    
    def __init__(self, output_dir="test_results/tuning", show_video=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_video = show_video
        
        print("🔧 Initializing detection services...")
        
        # YOLO Pose
        self.pose_estimator = YOLOv8PoseEstimator(model_size='n')
        
        # Fall detector with current thresholds
        self.fall_detector = SimpleFallDetector(confidence_threshold=0.20)
        
        # Seizure detector with strict thresholds
        self.seizure_detector = VSViGSeizureDetector(confidence_threshold=0.70)
        self.seizure_predictor = SeizurePredictor(
            temporal_window=5,
            smoothing_factor=0.8,
            alert_threshold=0.90,
            warning_threshold=0.80
        )
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'fall_detections': [],
            'seizure_detections': [],
            'fall_confidences': [],
            'seizure_confidences': []
        }
        
        # Performance tracking
        self.performance = {
            'frame_times': [],
            'detection_times': [],
            'avg_fps': 0,
            'min_fps': 0,
            'max_fps': 0,
            'total_processing_time': 0
        }
        
        # Detailed frame log
        self.frame_log = []
        
    def analyze_video(self, video_path, video_name, expected_falls=0, expected_seizures=0):
        """
        Analyze video với ground truth để đánh giá false positive
        
        Args:
            video_path: Đường dẫn video
            video_name: Tên video
            expected_falls: Số fall thật trong video (ground truth)
            expected_seizures: Số seizure thật trong video (ground truth)
        """
        print(f"\n{'='*80}")
        print(f"📹 Testing Video: {video_name}")
        print(f"{'='*80}")
        print(f"Expected Falls: {expected_falls}")
        print(f"Expected Seizures: {expected_seizures}")
        print()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video Info:")
        print(f"  - FPS: {fps}")
        print(f"  - Total Frames: {total_frames}")
        print(f"  - Duration: {duration:.1f}s")
        print()
        
        # Reset stats
        self.stats = {
            'total_frames': 0,
            'frames_with_person': 0,
            'fall_detections': [],
            'seizure_detections': [],
            'fall_confidences': [],
            'seizure_confidences': []
        }
        self.frame_log = []
        
        # Cooldown để tránh duplicate detection
        fall_cooldown_until = -1
        seizure_cooldown_until = -1
        fall_cooldown_frames = fps * 3  # 3 giây
        seizure_cooldown_frames = fps * 5  # 5 giây
        
        start_time = time.time()
        frame_idx = 0
        
        # Progress tracking
        last_progress = -1
        
        # Performance tracking
        last_fps_print = time.time()
        recent_frame_times = []
        
        # Create window if showing video
        if self.show_video:
            window_name = f"Detection Test - {video_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Get video dimensions to maintain aspect ratio
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Scale to fit screen (max 1280x720) while maintaining aspect ratio
            max_width = 1280
            max_height = 720
            scale = min(max_width / frame_width, max_height / frame_height, 1.0)
            display_width = int(frame_width * scale)
            display_height = int(frame_height * scale)
            
            cv2.resizeWindow(window_name, display_width, display_height)
            print(f"📺 Display: {frame_width}x{frame_height} → {display_width}x{display_height}")
        
        while True:
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            self.stats['total_frames'] = frame_idx
            timestamp = frame_idx / fps
            
            # Create display frame
            display_frame = frame.copy()
            
            # Extract keypoints for ALL persons in frame
            detection_start = time.time()
            all_persons = self.pose_estimator.extract_all_keypoints(frame, confidence_threshold=0.25)
            detection_time = time.time() - detection_start
            self.performance['detection_times'].append(detection_time)
            
            # Check if any person detected
            person_detected = len(all_persons) > 0
            
            # Calculate frame processing time and FPS
            frame_time = time.time() - frame_start_time
            self.performance['frame_times'].append(frame_time)
            recent_frame_times.append(frame_time)
            if len(recent_frame_times) > 30:  # Keep last 30 frames
                recent_frame_times.pop(0)
            
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            avg_recent_fps = len(recent_frame_times) / sum(recent_frame_times) if sum(recent_frame_times) > 0 else 0
            
            # Progress display with FPS
            progress = int((frame_idx / total_frames) * 100)
            if progress % 10 == 0 and progress != last_progress:
                print(f"⏳ Progress: {progress}% ({frame_idx}/{total_frames} frames) | FPS: {avg_recent_fps:.1f} | Detection: {detection_time*1000:.1f}ms")
                last_progress = progress
            
            # Print FPS every 5 seconds
            if time.time() - last_fps_print > 5.0:
                print(f"   ⚡ Current FPS: {current_fps:.1f} | Avg FPS: {avg_recent_fps:.1f} | Frame time: {frame_time*1000:.1f}ms")
                last_fps_print = time.time()
            
            if not person_detected:
                # No person detected
                self.frame_log.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'person_detected': False,
                    'fall_conf': 0.0,
                    'fall_detected': False,
                    'seizure_base_conf': 0.0,
                    'seizure_smoothed_conf': 0.0,
                    'seizure_detected': False
                })
                
                # Draw info on frame
                if self.show_video:
                    self._draw_info(display_frame, frame_idx, timestamp, 0, 0.0, False, 0.0, 0.0, False, 
                                   len(self.stats['fall_detections']), len(self.stats['seizure_detections']),
                                   avg_recent_fps, detection_time)
                    cv2.imshow(window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user")
                        break
                continue
            
            # Persons detected! 
            self.stats['frames_with_person'] += 1
            num_persons = len(all_persons)
            
            # Process each person and find highest fall/seizure confidence
            max_fall_conf = 0.0
            max_fall_detected = False
            max_fall_method = 'none'
            max_seizure_conf = 0.0
            max_seizure_detected = False
            max_seizure_smoothed = 0.0
            
            # Draw all persons and process detections
            for person_idx, person_data in enumerate(all_persons):
                keypoints = person_data['keypoints']  # (17, 3)
                person_bbox = person_data['bbox']      # [x1, y1, x2, y2]
                person_conf = person_data['confidence']
                
                # Draw bounding box
                if person_bbox:
                    x1, y1, x2, y2 = map(int, person_bbox)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {person_idx+1} ({person_conf:.2f})"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Capture will be done when event detected (see below)
                
                # Fall detection for this person
                fall_result = self.fall_detector.detect_fall(frame, timestamp=timestamp, person_bbox=person_bbox)
                fall_conf = fall_result.get('confidence', 0.0)
                fall_detected = fall_result.get('fall_detected', False)
                
                if fall_conf > max_fall_conf:
                    max_fall_conf = fall_conf
                    max_fall_detected = fall_detected
                    max_fall_method = fall_result.get('method', 'none')
                
                # Seizure detection for this person
                seizure_result = self.seizure_detector.detect_seizure(frame, keypoints)
                seizure_base_conf = seizure_result.get('confidence', 0.0)
                
                # Temporal smoothing
                temporal_ready = seizure_result.get('temporal_ready', False)
                if temporal_ready:
                    pred_result = self.seizure_predictor.update_prediction(seizure_base_conf)
                    seizure_smoothed_conf = pred_result['smoothed_confidence']
                    seizure_alert_level = pred_result['alert_level']
                    seizure_detected = seizure_alert_level in ['seizure', 'critical']
                    
                    if seizure_smoothed_conf > max_seizure_smoothed:
                        max_seizure_smoothed = seizure_smoothed_conf
                        max_seizure_detected = seizure_detected
                
                if seizure_base_conf > max_seizure_conf:
                    max_seizure_conf = seizure_base_conf
            
            # Use maximum confidence across all persons
            fall_conf = max_fall_conf
            fall_detected = max_fall_detected
            seizure_base_conf = max_seizure_conf
            seizure_smoothed_conf = max_seizure_smoothed
            seizure_detected = max_seizure_detected
            
            self.stats['fall_confidences'].append(fall_conf)
            self.stats['seizure_confidences'].append(seizure_base_conf)
            
            # Temporal smoothing
            temporal_ready = seizure_result.get('temporal_ready', False)
            if temporal_ready:
                pred_result = self.seizure_predictor.update_prediction(seizure_base_conf)
                seizure_smoothed_conf = pred_result['smoothed_confidence']
                seizure_alert_level = pred_result['alert_level']
                seizure_detected = seizure_alert_level in ['seizure', 'critical']
            else:
                seizure_smoothed_conf = 0.0
                seizure_alert_level = 'normal'
                seizure_detected = False
            
            self.stats['seizure_confidences'].append(seizure_base_conf)
            
            # Record fall event (with cooldown)
            if max_fall_detected and frame_idx > fall_cooldown_until:
                self.stats['fall_detections'].append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'confidence': max_fall_conf,
                    'method': max_fall_method,
                    'num_persons': num_persons
                })
                fall_cooldown_until = frame_idx + fall_cooldown_frames
                
                # 📸 CAPTURE IMAGE WHEN FALL DETECTED
                event_folder = self.output_dir / "event_captures" / video_name / "falls"
                event_folder.mkdir(parents=True, exist_ok=True)
                
                # Save full frame with all annotations
                full_event_path = event_folder / f"fall_frame_{frame_idx:06d}_conf{max_fall_conf:.2f}.jpg"
                cv2.imwrite(str(full_event_path), display_frame)
                
                # Save each person involved
                for person_idx, person_data in enumerate(all_persons):
                    bbox = person_data['bbox']
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        person_img = frame[y1:y2, x1:x2].copy()
                        person_event_path = event_folder / f"fall_frame_{frame_idx:06d}_person{person_idx+1}.jpg"
                        cv2.imwrite(str(person_event_path), person_img)
                
                print(f"  🚨 FALL at {timestamp:.1f}s (frame {frame_idx}) - Conf: {max_fall_conf:.3f} - Method: {max_fall_method} - Persons: {num_persons}")
                print(f"     📸 Saved to: {full_event_path}")
            
            # Record seizure event (with cooldown)
            if max_seizure_detected and frame_idx > seizure_cooldown_until:
                self.stats['seizure_detections'].append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'base_conf': max_seizure_conf,
                    'smoothed_conf': max_seizure_smoothed,
                    'alert_level': 'seizure',
                    'num_persons': num_persons
                })
                seizure_cooldown_until = frame_idx + seizure_cooldown_frames
                
                # 📸 CAPTURE IMAGE WHEN SEIZURE DETECTED
                event_folder = self.output_dir / "event_captures" / video_name / "seizures"
                event_folder.mkdir(parents=True, exist_ok=True)
                
                # Save full frame with all annotations
                full_event_path = event_folder / f"seizure_frame_{frame_idx:06d}_conf{max_seizure_smoothed:.2f}.jpg"
                cv2.imwrite(str(full_event_path), display_frame)
                
                # Save each person involved
                for person_idx, person_data in enumerate(all_persons):
                    bbox = person_data['bbox']
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        person_img = frame[y1:y2, x1:x2].copy()
                        person_event_path = event_folder / f"seizure_frame_{frame_idx:06d}_person{person_idx+1}.jpg"
                        cv2.imwrite(str(person_event_path), person_img)
                
                print(f"  ⚡ SEIZURE at {timestamp:.1f}s (frame {frame_idx}) - Base: {max_seizure_conf:.3f}, Smoothed: {max_seizure_smoothed:.3f} - Persons: {num_persons}")
                print(f"     📸 Saved to: {full_event_path}")
            
            # Log frame details
            self.frame_log.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'person_detected': True,
                'fall_conf': fall_conf,
                'fall_detected': fall_detected and frame_idx > fall_cooldown_until,
                'seizure_base_conf': seizure_base_conf,
                'seizure_smoothed_conf': seizure_smoothed_conf,
                'seizure_alert_level': seizure_alert_level,
                'seizure_detected': seizure_detected and frame_idx > seizure_cooldown_until
            })
            
            # Draw detection info on frame
            if self.show_video:
                # Draw skeleton keypoints for ALL persons
                # COCO skeleton connections
                connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (5, 11), (6, 12), (11, 12),  # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                ]
                
                for person_data in all_persons:
                    keypoints = person_data['keypoints']
                    
                    # Draw skeleton lines
                    for start_idx, end_idx in connections:
                        if keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3:
                            pt1 = tuple(map(int, keypoints[start_idx, :2]))
                            pt2 = tuple(map(int, keypoints[end_idx, :2]))
                            cv2.line(display_frame, pt1, pt2, (0, 255, 255), 2)
                    
                    # Draw keypoint circles
                    for i, kp in enumerate(keypoints):
                        if kp[2] > 0.3:  # confidence threshold
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(display_frame, (x, y), 4, (0, 0, 255), -1)
                
                # Person count
                person_count = num_persons
                
                self._draw_info(display_frame, frame_idx, timestamp, person_count, 
                               fall_conf, fall_detected and frame_idx > fall_cooldown_until,
                               seizure_base_conf, seizure_smoothed_conf, 
                               seizure_detected and frame_idx > seizure_cooldown_until,
                               len(self.stats['fall_detections']), len(self.stats['seizure_detections']),
                               avg_recent_fps, detection_time)
                
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord(' '):  # Space to pause
                    print("\n⏸️  PAUSED - Press any key to continue...")
                    cv2.waitKey(0)
        
        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        processing_time = time.time() - start_time
        
        # Calculate results
        detected_falls = len(self.stats['fall_detections'])
        detected_seizures = len(self.stats['seizure_detections'])
        
        fall_fps = detected_falls - expected_falls if detected_falls > expected_falls else 0
        seizure_fps = detected_seizures - expected_seizures if detected_seizures > expected_seizures else 0
        
        fall_missed = expected_falls - detected_falls if detected_falls < expected_falls else 0
        seizure_missed = expected_seizures - detected_seizures if detected_seizures < expected_seizures else 0
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 RESULTS for {video_name}")
        print(f"{'='*80}")
        
        # Calculate performance metrics
        if self.performance['frame_times']:
            avg_frame_time = sum(self.performance['frame_times']) / len(self.performance['frame_times'])
            self.performance['avg_fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.performance['min_fps'] = 1.0 / max(self.performance['frame_times']) if max(self.performance['frame_times']) > 0 else 0
            self.performance['max_fps'] = 1.0 / min(self.performance['frame_times']) if min(self.performance['frame_times']) > 0 else 0
        
        if self.performance['detection_times']:
            avg_detection_time = sum(self.performance['detection_times']) / len(self.performance['detection_times'])
        else:
            avg_detection_time = 0
        
        self.performance['total_processing_time'] = processing_time
        
        print(f"\n⏱️  Processing Performance:")
        print(f"  - Total Time: {processing_time:.1f}s")
        print(f"  - Video FPS: {fps}")
        print(f"  - Processing FPS: {total_frames/processing_time:.1f}")
        print(f"  - Avg FPS: {self.performance['avg_fps']:.1f}")
        print(f"  - Min FPS: {self.performance['min_fps']:.1f}")
        print(f"  - Max FPS: {self.performance['max_fps']:.1f}")
        print(f"  - Avg Frame Time: {avg_frame_time*1000:.1f}ms")
        print(f"  - Avg Detection Time: {avg_detection_time*1000:.1f}ms")
        print(f"  - Real-time Factor: {(total_frames/processing_time)/fps:.2f}x")
        print(f"\n👤 Person Detection:")
        print(f"  - Frames with person: {self.stats['frames_with_person']}/{total_frames} ({self.stats['frames_with_person']/total_frames*100:.1f}%)")
        
        print(f"\n🚨 FALL Detection:")
        print(f"  - Expected: {expected_falls}")
        print(f"  - Detected: {detected_falls}")
        print(f"  - False Positives: {fall_fps} {'✅' if fall_fps == 0 else '❌'}")
        print(f"  - Missed (False Negatives): {fall_missed} {'✅' if fall_missed == 0 else '❌'}")
        if self.stats['fall_confidences']:
            avg_fall_conf = sum(self.stats['fall_confidences']) / len(self.stats['fall_confidences'])
            max_fall_conf = max(self.stats['fall_confidences'])
            print(f"  - Avg Confidence: {avg_fall_conf:.3f}")
            print(f"  - Max Confidence: {max_fall_conf:.3f}")
        
        print(f"\n⚡ SEIZURE Detection:")
        print(f"  - Expected: {expected_seizures}")
        print(f"  - Detected: {detected_seizures}")
        print(f"  - False Positives: {seizure_fps} {'✅' if seizure_fps == 0 else '❌'}")
        print(f"  - Missed (False Negatives): {seizure_missed} {'✅' if seizure_missed == 0 else '❌'}")
        if self.stats['seizure_confidences']:
            avg_seizure_conf = sum(self.stats['seizure_confidences']) / len(self.stats['seizure_confidences'])
            max_seizure_conf = max(self.stats['seizure_confidences'])
            print(f"  - Avg Base Confidence: {avg_seizure_conf:.3f}")
            print(f"  - Max Base Confidence: {max_seizure_conf:.3f}")
        
        # Detailed detection list
        if self.stats['fall_detections']:
            print(f"\n📋 Fall Detection Details:")
            for i, fall in enumerate(self.stats['fall_detections'], 1):
                print(f"  {i}. Frame {fall['frame']} ({fall['timestamp']:.1f}s) - Conf: {fall['confidence']:.3f} - Method: {fall['method']}")
        
        if self.stats['seizure_detections']:
            print(f"\n📋 Seizure Detection Details:")
            for i, seizure in enumerate(self.stats['seizure_detections'], 1):
                print(f"  {i}. Frame {seizure['frame']} ({seizure['timestamp']:.1f}s) - Base: {seizure['base_conf']:.3f}, Smoothed: {seizure['smoothed_conf']:.3f}")
        
        # Overall assessment
        print(f"\n{'='*80}")
        total_fps = fall_fps + seizure_fps
        if total_fps == 0 and fall_missed == 0 and seizure_missed == 0:
            print("✅ PERFECT! No false positives, no missed detections")
        elif total_fps == 0:
            print(f"✅ NO FALSE POSITIVES! (but {fall_missed + seizure_missed} missed)")
        else:
            print(f"⚠️  {total_fps} FALSE POSITIVES DETECTED - Need tuning!")
        print(f"{'='*80}\n")
        
        # Save detailed report
        self._save_report(video_name, {
            'video_name': video_name,
            'video_path': str(video_path),
            'expected': {
                'falls': expected_falls,
                'seizures': expected_seizures
            },
            'detected': {
                'falls': detected_falls,
                'seizures': detected_seizures
            },
            'false_positives': {
                'falls': fall_fps,
                'seizures': seizure_fps,
                'total': total_fps
            },
            'false_negatives': {
                'falls': fall_missed,
                'seizures': seizure_missed
            },
            'statistics': self.stats,
            'performance': self.performance,
            'processing_time': processing_time,
            'frame_log': self.frame_log
        })
        
        return {
            'fall_fps': fall_fps,
            'seizure_fps': seizure_fps,
            'total_fps': total_fps,
            'fall_missed': fall_missed,
            'seizure_missed': seizure_missed
        }
    
    def _draw_info(self, frame, frame_idx, timestamp, person_count, fall_conf, fall_detected, 
                   seizure_base_conf, seizure_smoothed_conf, seizure_detected, total_falls, total_seizures,
                   current_fps=0, detection_time=0):
        """Draw detection information on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay (larger for performance info)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 260), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 35
        
        # Frame info with FPS
        cv2.putText(frame, f"Frame: {frame_idx} | Time: {timestamp:.1f}s | FPS: {current_fps:.1f}", (20, y), font, 0.5, (255, 255, 255), 2)
        y += 25
        
        # Performance info
        cv2.putText(frame, f"Detection: {detection_time*1000:.1f}ms", (20, y), font, 0.5, (255, 255, 0), 1)
        y += 30
        
        # Person count
        color = (0, 255, 0) if person_count > 0 else (128, 128, 128)
        cv2.putText(frame, f"Person: {person_count}", (20, y), font, 0.7, color, 2)
        y += 35
        
        # Fall detection
        if fall_detected:
            color = (0, 0, 255)
            text = f"FALL DETECTED! Conf: {fall_conf:.3f} [Total: {total_falls}]"
        else:
            color = (100, 100, 100) if fall_conf < 0.1 else (150, 150, 0)
            text = f"Fall: {fall_conf:.3f} [Total: {total_falls}]"
        cv2.putText(frame, text, (20, y), font, 0.6, color, 2)
        y += 30
        
        # Seizure detection
        if seizure_detected:
            color = (0, 0, 255)
            text = f"SEIZURE DETECTED! [Total: {total_seizures}]"
        else:
            color = (100, 100, 100) if seizure_base_conf < 0.1 else (150, 150, 0)
            text = f"Seizure Base: {seizure_base_conf:.3f}"
        cv2.putText(frame, text, (20, y), font, 0.6, color, 2)
        y += 25
        
        if seizure_smoothed_conf > 0:
            color = (0, 255, 255) if seizure_smoothed_conf >= 0.80 else (100, 150, 150)
            cv2.putText(frame, f"Smoothed: {seizure_smoothed_conf:.3f} [Total: {total_seizures}]", 
                       (20, y), font, 0.6, color, 2)
        y += 35
        
        # Controls
        cv2.putText(frame, "Press 'Q' to quit | 'SPACE' to pause", (20, y), font, 0.5, (200, 200, 200), 1)
        
        # Status in corner
        status_y = h - 30
        if fall_detected or seizure_detected:
            cv2.rectangle(frame, (w-250, status_y-25), (w-10, status_y+5), (0, 0, 255), -1)
            cv2.putText(frame, "!!! ALERT !!!", (w-230, status_y), font, 0.8, (255, 255, 255), 2)
    
    def _save_report(self, video_name, report_data):
        """Save detailed report to JSON"""
        report_file = self.output_dir / f"{video_name}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Report saved: {report_file}")


def main():
    """Main test function"""
    print("="*80)
    print("🧪 COMPREHENSIVE TESTING & TUNING FOR FALSE POSITIVE REDUCTION")
    print("="*80)
    print()
    
    tuner = ComprehensiveTuner()
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    resource_dir = script_dir / "resource"
    
    # Define test videos with ground truth
    # TODO: Update với ground truth thực tế từ video của bạn
    test_videos = [
        {
            'path': str(resource_dir / '1.mp4'),
            'name': 'video_1',
            'expected_falls': 1,  # Số fall thật trong video
            'expected_seizures': 0  # Số seizure thật trong video
        },
        # Thêm các video khác ở đây
        # {
        #     'path': str(resource_dir / '2.mp4'),
        #     'name': 'video_2',
        #     'expected_falls': 2,
        #     'expected_seizures': 1
        # },
    ]
    
    # Run tests
    overall_results = {
        'total_videos': 0,
        'total_fall_fps': 0,
        'total_seizure_fps': 0,
        'total_fall_missed': 0,
        'total_seizure_missed': 0
    }
    
    for video in test_videos:
        result = tuner.analyze_video(
            video['path'],
            video['name'],
            video['expected_falls'],
            video['expected_seizures']
        )
        
        if result:
            overall_results['total_videos'] += 1
            overall_results['total_fall_fps'] += result['fall_fps']
            overall_results['total_seizure_fps'] += result['seizure_fps']
            overall_results['total_fall_missed'] += result['fall_missed']
            overall_results['total_seizure_missed'] += result['seizure_missed']
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Videos Tested: {overall_results['total_videos']}")
    print(f"Total Fall False Positives: {overall_results['total_fall_fps']}")
    print(f"Total Seizure False Positives: {overall_results['total_seizure_fps']}")
    print(f"Total False Positives: {overall_results['total_fall_fps'] + overall_results['total_seizure_fps']}")
    print(f"Total Fall Missed: {overall_results['total_fall_missed']}")
    print(f"Total Seizure Missed: {overall_results['total_seizure_missed']}")
    
    if overall_results['total_fall_fps'] + overall_results['total_seizure_fps'] == 0:
        print(f"\n✅ EXCELLENT! Zero false positives across all videos!")
    else:
        print(f"\n⚠️  Need threshold tuning to reduce false positives")
        print(f"\n💡 Recommendations:")
        if overall_results['total_fall_fps'] > 0:
            print(f"  - Increase fall detection thresholds (confidence_threshold, vertical_movement, etc.)")
        if overall_results['total_seizure_fps'] > 0:
            print(f"  - Increase seizure detection thresholds (confidence_threshold, alert_threshold, etc.)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
