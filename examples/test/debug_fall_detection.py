"""
Debug Fall Detection - Chi tiết phân tích fall detection trên 1 video
"""

import cv2
import sys
import numpy as np
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator
from fall_detection.simple_fall_detector import SimpleFallDetector


def analyze_fall_in_video(video_path, start_frame=0, num_frames=300):
    """
    Phân tích chi tiết fall detection trong video
    
    Args:
        video_path: Đường dẫn video
        start_frame: Frame bắt đầu
        num_frames: Số frames phân tích
    """
    print("="*70)
    print("🔍 DEBUG FALL DETECTION")
    print("="*70)
    
    # Initialize
    pose_estimator = YOLOv8PoseEstimator(model_size='n')
    fall_detector = SimpleFallDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Analyzing frames: {start_frame} to {start_frame + num_frames}")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fall_events = []
    frame_count = start_frame
    
    print(f"\n{'Frame':<8} {'Time':<8} {'Person':<8} {'Method':<20} {'Confidence':<12} {'Status':<10}")
    print("-"*70)
    
    while frame_count < start_frame + num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # Detect person
        keypoints = pose_estimator.extract_keypoints(frame, confidence_threshold=0.25)
        
        if keypoints is not None:
            # Extract bbox
            keypoints_xy = keypoints[:, :2]
            valid_points = keypoints_xy[keypoints[:, 2] > 0.3]
            
            if len(valid_points) > 0:
                x_min, y_min = valid_points.min(axis=0)
                x_max, y_max = valid_points.max(axis=0)
                
                padding = 20
                x_min = max(0, int(x_min) - padding)
                y_min = max(0, int(y_min) - padding)
                x_max = min(frame.shape[1], int(x_max) + padding)
                y_max = min(frame.shape[0], int(y_max) + padding)
                
                person_bbox = [x_min, y_min, x_max, y_max]
                
                # Fall detection
                fall_result = fall_detector.detect_fall(
                    current_frame=frame,
                    timestamp=current_time,
                    person_bbox=person_bbox
                )
                
                confidence = fall_result.get('confidence', 0.0)
                method = fall_result.get('method', 'unknown')
                fall_detected = fall_result.get('fall_detected', False)
                
                # Get bbox info for debugging
                bbox_w = person_bbox[2] - person_bbox[0]
                bbox_h = person_bbox[3] - person_bbox[1]
                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                
                # Debug output
                status = "🚨 FALL!" if fall_detected else ""
                person_status = "✓"
                
                # Print ALL frames with person detection for debugging
                if confidence > 0.05 or fall_detected:  # Lowered to see more
                    print(f"{frame_count:<8} {current_time:<8.2f} {person_status:<8} "
                          f"{method:<20} {confidence:<12.3f} {status:<10} "
                          f"AR={aspect_ratio:.2f} bbox=({bbox_w}x{bbox_h})")
                
                if fall_detected:
                    fall_events.append({
                        'frame': frame_count,
                        'time': current_time,
                        'confidence': confidence,
                        'method': method
                    })
        
        frame_count += 1
    
    cap.release()
    
    # Summary
    print("\n" + "="*70)
    print(f"📊 SUMMARY")
    print("="*70)
    print(f"Analyzed frames: {num_frames}")
    print(f"Fall events detected: {len(fall_events)}")
    
    if fall_events:
        print(f"\n🚨 Fall Events:")
        for i, event in enumerate(fall_events, 1):
            print(f"   {i}. Frame {event['frame']} ({event['time']:.2f}s) - "
                  f"Confidence: {event['confidence']:.3f} - Method: {event['method']}")
    else:
        print(f"\n✓ No fall events detected in analyzed frames")
    
    print("\n" + "="*70)
    
    return fall_events


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug fall detection trong video')
    parser.add_argument('video', help='Đường dẫn video')
    parser.add_argument('--start', type=int, default=0, help='Frame bắt đầu (default: 0)')
    parser.add_argument('--frames', type=int, default=300, help='Số frames phân tích (default: 300)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    analyze_fall_in_video(video_path, args.start, args.frames)


if __name__ == "__main__":
    main()
