"""
Test đơn giản fall detection với in ra tất cả thông số
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


def test_simple():
    """Test đơn giản"""
    video_path = "resource/1.mp4"
    
    print("🔍 Testing Fall Detection...")
    print("="*80)
    
    # Initialize
    pose_estimator = YOLOv8PoseEstimator(model_size='n')
    fall_detector = SimpleFallDetector(confidence_threshold=0.1)  # Very low for testing
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"\n{'Frame':<8} {'Time':<8} {'Has Person':<12} {'Aspect Ratio':<15} {'Bbox Size':<20} {'Confidence':<12} {'Method':<20} {'FALL?':<10}")
    print("-"*120)
    
    frame_count = 0
    fall_count = 0
    
    while frame_count < 300:  # First 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # Detect person
        keypoints = pose_estimator.extract_keypoints(frame, confidence_threshold=0.25)
        
        has_person = "YES" if keypoints is not None else "NO"
        
        if keypoints is not None:
            # Get bbox
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
                
                # Calculate aspect ratio
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                bbox_size = f"{bbox_w}x{bbox_h}"
                
                # Fall detection
                fall_result = fall_detector.detect_fall(
                    current_frame=frame,
                    timestamp=current_time,
                    person_bbox=person_bbox
                )
                
                confidence = fall_result.get('confidence', 0.0)
                method = fall_result.get('method', 'none')
                fall_detected = fall_result.get('fall_detected', False)
                
                if fall_detected:
                    fall_count += 1
                
                # Print every frame with person
                fall_status = "🚨 FALL!" if fall_detected else ""
                print(f"{frame_count:<8} {current_time:<8.2f} {has_person:<12} {aspect_ratio:<15.3f} {bbox_size:<20} {confidence:<12.3f} {method:<20} {fall_status:<10}")
            else:
                print(f"{frame_count:<8} {current_time:<8.2f} {has_person:<12} {'N/A':<15} {'N/A':<20} {'0.000':<12} {'no_valid_points':<20}")
        else:
            print(f"{frame_count:<8} {current_time:<8.2f} {has_person:<12} {'N/A':<15} {'N/A':<20} {'0.000':<12} {'no_person':<20}")
        
        frame_count += 1
    
    cap.release()
    
    print("\n" + "="*80)
    print(f"📊 RESULT: {fall_count} fall(s) detected in {frame_count} frames")
    print("="*80)


if __name__ == "__main__":
    test_simple()
