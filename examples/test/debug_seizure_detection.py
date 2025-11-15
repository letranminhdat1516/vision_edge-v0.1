"""
Debug Seizure Detection - Kiểm tra chi tiết cơ chế phát hiện co giật
"""

import cv2
import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator
from seizure_detection.vsvig_detector import VSViGSeizureDetector
from seizure_detection.seizure_predictor import SeizurePredictor

def main():
    video_path = "resource/1.mp4"
    
    print("=" * 80)
    print("🔍 DEBUG SEIZURE DETECTION")
    print("=" * 80)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    pose_estimator = YOLOv8PoseEstimator(model_size='n')
    seizure_detector = VSViGSeizureDetector()
    seizure_predictor = SeizurePredictor(
        temporal_window=3,
        alert_threshold=0.70,
        warning_threshold=0.55
    )
    
    # Print configuration
    print(f"\n📊 Configuration:")
    print(f"   VSViGSeizureDetector.confidence_threshold: {seizure_detector.confidence_threshold}")
    print(f"   VSViGSeizureDetector.temporal_window: {seizure_detector.temporal_window}")
    print(f"   SeizurePredictor.alert_threshold: {seizure_predictor.alert_threshold}")
    print(f"   SeizurePredictor.warning_threshold: {seizure_predictor.warning_threshold}")
    print(f"   SeizurePredictor.temporal_window: {seizure_predictor.temporal_window}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {video_path}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Analyze first 300 frames
    print(f"\n🎬 Analyzing first 300 frames...")
    print("=" * 120)
    print(f"{'Frame':<8} {'Time':<8} {'Person':<8} {'Temporal':<10} {'Base_Conf':<12} {'Smoothed':<12} {'Alert':<10} {'Seizure?':<10}")
    print("=" * 120)
    
    seizure_count = 0
    
    for frame_num in range(1, 301):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_num / fps
        
        # Person detection
        keypoints = pose_estimator.extract_keypoints(frame, confidence_threshold=0.25)
        has_person = keypoints is not None
        
        if not has_person:
            print(f"{frame_num:<8} {current_time:<8.2f} {'NO':<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            continue
        
        # Extract bbox
        keypoints_xy = keypoints[:, :2]
        valid_points = keypoints_xy[keypoints[:, 2] > 0.3]
        
        if len(valid_points) == 0:
            print(f"{frame_num:<8} {current_time:<8.2f} {'YES':<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            continue
        
        x_min, y_min = valid_points.min(axis=0)
        x_max, y_max = valid_points.max(axis=0)
        padding = 20
        x_min = max(0, int(x_min) - padding)
        y_min = max(0, int(y_min) - padding)
        x_max = min(frame.shape[1], int(x_max) + padding)
        y_max = min(frame.shape[0], int(y_max) + padding)
        person_bbox = [x_min, y_min, x_max, y_max]
        
        # Seizure detection
        seizure_result = seizure_detector.detect_seizure(
            frame=frame,
            person_bbox=person_bbox
        )
        
        temporal_ready = seizure_result.get('temporal_ready', False)
        base_confidence = seizure_result.get('confidence', 0.0)
        
        # Predictor
        smoothed_confidence = 0.0
        alert_level = 'N/A'
        seizure_detected = False
        
        if temporal_ready:
            pred_result = seizure_predictor.update_prediction(base_confidence)
            smoothed_confidence = pred_result['smoothed_confidence']
            alert_level = pred_result['alert_level']
            seizure_detected = pred_result['seizure_detected']
            
            if seizure_detected:
                seizure_count += 1
        
        # Print row
        temporal_str = "READY" if temporal_ready else "BUFFER"
        base_conf_str = f"{base_confidence:.4f}" if temporal_ready else "N/A"
        smooth_str = f"{smoothed_confidence:.4f}" if temporal_ready else "N/A"
        seizure_str = "🚨 SEIZURE!" if seizure_detected else ""
        
        print(f"{frame_num:<8} {current_time:<8.2f} {'YES':<8} {temporal_str:<10} {base_conf_str:<12} {smooth_str:<12} {alert_level:<10} {seizure_str:<10}")
    
    cap.release()
    
    print("=" * 120)
    print(f"\n📊 RESULT: {seizure_count} seizure(s) detected in 300 frames")
    print("=" * 120)

if __name__ == "__main__":
    main()
