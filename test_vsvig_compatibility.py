#!/usr/bin/env python3
"""
Test VSViG-style call pattern for YOLOv8PoseEstimator
This mimics how VSViGDetector calls extract_keypoints
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator

def test_vsvig_style_call():
    """Test the exact call pattern used by VSViGDetector"""
    
    print("🧪 Testing VSViG-style call pattern...")
    
    # Initialize estimator
    estimator = YOLOv8PoseEstimator()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
        
    print("📹 Press 'q' to quit, 'space' to test keypoint extraction")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Test extraction every 30 frames or on spacebar
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') or frame_count % 30 == 0:
            # Simulate person detection bbox (like VSViGDetector would do)
            h, w = frame.shape[:2]
            person_bbox = [w//4, h//4, 3*w//4, 3*h//4]  # Center region
            
            print(f"\n🔍 Testing VSViG-style call: extract_keypoints(frame, person_bbox)")
            print(f"   person_bbox: {person_bbox}")
            
            # This is exactly how VSViGDetector calls it
            keypoints = estimator.extract_keypoints(frame, person_bbox)
            
            if keypoints is not None:
                print(f"✅ Success! Got keypoints shape: {keypoints.shape}")
                print(f"📊 Performance: {estimator.total_detections} detections, {estimator.successful_detections} successful")
                
                # Visualize keypoints using YOLOv8PoseEstimator's own method
                frame_with_pose = estimator.visualize_pose(frame, keypoints)
                cv2.imshow('YOLOv8 Pose - VSViG Compatible', frame_with_pose)
                
                # Show confidence for some key points
                print("📍 Key points confidence:")
                if len(keypoints) >= 17:
                    print(f"   Nose: {keypoints[0][2]:.2f}")
                    print(f"   Left shoulder: {keypoints[5][2]:.2f}")
                    print(f"   Right shoulder: {keypoints[6][2]:.2f}")
                    print(f"   Left hip: {keypoints[11][2]:.2f}")
                    print(f"   Right hip: {keypoints[12][2]:.2f}")
            else:
                print("❌ No keypoints detected")
        else:
            cv2.imshow('YOLOv8 Pose - VSViG Compatible', frame)
            
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📈 Final Performance Stats:")
    print(f"   Total attempts: {estimator.total_detections}")
    print(f"   Successful detections: {estimator.successful_detections}")
    if estimator.total_detections > 0:
        success_rate = (estimator.successful_detections / estimator.total_detections) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average inference time: {estimator.avg_inference_time*1000:.1f}ms")

if __name__ == "__main__":
    test_vsvig_style_call()
