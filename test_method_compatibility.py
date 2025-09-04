#!/usr/bin/env python3
"""
Test YOLOv8 Pose Estimator method compatibility
Tests both calling patterns: with confidence threshold and with person_bbox
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator

def test_method_compatibility():
    """Test both calling patterns for extract_keypoints"""
    
    print("🧪 Testing YOLOv8PoseEstimator method compatibility...")
    
    # Initialize estimator
    estimator = YOLOv8PoseEstimator()
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (400, 400), (255, 255, 255), -1)  # White rectangle (person-like shape)
    
    print("\n📊 Testing different calling patterns:")
    
    # Test 1: Normal confidence threshold call
    print("\n1️⃣ Testing: extract_keypoints(frame, confidence_threshold=0.4)")
    keypoints1 = estimator.extract_keypoints(frame, confidence_threshold=0.4)
    print(f"   Result: {keypoints1 is not None} ({'Success' if keypoints1 is not None else 'No detection'})")
    
    # Test 2: Positional confidence threshold call
    print("\n2️⃣ Testing: extract_keypoints(frame, 0.4)")
    keypoints2 = estimator.extract_keypoints(frame, 0.4)
    print(f"   Result: {keypoints2 is not None} ({'Success' if keypoints2 is not None else 'No detection'})")
    
    # Test 3: VSViG-style call with person_bbox (compatibility mode)
    print("\n3️⃣ Testing: extract_keypoints(frame, person_bbox) - VSViG compatibility")
    person_bbox = [200, 100, 400, 400]  # [x1, y1, x2, y2]
    keypoints3 = estimator.extract_keypoints(frame, person_bbox)
    print(f"   Result: {keypoints3 is not None} ({'Success' if keypoints3 is not None else 'No detection'})")
    
    # Test 4: Mixed parameters
    print("\n4️⃣ Testing: extract_keypoints(frame, 0.3, person_bbox)")
    keypoints4 = estimator.extract_keypoints(frame, 0.3, person_bbox)
    print(f"   Result: {keypoints4 is not None} ({'Success' if keypoints4 is not None else 'No detection'})")
    
    print(f"\n📈 Performance Stats:")
    print(f"   Total detections: {estimator.total_detections}")
    print(f"   Successful detections: {estimator.successful_detections}")
    if estimator.total_detections > 0:
        success_rate = (estimator.successful_detections / estimator.total_detections) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average inference time: {estimator.avg_inference_time*1000:.1f}ms")
    
    print("\n✅ Compatibility test completed!")

if __name__ == "__main__":
    test_method_compatibility()
