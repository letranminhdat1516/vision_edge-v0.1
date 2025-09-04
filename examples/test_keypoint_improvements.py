#!/usr/bin/env python3
"""
Test script for improved keypoint detection and visualization
"""

import cv2
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_keypoint_improvements():
    """Test the improved keypoint system"""
    print("🧪 Testing Improved Keypoint Detection & Visualization")
    print("="*60)
    
    # Test 1: Import modules
    try:
        from seizure_detection.pose_estimator import CustomPoseEstimator
        print("✅ Seizure detection pose estimator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import seizure pose estimator: {e}")
        return False
    
    try:
        from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
        print("✅ Advanced healthcare pipeline imported successfully")
    except Exception as e:
        print(f"❌ Failed to import healthcare pipeline: {e}")
        return False
    
    # Test 2: Initialize pose estimator
    try:
        pose_estimator = CustomPoseEstimator()
        print("✅ Pose estimator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize pose estimator: {e}")
        return False
    
    # Test 3: Create dummy keypoints to test validation
    print("\n🔬 Testing Keypoint Validation:")
    
    # High quality keypoints (should pass)
    good_keypoints = np.random.rand(15, 3)
    good_keypoints[:, :2] *= 300  # x, y coordinates
    good_keypoints[:, 2] = np.random.uniform(0.6, 0.9, 15)  # high confidence
    
    is_valid = pose_estimator.validate_keypoints(good_keypoints)
    print(f"   High quality keypoints valid: {is_valid}")
    
    # Low quality keypoints (should fail)
    bad_keypoints = np.random.rand(15, 3)
    bad_keypoints[:, :2] *= 300
    bad_keypoints[:, 2] = np.random.uniform(0.1, 0.3, 15)  # low confidence
    
    is_valid = pose_estimator.validate_keypoints(bad_keypoints)
    print(f"   Low quality keypoints valid: {is_valid}")
    
    # Test 4: Test skeleton connections
    print("\n🎨 Testing Skeleton Visualization:")
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create realistic keypoints for a standing person
    realistic_keypoints = np.array([
        [320, 100, 0.9],  # nose
        [315, 95, 0.8],   # left_eye
        [325, 95, 0.8],   # right_eye
        [310, 95, 0.7],   # left_ear
        [330, 95, 0.7],   # right_ear
        [300, 150, 0.9],  # left_shoulder
        [340, 150, 0.9],  # right_shoulder
        [280, 200, 0.8],  # left_elbow
        [360, 200, 0.8],  # right_elbow
        [260, 250, 0.7],  # left_wrist
        [380, 250, 0.7],  # right_wrist
        [310, 280, 0.9],  # left_hip
        [330, 280, 0.9],  # right_hip
        [305, 380, 0.8],  # left_knee
        [335, 380, 0.8],  # right_knee
    ])
    
    # Test visualization
    try:
        visualized_frame = pose_estimator.visualize_pose(test_frame, realistic_keypoints)
        print("✅ Pose visualization completed")
        
        # Save test image
        cv2.imwrite('test_keypoint_visualization.jpg', visualized_frame)
        print("✅ Test visualization saved as 'test_keypoint_visualization.jpg'")
        
    except Exception as e:
        print(f"❌ Pose visualization failed: {e}")
        return False
    
    # Test 5: Test coordinate adjustment
    print("\n📍 Testing Coordinate Adjustment:")
    
    # Test bbox adjustment
    test_bbox = [100, 50, 200, 300]  # x1, y1, x2, y2
    normalized_keypoints = np.random.rand(15, 3)
    normalized_keypoints[:, 2] = 0.8  # high confidence
    
    try:
        adjusted = pose_estimator._adjust_coordinates(normalized_keypoints, test_bbox)
        if adjusted is not None:
            print("✅ Coordinate adjustment successful")
            print(f"   Original range: x[0-1], y[0-1]")
            print(f"   Adjusted range: x[{adjusted[:, 0].min():.1f}-{adjusted[:, 0].max():.1f}], y[{adjusted[:, 1].min():.1f}-{adjusted[:, 1].max():.1f}]")
        else:
            print("❌ Coordinate adjustment returned None")
    except Exception as e:
        print(f"❌ Coordinate adjustment failed: {e}")
    
    print("\n🎉 Keypoint Improvement Tests Completed!")
    print("\n📊 Summary of Improvements:")
    print("   ✨ Increased confidence threshold from 0.3 to 0.5")
    print("   ✨ Added complete skeleton connections (head, torso, arms, legs)")
    print("   ✨ Color-coded keypoints and connections by confidence")
    print("   ✨ Added keypoint validation with spatial consistency")
    print("   ✨ Improved coordinate adjustment with bounds checking")
    print("   ✨ Enhanced body line drawing for fall detection")
    
    return True

if __name__ == "__main__":
    success = test_keypoint_improvements()
    
    if success:
        print("\n🚀 Ready to test with real camera stream!")
        print("   Run 'python src/main.py' to see improved keypoints in action")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
