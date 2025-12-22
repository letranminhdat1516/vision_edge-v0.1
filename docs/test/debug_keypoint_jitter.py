#!/usr/bin/env python3
"""
DEBUG KEYPOINT JITTER - Kiểm tra độ ổn định của keypoint detection
Khi người nằm yên, keypoints cũng phải yên (jitter < threshold)

Usage:
    python debug_keypoint_jitter.py --video 41 --start 300 --end 400
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator


def debug_keypoint_jitter(video_path: str, start_frame: int, end_frame: int):
    """Debug keypoint stability when person is lying still"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {video_path}")
    print(f"   Total frames: {total_frames}, FPS: {fps}")
    print(f"   Analyzing frames {start_frame} to {end_frame}")
    
    # Initialize pose estimator
    pose_estimator = YOLOv8PoseEstimator(model_size='n')
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Collect keypoints
    all_keypoints = []
    all_confidences = []
    frame_numbers = []
    
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    for frame_idx in range(start_frame, min(end_frame, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = pose_estimator.extract_keypoints(frame)
        
        if keypoints is not None:
            all_keypoints.append(keypoints[:, :2])  # x, y only
            all_confidences.append(keypoints[:, 2])  # confidence
            frame_numbers.append(frame_idx)
        else:
            print(f"   Frame {frame_idx}: No keypoints detected")
    
    cap.release()
    
    if len(all_keypoints) < 2:
        print("❌ Not enough keypoints collected")
        return
    
    keypoints_array = np.array(all_keypoints)  # Shape: (N, 17, 2)
    confidences_array = np.array(all_confidences)  # Shape: (N, 17)
    
    print(f"\n📊 Collected {len(all_keypoints)} frames with keypoints")
    print(f"   Keypoints shape: {keypoints_array.shape}")
    
    # Calculate jitter (frame-to-frame movement)
    jitter = np.diff(keypoints_array, axis=0)  # Shape: (N-1, 17, 2)
    jitter_magnitude = np.sqrt(np.sum(jitter**2, axis=2))  # Shape: (N-1, 17)
    
    print("\n" + "="*70)
    print("🔍 KEYPOINT JITTER ANALYSIS (frame-to-frame movement)")
    print("="*70)
    print(f"{'Keypoint':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Conf':>8}")
    print("-"*70)
    
    high_jitter_joints = []
    
    for i, name in enumerate(KEYPOINT_NAMES):
        joint_jitter = jitter_magnitude[:, i]
        joint_conf = confidences_array[:, i]
        
        mean_jitter = np.mean(joint_jitter)
        std_jitter = np.std(joint_jitter)
        min_jitter = np.min(joint_jitter)
        max_jitter = np.max(joint_jitter)
        mean_conf = np.mean(joint_conf)
        
        # Flag high jitter
        flag = "⚠️" if mean_jitter > 3.0 else "  "
        if mean_jitter > 3.0:
            high_jitter_joints.append((name, mean_jitter, mean_conf))
        
        print(f"{flag}{name:<13} {mean_jitter:>8.2f} {std_jitter:>8.2f} {min_jitter:>8.2f} {max_jitter:>8.2f} {mean_conf:>8.2f}")
    
    print("-"*70)
    
    # Overall statistics
    overall_mean = np.mean(jitter_magnitude)
    overall_max = np.max(jitter_magnitude)
    
    print(f"\n📈 OVERALL JITTER:")
    print(f"   Mean: {overall_mean:.2f} px/frame")
    print(f"   Max: {overall_max:.2f} px/frame")
    
    if high_jitter_joints:
        print(f"\n⚠️ HIGH JITTER JOINTS (>3.0 px/frame):")
        for name, jitter, conf in high_jitter_joints:
            print(f"   - {name}: {jitter:.2f} px (conf={conf:.2f})")
    
    # Calculate what this jitter would look like as motion metrics
    print("\n" + "="*70)
    print("🎯 IMPACT ON SEIZURE DETECTION")
    print("="*70)
    
    # Simulate motion metrics with this jitter
    velocities = jitter_magnitude  # This IS the velocity (displacement per frame)
    mean_velocity = np.mean(velocities)
    
    # Calculate displacement over 10 frames (like in seizure detection)
    if len(keypoints_array) >= 10:
        displacement_10 = np.mean([
            np.sqrt(np.sum((keypoints_array[9, j, :] - keypoints_array[0, j, :])**2))
            for j in range(17)
        ])
        print(f"   Mean velocity: {mean_velocity:.2f} px/frame")
        print(f"   10-frame displacement: {displacement_10:.2f} px")
    
    # Check oscillation (sign changes in Y direction)
    y_velocities = jitter[:, :, 1]  # Y component only
    sign_changes = 0
    for j in range(17):
        for i in range(1, len(y_velocities)):
            if y_velocities[i-1, j] * y_velocities[i, j] < 0:
                sign_changes += 1
    
    oscillation_ratio = sign_changes / (17 * (len(y_velocities) - 1)) if len(y_velocities) > 1 else 0
    print(f"   Oscillation ratio: {oscillation_ratio:.2f}")
    
    print("\n💡 DIAGNOSIS:")
    if overall_mean > 3.0:
        print("   ⚠️ HIGH JITTER DETECTED!")
        print("   → Keypoints are unstable even when person is still")
        print("   → This will cause false positive seizure detection")
        print("\n   RECOMMENDATIONS:")
        print("   1. Add keypoint smoothing/filtering")
        print("   2. Require CONSISTENT high motion (not just any motion)")
        print("   3. Use higher confidence threshold for keypoints")
    elif overall_mean > 1.5:
        print("   ⚡ MODERATE JITTER")
        print("   → Some keypoint instability detected")
        print("   → May cause occasional false positives")
    else:
        print("   ✅ LOW JITTER")
        print("   → Keypoints are stable when person is still")
        print("   → Seizure detection should be reliable")
    
    # Visualize jitter over time
    print("\n📊 JITTER OVER TIME (first 50 frames):")
    print("-"*70)
    for i in range(min(50, len(jitter_magnitude))):
        frame_jitter = np.mean(jitter_magnitude[i])
        bar_len = int(frame_jitter * 2)
        bar = "█" * min(bar_len, 40)
        print(f"   Frame {frame_numbers[i]:4d}: {frame_jitter:5.1f} px |{bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='41', help='Video number')
    parser.add_argument('--start', type=int, default=300, help='Start frame (default: 300 = lying still)')
    parser.add_argument('--end', type=int, default=400, help='End frame')
    args = parser.parse_args()
    
    # Find video
    script_dir = Path(__file__).parent
    video_path = script_dir / "resource" / f"{args.video}.mp4"
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    debug_keypoint_jitter(str(video_path), args.start, args.end)


if __name__ == "__main__":
    main()
