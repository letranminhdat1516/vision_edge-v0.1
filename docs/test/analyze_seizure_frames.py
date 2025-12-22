#!/usr/bin/env python3
"""
ANALYZE SEIZURE FRAMES - Phân tích motion metrics ở các frame cụ thể
Giúp hiểu rõ sự khác biệt giữa:
- Frames bình thường (1-100)
- Frames nằm/lật (100-300)
- Frames có co giật thật (900-1400)

Usage:
    python analyze_seizure_frames.py --video 41
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seizure_detection.yolov8_pose_estimator import YOLOv8PoseEstimator


def analyze_frames(video_path: str, frame_ranges: list):
    """Analyze motion metrics for specific frame ranges"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {video_path}")
    print(f"   Total frames: {total_frames}, FPS: {fps}")
    
    # Initialize pose estimator
    pose_estimator = YOLOv8PoseEstimator(model_size='n')
    
    # Buffer for temporal analysis
    keypoint_buffer = []
    BUFFER_SIZE = 15
    
    for start_frame, end_frame, label in frame_ranges:
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING: {label} (frames {start_frame}-{end_frame})")
        print(f"{'='*60}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        keypoint_buffer.clear()
        
        metrics_list = []
        
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints (returns single value)
            keypoints = pose_estimator.extract_keypoints(frame)
            
            if keypoints is not None:
                keypoint_buffer.append(keypoints)
                if len(keypoint_buffer) > BUFFER_SIZE:
                    keypoint_buffer.pop(0)
                
                # Calculate metrics when buffer is full
                if len(keypoint_buffer) >= 10:
                    metrics = calculate_motion_metrics(keypoint_buffer)
                    metrics['frame'] = frame_idx
                    metrics_list.append(metrics)
        
        # Print summary statistics
        if metrics_list:
            print_metrics_summary(metrics_list, label)
    
    cap.release()


def calculate_motion_metrics(keypoint_buffer):
    """Calculate all motion metrics from keypoint buffer"""
    
    keypoint_sequence = np.array(keypoint_buffer[-10:])  # Last 10 frames
    coords = keypoint_sequence[:, :, :2]  # (T, P, 2)
    num_frames = len(keypoint_sequence)
    
    # Velocities
    velocities = np.diff(coords, axis=0)  # (T-1, P, 2)
    vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=2))  # (T-1, P)
    
    # 1. Mean velocity
    mean_velocity = np.mean(vel_magnitudes)
    max_velocity = np.max(vel_magnitudes)
    
    # 2. Total displacement
    total_displacement = np.mean([
        np.sqrt(np.sum((coords[-1, j, :] - coords[0, j, :])**2))
        for j in range(min(17, coords.shape[1]))
    ])
    
    # 3. Oscillation analysis
    important_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    upper_joints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    lower_joints = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    
    oscillation_counts = []
    sign_change_per_frame = [0] * (num_frames - 2)
    
    for joint in important_joints:
        if joint < coords.shape[1]:
            y_vel = velocities[:, joint, 1]
            sign_changes = 0
            
            for i in range(1, len(y_vel)):
                if y_vel[i-1] * y_vel[i] < 0:
                    sign_changes += 1
                    if i - 1 < len(sign_change_per_frame):
                        sign_change_per_frame[i-1] += 1
            
            oscillation_counts.append(sign_changes)
    
    mean_oscillation = np.mean(oscillation_counts) if oscillation_counts else 0
    oscillation_ratio = mean_oscillation / max(num_frames - 1, 1)
    osc_variance = np.var(oscillation_counts) if len(oscillation_counts) > 1 else 0
    
    # 4. Synchronization
    max_sync = max(sign_change_per_frame) if sign_change_per_frame else 0
    high_sync_frames = sum(1 for x in sign_change_per_frame if x >= 6)
    sync_ratio = high_sync_frames / max(len(sign_change_per_frame), 1)
    
    # 5. Upper vs Lower body activity
    upper_activity = np.mean([vel_magnitudes[:, j].mean() for j in upper_joints if j < vel_magnitudes.shape[1]])
    lower_activity = np.mean([vel_magnitudes[:, j].mean() for j in lower_joints if j < vel_magnitudes.shape[1]])
    
    if max(upper_activity, lower_activity) > 0:
        body_balance = min(upper_activity, lower_activity) / max(upper_activity, lower_activity)
    else:
        body_balance = 0
    
    # 6. Frequency estimation (cycles per 10 frames)
    # At 30fps, 10 frames = 0.33 seconds
    # If oscillation_ratio = 0.5, that's 5 direction changes in 10 frames
    # = 2.5 cycles in 0.33 seconds = 7.5 Hz
    estimated_freq_hz = (mean_oscillation / 2) / (10 / 30)  # cycles per second
    
    return {
        'mean_vel': mean_velocity,
        'max_vel': max_velocity,
        'displacement': total_displacement,
        'osc_ratio': oscillation_ratio,
        'osc_var': osc_variance,
        'sync_ratio': sync_ratio,
        'body_balance': body_balance,
        'upper_activity': upper_activity,
        'lower_activity': lower_activity,
        'est_freq_hz': estimated_freq_hz,
    }


def print_metrics_summary(metrics_list, label):
    """Print summary statistics for a range of frames"""
    
    if not metrics_list:
        print("No metrics collected")
        return
    
    keys = ['mean_vel', 'max_vel', 'displacement', 'osc_ratio', 'osc_var', 
            'sync_ratio', 'body_balance', 'upper_activity', 'lower_activity', 'est_freq_hz']
    
    print(f"\n📈 Summary for {label} ({len(metrics_list)} samples):")
    print("-" * 60)
    print(f"{'Metric':<15} {'Min':>10} {'Mean':>10} {'Max':>10} {'Std':>10}")
    print("-" * 60)
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        print(f"{key:<15} {min(values):>10.2f} {np.mean(values):>10.2f} {max(values):>10.2f} {np.std(values):>10.2f}")
    
    print("-" * 60)
    
    # Key insights
    print("\n🔑 Key Observations:")
    avg_osc = np.mean([m['osc_ratio'] for m in metrics_list])
    avg_sync = np.mean([m['sync_ratio'] for m in metrics_list])
    avg_disp = np.mean([m['displacement'] for m in metrics_list])
    avg_balance = np.mean([m['body_balance'] for m in metrics_list])
    avg_freq = np.mean([m['est_freq_hz'] for m in metrics_list])
    
    print(f"   - Oscillation ratio: {avg_osc:.2f} (>0.5 = high frequency motion)")
    print(f"   - Sync ratio: {avg_sync:.2f} (>0.3 = synchronized tremor)")
    print(f"   - Displacement: {avg_disp:.1f}px (>50 = large movement)")
    print(f"   - Body balance: {avg_balance:.2f} (>0.3 = whole body motion)")
    print(f"   - Estimated frequency: {avg_freq:.1f} Hz (seizure = 3-8 Hz)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='41', help='Video number')
    args = parser.parse_args()
    
    # Find video
    script_dir = Path(__file__).parent
    video_path = script_dir / "resource" / f"{args.video}.mp4"
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Define frame ranges to analyze
    frame_ranges = [
        (1, 50, "STANDING/WALKING"),
        (100, 200, "SITTING DOWN / LYING DOWN"),
        (300, 400, "LYING STILL"),
        (500, 600, "MOVING ON FLOOR"),
        (900, 1000, "SEIZURE START (if any)"),
        (1000, 1200, "SEIZURE MIDDLE (if any)"),
        (1200, 1400, "SEIZURE END (if any)"),
        (2000, 2100, "POST SEIZURE (if any)"),
    ]
    
    analyze_frames(str(video_path), frame_ranges)
    
    print("\n" + "="*60)
    print("💡 USE THESE INSIGHTS TO CALIBRATE SEIZURE DETECTION THRESHOLDS")
    print("="*60)


if __name__ == "__main__":
    main()
