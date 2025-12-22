#!/usr/bin/env python3
"""
Test Simple Seizure Detector
Logic đơn giản: Chỉ detect khi NẰM + VSViG high + liên tục
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seizure_detection.simple_seizure_detector import SimpleSeizureDetector
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='41', help='Video number or path')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=-1, help='End frame (-1 = all)')
    args = parser.parse_args()
    
    # Find video
    script_dir = Path(__file__).parent
    resource_folder = script_dir / "resource"
    
    if args.video.isdigit():
        video_path = resource_folder / f"{args.video}.mp4"
    else:
        video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Video: {video_path}")
    
    # Load models
    print("🔄 Loading models...")
    detector = SimpleSeizureDetector()
    detector.load_models()
    
    person_detector = YOLO('yolov8s.pt')
    pose_model = YOLO('yolov8n-pose.pt')
    
    print("✅ Models loaded")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    
    end_frame = args.end if args.end > 0 else total_frames
    
    print(f"📊 FPS: {fps}, Frames: {args.start} to {end_frame}")
    print("=" * 60)
    
    frame_count = args.start
    seizure_count = 0
    warning_count = 0
    
    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect person
        results = person_detector(frame, classes=[0], conf=0.3, verbose=False)
        
        if len(results[0].boxes) == 0:
            frame_count += 1
            continue
        
        # Get largest person bbox
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = np.argmax(areas)
        bbox = boxes[best_idx].astype(int)
        
        # Get pose keypoints
        pose_results = pose_model(frame, verbose=False)
        
        if len(pose_results[0].keypoints) == 0:
            frame_count += 1
            continue
        
        keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
        
        # Detect seizure
        result = detector.detect(frame, keypoints)
        
        # Draw on frame
        color = (0, 255, 0)  # Green = normal
        label = "NORMAL"
        
        if result['alert_level'] == 'warning':
            color = (0, 165, 255)  # Orange
            label = f"WARNING ({result['accumulation']}/90)"
            warning_count += 1
        elif result['alert_level'] == 'critical':
            color = (0, 0, 255)  # Red
            label = f"🚨 SEIZURE CRITICAL! ({result['accumulation']}/90)"
            seizure_count += 1
        
        # Draw bbox
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw info
        info_y = 30
        cv2.putText(frame, f"Frame: {frame_count}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, f"Posture: {result['posture']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, f"VSViG: {result['vsvig_conf']:.2f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, f"Accum: {result['accumulation']}/90", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, label, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
        
        # Show
        cv2.imshow('Simple Seizure Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Pause
        
        frame_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            print(f"⏳ Frame {frame_count}/{end_frame} | Seizures: {seizure_count} | Warnings: {warning_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("=" * 60)
    print(f"✅ Done! Processed {frame_count - args.start} frames")
    print(f"🚨 Seizures detected: {seizure_count}")
    print(f"⚠️ Warnings: {warning_count}")


if __name__ == '__main__':
    main()
