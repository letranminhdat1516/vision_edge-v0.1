"""Debug: Compare fall detection between test_fall_only and test_production"""
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '.')

from fall_detection.simple_fall_detector import SimpleFallDetector
from ultralytics import YOLO
import cv2
import numpy as np
import time

video_path = 'resource/chung.mp4'

# Create TWO separate detectors to simulate both tests
detector1 = SimpleFallDetector()  # For simulating test_fall_only
detector2 = SimpleFallDetector()  # For simulating test_production

yolo = YOLO('yolov8s.pt')

print("="*70)
print("COMPARING: test_fall_only vs test_production fall detection")
print("="*70)

# Process frames from start to 2780 (like test_fall_only does)
cap = cv2.VideoCapture(video_path)

frame_buffer_prod = []  # Production-style motion calculation
buffer_size = 5

frame_count = 0
start_time = time.time()

print("\nProcessing frames 0 -> 2780 to warm up buffer like test_fall_only...")

while frame_count <= 2780:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO detect person
    results = yolo(frame, conf=0.15, classes=[0], verbose=False)
    person_bbox = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            person_bbox = [int(x1), int(y1), int(x2), int(y2)]
            break
    
    # Method 1: test_fall_only style motion (simple diff)
    # Actually test_fall_only also uses similar motion calculation
    
    # Method 2: test_production style motion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_prod = 0.0
    if len(frame_buffer_prod) >= 2:
        prev_gray = frame_buffer_prod[-1]
        diff = cv2.absdiff(gray, prev_gray)
        motion_prod = np.sum(diff > 25) / diff.size
    frame_buffer_prod.append(gray)
    if len(frame_buffer_prod) > buffer_size:
        frame_buffer_prod.pop(0)
    
    current_time = time.time()
    
    # Fall detection with BOTH detectors
    result1 = detector1.detect_fall(frame, timestamp=current_time, person_bbox=person_bbox, motion_level=motion_prod)
    result2 = detector2.detect_fall(frame, timestamp=current_time, person_bbox=person_bbox, motion_level=motion_prod)
    
    # Print details for frames around 2769
    if frame_count >= 2765 and frame_count <= 2775:
        conf1 = result1.get('confidence', 0)
        method1 = result1.get('method', 'unknown')
        fall1 = result1.get('fall_detected', False)
        
        print(f"\nFrame {frame_count}:")
        print(f"  bbox={person_bbox}")
        print(f"  motion={motion_prod:.4f}")
        print(f"  Detector1: conf={conf1:.3f}, method={method1}, fall={fall1}")
        
        # Check production filter
        has_real_motion = motion_prod > 0.015
        is_rapid = method1 == 'rapid_downward'
        is_sideways = method1 == 'sideways_fall'
        would_pass = conf1 >= 0.28 and (has_real_motion or is_rapid or is_sideways)
        print(f"  Production filter: motion>0.015={has_real_motion}, rapid={is_rapid}, sideways={is_sideways}")
        print(f"  WOULD PASS PRODUCTION: {would_pass}")
        
        if fall1:
            print(f"  ✅ FALL DETECTED!")
    
    frame_count += 1
    
    # Progress update
    if frame_count % 500 == 0:
        print(f"  ... processed {frame_count} frames")

cap.release()
print("\n" + "="*70)
print("Done")
