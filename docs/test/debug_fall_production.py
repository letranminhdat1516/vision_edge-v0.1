"""Debug fall detection in production context"""
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '.')

from fall_detection.simple_fall_detector import SimpleFallDetector
from ultralytics import YOLO
import cv2
import numpy as np

video_path = 'resource/chung.mp4'
cap = cv2.VideoCapture(video_path)

# Jump to frame 2760 - need 9 frames before 2769 to warm up buffer
cap.set(cv2.CAP_PROP_POS_FRAMES, 2760)

detector = SimpleFallDetector()
# Use yolov8s.pt like test_fall_only.py actually uses
yolo = YOLO('yolov8s.pt')
print(f"Using YOLO model: yolov8s.pt")

prev_frame = None

print("Testing fall detection at frames 2765-2775...")
print("Using YOLO to detect person (like production)")  
print("Using time.time() for timestamp (like test_fall_only)")
print("="*60)

# Enable detailed logging for fall detector
import logging
logging.getLogger('fall_detection.simple_fall_detector').setLevel(logging.INFO)

# Check if video opened
if not cap.isOpened():
    print("ERROR: Could not open video!")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video has {total_frames} frames")
print(f"Current position after seek: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")

import time as time_module

for i in range(2760, 2780):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: Failed to read!")
        break
    
    # Calculate motion (same as production)
    motion = 0.0
    if prev_frame is not None:
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        motion = np.mean(diff) / 255.0
    prev_frame = frame.copy()
    
    # Detect person with YOLO (like production)
    results = yolo(frame, conf=0.15, classes=[0], verbose=False)
    person_bbox = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            person_bbox = [int(x1), int(y1), int(x2), int(y2)]
            break
        if person_bbox:
            break
    
    # Detect fall - USE time.time() like test_fall_only does!
    current_time = time_module.time()
    fall_result = detector.detect_fall(
        frame, 
        timestamp=current_time,  # Use real timestamp
        person_bbox=person_bbox, 
        motion_level=motion
    )
    
    conf = fall_result.get('confidence', 0)
    method = fall_result.get('method', 'unknown')
    fall_detected = fall_result.get('fall_detected', False)
    
    # Print frames with fall or high confidence - print more debug info
    print(f"Frame {i}: conf={conf:.3f}, method={method}, fall={fall_detected}, bbox={person_bbox}, motion={motion:.4f}")
    print(f"  -> buffer_len={len(detector.frame_buffer)}")
    # Check frame buffer content and calculate movements
    if len(detector.frame_buffer) >= 2:
        first_bbox = detector.frame_buffer[0].get('bbox')
        last_bbox = detector.frame_buffer[-1].get('bbox')
        print(f"  -> buffer[0].bbox={first_bbox}")
        print(f"  -> buffer[-1].bbox={last_bbox}")
        
        # Calculate movements like fall detector does
        if first_bbox and last_bbox:
            x1_1, y1_1, x2_1, y2_1 = first_bbox
            x1_2, y1_2, x2_2, y2_2 = last_bbox
            
            # Calculate centers
            center1_x = (x1_1 + x2_1) / 2
            center1_y = (y1_1 + y2_1) / 2
            center2_x = (x1_2 + x2_2) / 2
            center2_y = (y1_2 + y2_2) / 2
            
            # Calculate movement
            horizontal = abs(center2_x - center1_x)
            vertical = abs(center2_y - center1_y)
            
            # Calculate aspect ratios
            w1, h1 = (x2_1 - x1_1), (y2_1 - y1_1)
            w2, h2 = (x2_2 - x1_2), (y2_2 - y1_2)
            aspect1 = w1 / h1 if h1 > 0 else 0
            aspect2 = w2 / h2 if h2 > 0 else 0
            aspect_change = aspect2 / aspect1 if aspect1 > 0 else 0
            
            print(f"  -> horiz={horizontal:.1f}px, vert={vertical:.1f}px")
            print(f"  -> aspect: {aspect1:.2f} -> {aspect2:.2f} (change={aspect_change:.2f}x)")
            
            # Check sideways fall conditions
            is_sideways = (horizontal > 40 and aspect_change > 1.2 and aspect2 > 1.4)
            print(f"  -> Sideways fall pattern: {is_sideways} (need: horiz>40, aspect_change>1.2, aspect2>1.4)")

cap.release()
print("="*60)
print("Done")
print("="*60)
print("Done")
