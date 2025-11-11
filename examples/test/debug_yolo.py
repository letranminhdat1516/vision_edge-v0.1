#!/usr/bin/env python3
"""
Debug YOLO Detection - Kiểm tra tại sao YOLO không detect được person
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ultralytics import YOLO
import numpy as np

def test_yolo_on_video(video_path: str):
    """Test YOLO trực tiếp trên video"""
    
    print("="*100)
    print("🔍 YOLO DETECTION DEBUG")
    print("="*100)
    print(f"📹 Video: {video_path}")
    print("="*100 + "\n")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video loaded:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f}s\n")
    
    # Load YOLO models với confidence rất thấp
    print("🤖 Loading YOLO models...\n")
    
    models_to_test = [
        ('yolov8n', 0.1),   # Nano - fastest
        ('yolov8s', 0.1),   # Small
        ('yolov8m', 0.1),   # Medium
    ]
    
    results_summary = []
    
    for model_name, conf_threshold in models_to_test:
        print(f"\n{'='*100}")
        print(f"🧪 Testing: {model_name} (confidence >= {conf_threshold})")
        print(f"{'='*100}\n")
        
        try:
            # Load model
            model = YOLO(f'{model_name}.pt')
            print(f"✅ Model loaded: {model_name}")
            
            # Test trên nhiều frames
            test_frames = [30, 100, 200, 500, 1000, 1500, 2000]  # Test frames khác nhau
            detections_per_frame = []
            
            for frame_num in test_frames:
                if frame_num >= total_frames:
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Run YOLO với confidence thấp
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Count persons
                person_count = 0
                all_detections = []
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        all_detections.append({
                            'class': class_name,
                            'confidence': conf
                        })
                        
                        if class_name == 'person':
                            person_count += 1
                
                detections_per_frame.append({
                    'frame': frame_num,
                    'persons': person_count,
                    'all_objects': len(all_detections),
                    'detections': all_detections[:5]  # First 5 detections
                })
                
                print(f"Frame {frame_num:4d}: Persons={person_count}, Total objects={len(all_detections)}")
                if all_detections[:3]:
                    for det in all_detections[:3]:
                        print(f"            - {det['class']}: {det['confidence']:.3f}")
            
            # Summary
            total_persons = sum(d['persons'] for d in detections_per_frame)
            avg_persons = total_persons / len(detections_per_frame) if detections_per_frame else 0
            
            print(f"\n📊 Summary for {model_name}:")
            print(f"   Frames tested: {len(detections_per_frame)}")
            print(f"   Total person detections: {total_persons}")
            print(f"   Average persons per frame: {avg_persons:.2f}")
            
            results_summary.append({
                'model': model_name,
                'confidence': conf_threshold,
                'total_persons': total_persons,
                'avg_persons': avg_persons,
                'frames_tested': len(detections_per_frame)
            })
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # Final summary
    print(f"\n\n{'='*100}")
    print(f"🎯 FINAL SUMMARY")
    print(f"{'='*100}\n")
    
    for result in results_summary:
        print(f"{result['model']:12s} | Conf={result['confidence']} | "
              f"Total Persons={result['total_persons']:3d} | "
              f"Avg={result['avg_persons']:.2f} | "
              f"Frames={result['frames_tested']}")
    
    print(f"\n{'='*100}")
    
    # Recommend best model
    if results_summary:
        best = max(results_summary, key=lambda x: x['total_persons'])
        if best['total_persons'] > 0:
            print(f"✅ RECOMMENDED: Use {best['model']} (detected {best['total_persons']} persons)")
        else:
            print(f"❌ NO MODEL DETECTED PERSONS!")
            print(f"   Possible issues:")
            print(f"   - Video quality too low")
            print(f"   - No persons visible in tested frames")
            print(f"   - Video format not compatible")
            print(f"\n💡 Suggestions:")
            print(f"   1. Check video manually to see if persons are visible")
            print(f"   2. Try different frames (person may appear later)")
            print(f"   3. Test with higher quality video")
    
    cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_yolo.py <video_number>")
        print("Example: python debug_yolo.py 1")
        sys.exit(1)
    
    video_number = sys.argv[1]
    
    # Find video
    resource_folder = Path(__file__).parent / "resource"
    video_path_lower = resource_folder / f"{video_number}.mp4"
    video_path_upper = resource_folder / f"{video_number}.MP4"
    
    if video_path_lower.exists():
        video_path = str(video_path_lower)
    elif video_path_upper.exists():
        video_path = str(video_path_upper)
    else:
        print(f"❌ Video {video_number} not found in {resource_folder}")
        sys.exit(1)
    
    test_yolo_on_video(video_path)
