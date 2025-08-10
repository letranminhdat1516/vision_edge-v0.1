#!/usr/bin/env python3
"""
Functional Fall Detection using MoveNet/PoseNet (from fall-detection repo)
Realtime detection from RTSP camera IMOU
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Model paths (can be changed if needed)
MODEL_PATH = str(Path(__file__).parent.parent.parent.parent / "repos_clone" / "fall-detection" / "ai_models" / "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
POSE_LABELS_PATH = str(Path(__file__).parent.parent.parent.parent / "repos_clone" / "fall-detection" / "ai_models" / "pose_labels.txt")

# Load MoveNet model (TFLite)
def load_movenet_model(model_path: str) -> Any:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Debug model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"DEBUG: Model input shape: {input_details[0]['shape']}")
    print(f"DEBUG: Model output shape: {output_details[0]['shape']}")
    
    return interpreter

# Preprocess frame for MoveNet
def preprocess_frame(frame: np.ndarray, input_size: Tuple[int, int]=(256,256)) -> np.ndarray:
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Run inference MoveNet
def run_movenet_inference(interpreter: Any, frame: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debug input frame (disabled for cleaner output)
    # print(f"DEBUG: Input frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    
    # Debug output keypoints (disabled for cleaner output)
    # print(f"DEBUG: Output keypoints shape: {keypoints.shape}")
    # if keypoints.size > 0:
    #     print(f"DEBUG: Output keypoints confidence range: {np.min(keypoints):.3f} to {np.max(keypoints):.3f}")
    
    return keypoints

# Simple fall detection logic (functional)
def detect_fall_from_keypoints(keypoints: np.ndarray, threshold: float=0.7) -> Dict[str, Any]:
    """
    Detect fall based on keypoints analysis
    Logic: Check if person's torso is more horizontal than vertical
    """
    try:
        # keypoints shape: (1, 1, 17, 3) for MoveNet
        kp = keypoints[0,0]
        
        # Get key points for fall detection
        nose_y = kp[0][1]        # Head (nose)
        left_hip_y = kp[11][1]   # Left hip
        right_hip_y = kp[12][1]  # Right hip
        left_knee_y = kp[13][1]  # Left knee
        right_knee_y = kp[14][1] # Right knee
        
        # Calculate average positions
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2
        
        # Check confidence of key points
        nose_conf = kp[0][2]
        hip_conf = (kp[11][2] + kp[12][2]) / 2
        knee_conf = (kp[13][2] + kp[14][2]) / 2
        
        # Need minimum confidence to make detection
        if nose_conf < 0.3 or hip_conf < 0.3 or knee_conf < 0.3:
            return {
                'fall_detected': False,
                'confidence': 0.0,
                'reason': 'Low keypoint confidence'
            }
        
        # Fall detection logic: 
        # Normal standing: nose_y < hip_y < knee_y
        # Fall: nose_y and hip_y are close to knee_y level
        head_to_hip_distance = abs(nose_y - avg_hip_y)
        hip_to_knee_distance = abs(avg_hip_y - avg_knee_y)
        
        # If head is very close to hip level (horizontal body), likely a fall
        fall_ratio = head_to_hip_distance / max(hip_to_knee_distance, 0.1)  # Avoid division by zero
        
        # Fall detected if ratio is very small (head close to hip level)
        # Use more strict detection to avoid false positives
        fall_detected = fall_ratio < (1.0 - threshold) and head_to_hip_distance < 0.2
        
        confidence = (1.0 - fall_ratio) if fall_detected else 0.0
        
        # Debug info (uncomment để debug)
        # print(f"Fall ratio: {fall_ratio:.3f}, Threshold: {threshold}, Head-Hip dist: {head_to_hip_distance:.3f}")
        
        return {
            'fall_detected': fall_detected,
            'confidence': min(max(confidence, 0.0), 1.0),
            'fall_ratio': fall_ratio,
            'keypoints': kp
        }
        
    except Exception as e:
        return {
            'fall_detected': False,
            'confidence': 0.0,
            'error': str(e)
        }

# Main functional pipeline for RTSP
def run_fall_detection_rtsp(rtsp_url: str, model_path: str=MODEL_PATH, threshold: float=0.7):
    interpreter = load_movenet_model(model_path)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from RTSP stream.")
            break
        input_frame = preprocess_frame(frame)
        keypoints = run_movenet_inference(interpreter, input_frame)
        result = detect_fall_from_keypoints(keypoints, threshold)
        # Visualization
        if result['fall_detected']:
            cv2.putText(frame, "FALL DETECTED!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage (can be called from main pipeline)

if __name__ == "__main__":
    from healthcare_monitor_functional.core.config import get_camera_config
    config = get_camera_config()
    print(f"[INFO] Using RTSP URL: {config['url']}")
    run_fall_detection_rtsp(config['url'])
