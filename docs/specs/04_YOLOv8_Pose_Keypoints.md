# YOLOv8-Pose Keypoint Extraction

- Module: [src/seizure_detection/yolov8_pose_estimator.py](src/seizure_detection/yolov8_pose_estimator.py) → class `YOLOv8PoseEstimator`

## Algorithm
- Load `yolov8{size}-pose.pt` (`n/s/m/l/x`).
- Inference → select best person by highest `box.conf`.
- Threshold: `best_confidence >= confidence_threshold` (default 0.5).
- Extract 17 COCO keypoints `(x,y,conf)` for best person.

## Data
- `keypoint_names` and `skeleton_connections` defined for rendering/consistency.

Notes (VI): Trích xuất khớp (17 điểm) người có độ tin cậy cao nhất, ngưỡng 0.5.
