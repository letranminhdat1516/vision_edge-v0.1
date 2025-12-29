# YOLOv8-Pose Keypoint Extraction

- **Module**: [src/seizure_detection/yolov8_pose_estimator.py](src/seizure_detection/yolov8_pose_estimator.py) → class `YOLOv8PoseEstimator`
- **Mục đích**: Trích xuất 17 keypoints (khớp cơ thể) theo chuẩn COCO từ frame video, phục vụ phân tích tư thế và phát hiện bất thường.

---

## 1. Tham số khởi tạo (Constructor Parameters)

| Parameter    | Type | Default | Mô tả                                                                              |
| ------------ | ---- | ------- | ---------------------------------------------------------------------------------- |
| `model_size` | str  | `'n'`   | Kích thước model: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge) |

### Model Size Trade-offs

| Size         | Speed          | Accuracy   | Memory | Use Case                   |
| ------------ | -------------- | ---------- | ------ | -------------------------- |
| `n` (nano)   | ⚡⚡⚡ Fastest | ⭐⭐       | ~6MB   | Edge devices, Raspberry Pi |
| `s` (small)  | ⚡⚡ Fast      | ⭐⭐⭐     | ~14MB  | Balanced performance       |
| `m` (medium) | ⚡ Medium      | ⭐⭐⭐⭐   | ~50MB  | Higher accuracy needed     |
| `l` (large)  | 🐢 Slow        | ⭐⭐⭐⭐⭐ | ~90MB  | Maximum accuracy           |
| `x` (xlarge) | 🐢🐢 Slowest   | ⭐⭐⭐⭐⭐ | ~130MB | Research/Offline           |

---

## 2. COCO Pose Keypoints (17 điểm chuẩn)

```
                    0: nose
                   /    \
            1: left_eye  2: right_eye
                 |            |
            3: left_ear  4: right_ear

           5: left_shoulder ─── 6: right_shoulder
                 |                    |
           7: left_elbow        8: right_elbow
                 |                    |
           9: left_wrist       10: right_wrist

          11: left_hip ──────── 12: right_hip
                 |                    |
          13: left_knee        14: right_knee
                 |                    |
          15: left_ankle       16: right_ankle
```

### Keypoint Index Reference

| Index | Name             | Vị trí           | Essential\* |
| ----- | ---------------- | ---------------- | ----------- |
| 0     | `nose`           | Mũi              | ❌          |
| 1     | `left_eye`       | Mắt trái         | ❌          |
| 2     | `right_eye`      | Mắt phải         | ❌          |
| 3     | `left_ear`       | Tai trái         | ❌          |
| 4     | `right_ear`      | Tai phải         | ❌          |
| 5     | `left_shoulder`  | Vai trái         | ✅          |
| 6     | `right_shoulder` | Vai phải         | ✅          |
| 7     | `left_elbow`     | Khuỷu tay trái   | ❌          |
| 8     | `right_elbow`    | Khuỷu tay phải   | ❌          |
| 9     | `left_wrist`     | Cổ tay trái      | ❌          |
| 10    | `right_wrist`    | Cổ tay phải      | ❌          |
| 11    | `left_hip`       | Hông trái        | ✅          |
| 12    | `right_hip`      | Hông phải        | ✅          |
| 13    | `left_knee`      | Đầu gối trái     | ❌          |
| 14    | `right_knee`     | Đầu gối phải     | ❌          |
| 15    | `left_ankle`     | Mắt cá chân trái | ❌          |
| 16    | `right_ankle`    | Mắt cá chân phải | ❌          |

\*Essential = Bắt buộc phải nhìn thấy ít nhất 2/4 điểm này để keypoints hợp lệ

---

## 3. Skeleton Connections (Kết nối xương)

```python
skeleton_connections = [
    # Head (Đầu)
    (0, 1), (0, 2),     # nose → eyes
    (1, 3), (2, 4),     # eyes → ears

    # Arms (Tay)
    (5, 6),             # shoulders (vai)
    (5, 7), (7, 9),     # left arm (tay trái)
    (6, 8), (8, 10),    # right arm (tay phải)

    # Torso (Thân)
    (5, 11), (6, 12),   # shoulders → hips
    (11, 12),           # hips (hông)

    # Legs (Chân)
    (11, 13), (13, 15), # left leg (chân trái)
    (12, 14), (14, 16), # right leg (chân phải)
]
```

---

## 4. Algorithm Chi Tiết

### Bước 1: Load Model

```python
model_name = f'yolov8{model_size}-pose.pt'  # e.g., 'yolov8n-pose.pt'
self.model = YOLO(model_name)
```

### Bước 2: Inference

```python
results = self.model(frame, verbose=False)
```

**Output structure:**

- `results[0].boxes` - Bounding boxes của tất cả người phát hiện được
- `results[0].keypoints` - Keypoints tương ứng với mỗi person

### Bước 3: Chọn Best Person (Highest Confidence)

```python
for i in range(len(boxes)):
    box = boxes[i]
    conf_value = box.conf.item()  # Confidence của bounding box

    if conf_value > best_confidence:
        best_confidence = conf_value
        best_person_idx = i
```

**Logic**: Trong một frame có nhiều người, chọn người có `box.conf` cao nhất để phân tích.

### Bước 4: Threshold Check

```python
if best_confidence < confidence_threshold:  # default 0.5
    return None  # Reject - không đủ tin cậy
```

### Bước 5: Extract Keypoints

```python
keypoints_tensor = results[0].keypoints.data
keypoints_data = keypoints_tensor[best_person_idx]  # Shape: (17, 3)

# Convert to numpy
keypoints = keypoints_data.cpu().numpy()
```

**Output shape**: `(17, 3)` với mỗi hàng là `[x, y, confidence]`

- `x`: Tọa độ X (pixels)
- `y`: Tọa độ Y (pixels)
- `confidence`: Độ tin cậy của keypoint đó (0.0 - 1.0)

### Bước 6: Validate Keypoints

```python
def _validate_keypoints(keypoints, min_visible_points=5):
    # Check shape
    if keypoints.shape != (17, 3):
        return False

    # Count visible keypoints (confidence > 0.3)
    visible_points = np.sum(keypoints[:, 2] > 0.3)

    # Check essential points (shoulders + hips)
    essential_points = [5, 6, 11, 12]
    essential_visible = np.sum(keypoints[essential_points, 2] > 0.3)

    return visible_points >= 5 and essential_visible >= 2
```

**Validation Rules:**

1. Phải có đúng 17 keypoints
2. Ít nhất 5 keypoints có `confidence > 0.3`
3. Ít nhất 2/4 essential points (vai + hông) nhìn thấy được

---

## 5. Output Format

### Single Person (`extract_keypoints`)

```python
def extract_keypoints(frame, confidence_threshold=0.5) -> Optional[np.ndarray]:
    # Returns: np.ndarray (17, 3) or None
```

### All Persons (`extract_all_keypoints`)

```python
def extract_all_keypoints(frame, confidence_threshold=0.5) -> List[Dict]:
    # Returns: [
    #     {
    #         'keypoints': np.ndarray (17, 3),
    #         'bbox': [x1, y1, x2, y2],
    #         'confidence': 0.85
    #     },
    #     ...
    # ]
```

---

## 6. Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: BGR Frame (H, W, 3)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: YOLOv8-Pose Inference                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │ results = model(frame, verbose=False)       │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Find Best Person                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │ Loop through all detected persons           │                │
│  │ Select person with highest box.conf         │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Confidence Check                                        │
│  ┌──────────────────────────────────┐                           │
│  │ best_confidence >= 0.5 ?         │──No──▶ Return None        │
│  └──────────────────────────────────┘                           │
│              │ Yes                                               │
└──────────────┼──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Extract & Convert Keypoints                             │
│  ┌─────────────────────────────────────────────┐                │
│  │ keypoints = tensor[best_idx].cpu().numpy()  │                │
│  │ Shape: (17, 3) = [x, y, conf] × 17          │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Validate Keypoints                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ visible_points >= 5 ?                                     │   │
│  │ essential_visible (shoulders + hips) >= 2 ?              │   │
│  └──────────────────────────────────────────────────────────┘   │
│              │ Yes                    │ No                       │
│              ▼                        ▼                          │
│     Return keypoints (17,3)     Return None                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Performance Tracking

```python
def get_performance_stats() -> Dict:
    return {
        'total_detections': 1000,        # Tổng số lần gọi
        'successful_detections': 850,    # Số lần trích xuất thành công
        'success_rate': 85.0,            # Tỷ lệ thành công (%)
        'avg_inference_time_ms': 15.2,   # Thời gian inference trung bình
        'model_size': 'n'                # Model đang dùng
    }
```

---

## 8. Visualization

```python
def visualize_pose(frame, keypoints, show_confidence=True) -> np.ndarray:
    # Vẽ keypoints (circles)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            radius = 6 if conf > 0.8 else 5 if conf > 0.5 else 4
            cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), -1)

    # Vẽ skeleton (lines)
    for pt1_idx, pt2_idx in skeleton_connections:
        if keypoints[pt1_idx, 2] > 0.3 and keypoints[pt2_idx, 2] > 0.3:
            pt1 = (int(keypoints[pt1_idx, 0]), int(keypoints[pt1_idx, 1]))
            pt2 = (int(keypoints[pt2_idx, 0]), int(keypoints[pt2_idx, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    return frame
```

---

## 9. Use Cases trong Healthcare Monitoring

| Scenario              | Keypoints Used                  | Analysis                                  |
| --------------------- | ------------------------------- | ----------------------------------------- |
| **Fall Detection**    | Hips (11, 12), Shoulders (5, 6) | Theo dõi vị trí center body qua thời gian |
| **Seizure Detection** | All 17 points                   | Phân tích amplitude và frequency dao động |
| **Posture Analysis**  | Shoulders, Hips, Knees          | Góc giữa các khớp để xác định tư thế      |
| **Lying Detection**   | Hips, Shoulders                 | Aspect ratio của bounding box             |

---

## 10. Complexity Analysis

| Metric             | YOLOv8n   | YOLOv8s    | YOLOv8m    |
| ------------------ | --------- | ---------- | ---------- |
| **Inference Time** | ~10-15ms  | ~20-30ms   | ~40-60ms   |
| **GPU Memory**     | ~200MB    | ~400MB     | ~800MB     |
| **CPU Only**       | ~50-100ms | ~100-200ms | ~200-400ms |

---

## Notes (VI)

Trích xuất 17 khớp cơ thể (COCO format) từ người có độ tin cậy bbox cao nhất (≥0.5). Validation yêu cầu ít nhất 5 keypoints nhìn thấy và 2/4 essential points (vai + hông). Model nano phù hợp cho edge devices, model lớn hơn cho accuracy cao hơn.
