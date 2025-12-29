# Keyframe Detection (Frame Difference + History)

- Module: [src/video_processing/simple_processing.py](src/video_processing/simple_processing.py) → class `SimpleKeyframeDetector`

## Algorithm
- Grayscale + Gaussian blur (9×9); compute abs diff with last frame.
- Normalize `normalized_diff = nonzero(diff) / (H×W)`.
- Real-time threshold: `is_keyframe ← normalized_diff > min_diff_threshold` (default 0.01).
- History refinement: require `normalized_diff > 1.5 × mean(last 5 diffs)`.

## Output
- `(is_keyframe, normalized_diff)`.

Notes (VI): Chọn keyframe khi độ khác biệt ảnh vượt ngưỡng và lớn hơn trung bình gần đây.
