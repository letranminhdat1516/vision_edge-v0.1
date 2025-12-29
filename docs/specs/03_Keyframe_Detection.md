# Keyframe Detection (Frame Difference + History)

- **Module**: [src/video_processing/simple_processing.py](src/video_processing/simple_processing.py) → class `SimpleKeyframeDetector`
- **Mục đích**: Phát hiện các frame "quan trọng" (keyframe) trong video stream để giảm tải xử lý, chỉ phân tích những frame có sự thay đổi đáng kể.

---

## 1. Tham số khởi tạo (Constructor Parameters)

| Parameter            | Type  | Default | Mô tả                                                       |
| -------------------- | ----- | ------- | ----------------------------------------------------------- |
| `threshold`          | float | 0.3     | Ngưỡng phát hiện peak (0.1-0.9), dùng cho phân tích offline |
| `max_keyframes`      | int   | 5       | Số lượng keyframe tối đa được theo dõi                      |
| `min_diff_threshold` | float | 0.01    | Ngưỡng tối thiểu để xem xét là keyframe (1% pixel thay đổi) |

### Các biến nội bộ (Internal State)

```python
self.last_frame = None       # Frame trước đó (grayscale + blur)
self.diff_history = []       # Lịch sử các giá trị diff (tối đa 50, giữ lại 30)
self.frame_count = 0         # Đếm số frame đã xử lý
```

---

## 2. Algorithm Chi Tiết

### Bước 1: Tiền xử lý ảnh (Preprocessing)

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Chuyển BGR → Grayscale
blur_gray = cv2.GaussianBlur(gray, (9, 9), 0.0)     # Làm mờ Gaussian 9×9
```

**Giải thích:**

- **Grayscale**: Giảm từ 3 channels (BGR) xuống 1 channel → giảm computation 3 lần
- **Gaussian Blur (9×9)**:
  - Loại bỏ noise nhỏ (nhiễu sensor, compression artifacts)
  - Sigma = 0.0 → OpenCV tự tính sigma dựa trên kernel size
  - Kernel 9×9 đủ lớn để smooth noise nhưng không mất chi tiết quan trọng

### Bước 2: Xử lý Frame Đầu Tiên

```python
if self.last_frame is None:
    self.last_frame = blur_gray
    self.diff_history.append(0)
    return True, 1.0  # Frame đầu luôn là keyframe
```

**Logic**: Frame đầu tiên không có gì để so sánh → mặc định là keyframe với confidence = 1.0

### Bước 3: Tính Frame Difference

```python
diff = cv2.subtract(blur_gray, self.last_frame)     # Hiệu tuyệt đối pixel-wise
diff_magnitude = cv2.countNonZero(diff)             # Đếm số pixel khác 0
```

**Chi tiết toán học:**

- `cv2.subtract(A, B)` = |A - B| (saturated arithmetic, không âm)
- `countNonZero()` = số pixel có giá trị > 0 (có thay đổi)

### Bước 4: Chuẩn hóa (Normalization)

```python
normalized_diff = diff_magnitude / (frame.shape[0] * frame.shape[1])
```

**Công thức:**
$$\text{normalized\_diff} = \frac{\text{countNonZero}(\text{diff})}{H \times W}$$

**Ý nghĩa:**

- Giá trị trong khoảng [0, 1]
- 0.0 = không có pixel nào thay đổi (frame giống hệt)
- 1.0 = tất cả pixel đều thay đổi (frame hoàn toàn khác)
- 0.01 (1%) = có 1% tổng số pixel thay đổi

**Ví dụ** (frame 640×480 = 307,200 pixels):

- `normalized_diff = 0.01` → 3,072 pixels thay đổi
- `normalized_diff = 0.05` → 15,360 pixels thay đổi

### Bước 5: Quản lý History

```python
self.diff_history.append(normalized_diff)
self.last_frame = blur_gray.copy()

# Giới hạn memory: giữ tối đa 50, cắt xuống còn 30
if len(self.diff_history) > 50:
    self.diff_history = self.diff_history[-30:]
```

**Mục đích**: Tránh memory leak khi chạy lâu dài (streaming 24/7)

### Bước 6: Quyết định Keyframe (Two-Stage Decision)

#### Stage 1: Threshold Check

```python
is_keyframe = normalized_diff > self.min_diff_threshold  # > 0.01 (1%)
```

#### Stage 2: History-Based Refinement

```python
if len(self.diff_history) >= 5:
    recent_avg = np.mean(self.diff_history[-5:])         # Trung bình 5 frame gần nhất
    is_keyframe = is_keyframe and (normalized_diff > recent_avg * 1.5)
```

**Điều kiện cuối cùng:**

$$
\text{is\_keyframe} = \begin{cases}
\text{True} & \text{if } d > 0.01 \text{ AND } d > 1.5 \times \bar{d}_{5} \\
\text{False} & \text{otherwise}
\end{cases}
$$

Trong đó:

- $d$ = `normalized_diff` của frame hiện tại
- $\bar{d}_{5}$ = trung bình `normalized_diff` của 5 frame gần nhất

---

## 3. Tại sao cần History Refinement?

### Vấn đề với Threshold Cố Định

| Tình huống                            | Threshold cố định 0.01                           | Với History                             |
| ------------------------------------- | ------------------------------------------------ | --------------------------------------- |
| Camera tĩnh, không có gì di chuyển    | Mọi frame đều < 0.01 → không keyframe            | ✅ Đúng                                 |
| Camera rung nhẹ (noise ổn định ~0.02) | Mọi frame đều > 0.01 → **tất cả là keyframe** ❌ | Chỉ keyframe khi > 0.02 × 1.5 = 0.03 ✅ |
| Có người đi ngang qua (diff = 0.08)   | Keyframe ✅                                      | Keyframe (0.08 > 0.03) ✅               |

### Adaptive Behavior

History refinement giúp thuật toán **tự thích nghi** với:

- Mức noise cơ bản của camera
- Điều kiện ánh sáng (flickering)
- Môi trường có chuyển động nền ổn định

---

## 4. Output

```python
def is_keyframe(self, frame) -> Tuple[bool, float]:
    return (is_keyframe, normalized_diff)
```

| Output            | Type  | Mô tả                                                     |
| ----------------- | ----- | --------------------------------------------------------- |
| `is_keyframe`     | bool  | True nếu frame này là keyframe                            |
| `normalized_diff` | float | Độ khác biệt chuẩn hóa [0, 1] - dùng làm confidence score |

---

## 5. Statistics API

```python
def get_stats(self) -> Dict[str, Any]:
    return {
        'frame_count': self.frame_count,           # Tổng số frame đã xử lý
        'diff_history_length': len(self.diff_history),
        'avg_diff': np.mean(self.diff_history),    # Diff trung bình (baseline)
        'threshold': self.threshold,
        'min_diff_threshold': self.min_diff_threshold
    }
```

---

## 6. Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: BGR Frame                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Preprocessing                                           │
│  ┌─────────────────┐    ┌─────────────────────┐                 │
│  │ BGR → Grayscale │ → │ Gaussian Blur (9×9) │                  │
│  └─────────────────┘    └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: First Frame Check                                       │
│  ┌──────────────────────────┐                                   │
│  │ last_frame is None?      │──Yes──▶ Return (True, 1.0)        │
│  └──────────────────────────┘                                   │
│              │ No                                                │
└──────────────┼──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Frame Difference                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │ diff = |current_blur - last_frame|          │                │
│  │ diff_magnitude = countNonZero(diff)         │                │
│  │ normalized_diff = diff_magnitude / (H × W)  │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Threshold Check (Stage 1)                               │
│  ┌─────────────────────────────────────┐                        │
│  │ normalized_diff > 0.01 ?            │──No──▶ Return (False)  │
│  └─────────────────────────────────────┘                        │
│              │ Yes                                               │
└──────────────┼──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: History Refinement (Stage 2)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ IF history >= 5 frames:                                  │    │
│  │   recent_avg = mean(last 5 diffs)                       │    │
│  │   is_keyframe = normalized_diff > recent_avg × 1.5      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: (is_keyframe: bool, normalized_diff: float)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Complexity Analysis

| Metric                 | Value    | Giải thích                 |
| ---------------------- | -------- | -------------------------- |
| **Time Complexity**    | O(H × W) | Duyệt qua tất cả pixels    |
| **Space Complexity**   | O(H × W) | Lưu `last_frame` grayscale |
| **Memory (640×480)**   | ~300 KB  | 640×480×1 byte + history   |
| **Memory (1920×1080)** | ~2 MB    | 1920×1080×1 byte + history |

---

## 8. Use Cases trong Healthcare Monitoring

| Scenario               | Expected Behavior                            |
| ---------------------- | -------------------------------------------- |
| **Bệnh nhân nằm yên**  | Hầu hết frame bị skip → tiết kiệm CPU        |
| **Bệnh nhân trở mình** | Keyframe được detect → trigger pose analysis |
| **Bệnh nhân ngã**      | Rapid keyframes → liên tục phân tích         |
| **Camera flickering**  | History filtering loại bỏ false positives    |

---

## 9. Tuning Guidelines

| Tình huống                          | Điều chỉnh                              |
| ----------------------------------- | --------------------------------------- |
| Quá nhiều keyframe (noise cao)      | Tăng `min_diff_threshold` lên 0.02-0.03 |
| Bỏ lỡ chuyển động nhỏ               | Giảm `min_diff_threshold` xuống 0.005   |
| Phản ứng chậm với thay đổi đột ngột | Giảm history window (sửa code: 5 → 3)   |
| Quá nhạy với lighting changes       | Tăng hệ số 1.5 lên 2.0                  |

---

## Notes (VI)

Chọn keyframe khi độ khác biệt ảnh vượt ngưỡng tối thiểu (1% pixels) VÀ lớn hơn 1.5 lần trung bình của 5 frame gần đây. Cơ chế adaptive này giúp loại bỏ noise và chỉ phát hiện thay đổi thực sự đáng chú ý.
