# Reliability Score Calculation (Tính Độ Nguy Hiểm)

## Tổng Quan

`reliability_score` là điểm số đánh giá **độ nguy hiểm** của một sự kiện phát hiện, khác với `confidence_score` (độ tin cậy của AI).

- **confidence_score**: AI tin tưởng bao nhiêu % vào detection (0-100%)
- **reliability_score**: Sự kiện này nguy hiểm đến mức nào (0-100%)

## Công Thức Tính

### Tổng Điểm = Base Score + Severity Score + Quality Score + Context Score

```
Reliability Score = min(max(
    (confidence × 0.4) +          // 40% từ confidence
    (event_severity) +            // 30% từ loại sự kiện
    (detection_quality) +         // 15% từ chất lượng detection
    (context_factors)             // 15% từ bối cảnh
, 0.0), 1.0)
```

---

## 1. Base Score (40% trọng số)

**Nguồn**: Độ tin cậy của AI model

```python
base_score = confidence * 0.4
```

**Ví dụ**:

- Confidence 85% → Base score = 0.85 × 0.4 = **0.34**
- Confidence 50% → Base score = 0.50 × 0.4 = **0.20**

---

## 2. Event Severity Score (30% trọng số)

**Nguồn**: Loại sự kiện được phát hiện

| Event Type          | Severity Score | Lý Do                                      |
| ------------------- | -------------- | ------------------------------------------ |
| `fall`              | 0.30           | Té ngã - rất nguy hiểm, cần cấp cứu ngay   |
| `manual_emergency`  | 0.30           | Khẩn cấp thủ công - người dùng báo động    |
| `abnormal_behavior` | 0.28           | Co giật/hành vi bất thường - nguy hiểm cao |
| `seizure`           | 0.28           | Co giật - nguy hiểm cao                    |
| `sleep`             | 0.05           | Ngủ - ít nguy hiểm                         |
| `normal_activity`   | 0.02           | Hoạt động bình thường - không nguy hiểm    |
| Unknown             | 0.15           | Mức trung bình                             |

**Ví dụ**:

- Event: `fall` → Severity = **0.30**
- Event: `sleep` → Severity = **0.05**

---

## 3. Detection Quality Score (15% trọng số)

**Nguồn**: Chất lượng của detection data

### 3.1 Base Quality (0.10)

- Có bounding boxes → +0.10

### 3.2 Multiple Detections Bonus (0.03)

- Phát hiện ≥ 2 objects → +0.03
- Lý do: Người té có thể xuất hiện nhiều pose khác nhau

### 3.3 Keypoints Bonus (0.02)

- Có pose keypoints data → +0.02
- Lý do: Dữ liệu pose chi tiết hơn = đánh giá chính xác hơn

**Ví dụ**:

```python
# Case 1: 1 bounding box, không có keypoints
quality_score = 0.10

# Case 2: 2 bounding boxes, có keypoints
quality_score = 0.10 + 0.03 + 0.02 = 0.15
```

---

## 4. Context Factors Score (15% trọng số)

**Nguồn**: Thông tin bối cảnh

### 4.1 Alert Level

| Alert Level | Context Score |
| ----------- | ------------- |
| `critical`  | 0.15          |
| `high`      | 0.12          |
| `warning`   | 0.08          |
| Other       | 0.00          |

### 4.2 Consecutive Detections Bonus (0.03)

- Phát hiện liên tục ≥ 3 lần → +0.03
- Lý do: Phát hiện liên tục = sự kiện đang diễn ra thật

**Ví dụ**:

```python
# Case 1: Alert level = critical, 1 detection
context_score = 0.15

# Case 2: Alert level = critical, 5 consecutive detections
context_score = 0.15 + 0.03 = 0.18
```

---

## Ví Dụ Tính Toán Cụ Thể

### Ví dụ 1: Fall Detection - Nguy Hiểm Cao

```python
Event Data:
- confidence: 0.85
- event_type: 'fall'
- bounding_boxes: [{'bbox': [100, 200, 300, 400], 'keypoints': [...]}]
- context: {'alert_level': 'critical', 'consecutive_detections': 5}

Calculation:
1. Base Score:     0.85 × 0.4 = 0.34
2. Severity:       0.30 (fall)
3. Quality:        0.10 (has bbox) + 0.02 (has keypoints) = 0.12
4. Context:        0.15 (critical) + 0.03 (consecutive) = 0.18

Total: 0.34 + 0.30 + 0.12 + 0.18 = 0.94

Reliability Score: 94% (CỰC KỲ NGUY HIỂM)
```

### Ví dụ 2: Sleep Detection - Ít Nguy Hiểm

```python
Event Data:
- confidence: 0.67
- event_type: 'sleep'
- bounding_boxes: [{'bbox': [50, 100, 200, 300]}]
- context: {'alert_level': 'warning'}

Calculation:
1. Base Score:     0.67 × 0.4 = 0.268
2. Severity:       0.05 (sleep)
3. Quality:        0.10 (has bbox)
4. Context:        0.08 (warning)

Total: 0.268 + 0.05 + 0.10 + 0.08 = 0.498

Reliability Score: 50% (MỨC TRUNG BÌNH)
```

### Ví dụ 3: Abnormal Behavior - Nguy Hiểm Cao

```python
Event Data:
- confidence: 0.71
- event_type: 'abnormal_behavior'
- bounding_boxes: [{'bbox': [93, 272, 225, 694], 'confidence': 1.0}]
- context: {'alert_level': 'critical'}

Calculation:
1. Base Score:     0.71 × 0.4 = 0.284
2. Severity:       0.28 (abnormal_behavior)
3. Quality:        0.10 (has bbox)
4. Context:        0.15 (critical)

Total: 0.284 + 0.28 + 0.10 + 0.15 = 0.814

Reliability Score: 81.4% (NGUY HIỂM CAO)
```

---

## Phân Loại Mức Độ Nguy Hiểm

| Reliability Score | Mức Độ                  | Hành Động                           |
| ----------------- | ----------------------- | ----------------------------------- |
| 0.80 - 1.00       | 🔴 **CỰC KỲ NGUY HIỂM** | Gửi cảnh báo ngay lập tức, báo động |
| 0.60 - 0.79       | 🟠 **NGUY HIỂM CAO**    | Cảnh báo ưu tiên, cần xử lý nhanh   |
| 0.40 - 0.59       | 🟡 **CẢNH BÁO**         | Theo dõi chặt chẽ                   |
| 0.20 - 0.39       | 🟢 **THẤP**             | Ghi nhận, theo dõi thường           |
| 0.00 - 0.19       | ⚪ **RẤT THẤP**         | Chỉ ghi log                         |

---

## Sử Dụng Trong Code

```python
# In postgresql_healthcare_service.py

reliability_score = self._calculate_reliability_score(
    confidence=0.85,
    event_type='fall',
    bounding_boxes=[{'bbox': [100, 200, 300, 400], 'keypoints': [...]}],
    context={'alert_level': 'critical', 'consecutive_detections': 5}
)

# Kết quả: reliability_score = 0.94 (94%)
```

---

## Lưu Ý Quan Trọng

1. **Độc lập với confidence**: Một event có confidence thấp nhưng vẫn có thể rất nguy hiểm (ví dụ: fall với confidence 60% vẫn là nguy hiểm cao)

2. **Context rất quan trọng**: Phát hiện liên tục nhiều lần tăng độ tin cậy của reliability score

3. **Cập nhật trọng số**: Có thể điều chỉnh các trọng số (40%, 30%, 15%, 15%) dựa trên feedback thực tế

4. **Lưu vào database**: Field `reliability_score` trong bảng `event_detections` để phân tích sau này

---

## Changelog

- **2025-11-06**: Tạo công thức tính reliability score với 4 yếu tố chính
