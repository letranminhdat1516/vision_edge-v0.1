# Seizure Predictor – Temporal Analysis

- **Module**: [src/seizure_detection/seizure_predictor.py](src/seizure_detection/seizure_predictor.py) → class `SeizurePredictor`
- **Mục đích**: Phân tích temporal và dự đoán khả năng co giật dựa trên confidence scores từ model VSViG/pose analysis.

---

## 1. Tham số khởi tạo (Constructor Parameters)

| Parameter           | Type  | Default | Mô tả                            |
| ------------------- | ----- | ------- | -------------------------------- |
| `temporal_window`   | int   | 5       | Số frames trong cửa sổ phân tích |
| `smoothing_factor`  | float | 0.8     | Hệ số làm mượt mũ (α)            |
| `alert_threshold`   | float | 0.90    | Ngưỡng cảnh báo CRITICAL         |
| `warning_threshold` | float | 0.80    | Ngưỡng cảnh báo WARNING          |

### Internal Buffers

```python
self.confidence_history = deque(maxlen=temporal_window)  # Raw confidences
self.smoothed_confidence = 0.0                           # EMA smoothed value
self.prediction_history = deque(maxlen=100)              # Last 100 predictions

# Alert state
self.current_alert_level = 'normal'   # 'normal' | 'warning' | 'critical'
self.seizure_start_time = None        # Thời điểm bắt đầu seizure
self.seizure_duration = 0.0           # Duration (seconds)
```

---

## 2. Exponential Moving Average (EMA) Smoothing

### Công thức

$$S_t = \alpha \cdot x_t + (1 - \alpha) \cdot S_{t-1}$$

Với:

- $S_t$ = Smoothed confidence tại thời điểm t
- $x_t$ = Raw confidence mới
- $\alpha$ = Smoothing factor (0.8)
- $S_{t-1}$ = Smoothed confidence trước đó

### Implementation

```python
if len(confidence_history) == 1:
    smoothed_confidence = confidence  # First value
else:
    smoothed_confidence = (
        smoothing_factor * confidence +          # 80% weight cho giá trị mới
        (1 - smoothing_factor) * smoothed_confidence  # 20% weight cho history
    )
```

### EMA Behavior Visualization

```
α = 0.8 (High weight for new values → Quick response)

Raw:      0.3  0.5  0.7  0.9  0.85  0.88  0.92
Smoothed: 0.3  0.46 0.65 0.85 0.85  0.87  0.91
                    ↑
          Nhanh chóng theo kịp raw values
```

### Tại sao dùng α = 0.8?

| α Value | Behavior                | Use Case                        |
| ------- | ----------------------- | ------------------------------- |
| 0.2     | Rất mượt, phản ứng chậm | Loại bỏ noise mạnh              |
| 0.5     | Cân bằng                | General purpose                 |
| **0.8** | Phản ứng nhanh, ít lag  | **Real-time seizure detection** |
| 0.95    | Gần như raw values      | Almost no smoothing             |

---

## 3. Temporal Pattern Analysis

```python
def _analyze_temporal_pattern() -> Dict:
    if len(confidence_history) < 5:
        return {'trend': 'insufficient_data', ...}

    history = np.array(list(confidence_history))
```

### 3.1 Trend Detection (Linear Regression)

```python
x = np.arange(len(history))
trend_slope = np.polyfit(x, history, 1)[0]  # Slope of linear fit

if trend_slope > 0.01:
    trend = 'increasing'   # Confidence đang tăng
elif trend_slope < -0.01:
    trend = 'decreasing'   # Confidence đang giảm
else:
    trend = 'stable'       # Ổn định
```

**Visualization:**

```
Increasing (slope > 0.01):     Decreasing (slope < -0.01):     Stable:
    ●                               ●                           ● ● ●
  ●                                   ●                        ●     ●
●                                       ●
```

### 3.2 Volatility (Standard Deviation)

```python
volatility = float(np.std(history))
```

| Volatility  | Meaning                         |
| ----------- | ------------------------------- |
| < 0.05      | Rất ổn định                     |
| 0.05 - 0.15 | Bình thường                     |
| > 0.15      | Biến động mạnh (có thể co giật) |

### 3.3 Peak Confidence

```python
peak_confidence = float(np.max(history))
```

### 3.4 Sustained High Detection

```python
recent_window = history[-10:] if len(history) >= 10 else history
sustained_high = bool(np.mean(recent_window) > warning_threshold)  # > 0.80
```

**Logic**: Nếu mean của 10 frames gần nhất > 0.80 → sustained_high = True

---

## 4. Alert Level Determination

### Decision Logic

```python
def _determine_alert_level(raw_conf, smooth_conf, temporal):
    # CRITICAL: Seizure detected
    if smooth_conf >= 0.90 or raw_conf >= 1.00:  # 0.90 + 0.10
        alert_level = 'critical'
        seizure_detected = True
        # Start tracking duration

    # WARNING: High risk
    elif smooth_conf >= 0.80:
        alert_level = 'warning'
        seizure_detected = False
        # Reset duration if was critical

    # NORMAL: Safe
    else:
        alert_level = 'normal'
        seizure_detected = False
        # Reset all tracking

    # Enhanced: Sustained high + increasing trend → at least warning
    if temporal['sustained_high'] and temporal['trend'] == 'increasing':
        if alert_level == 'normal':
            alert_level = 'warning'
```

### Alert Thresholds Diagram

```
Confidence
    1.0 ─────────────────────────────────────────
        │                    ▲
    0.90 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ CRITICAL ─ ─ ─
        │                    │
    0.80 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ WARNING ─ ─ ─
        │                    │
    0.50 ─────────────────────────────────────────
        │                    │
    0.0  ─────────────────────────────────────────
                          NORMAL
```

---

## 5. Seizure Duration Tracking

```python
# When entering CRITICAL state
if alert_level == 'critical':
    if seizure_start_time is None:
        seizure_start_time = current_time  # Start tracking

    seizure_duration = current_time - seizure_start_time

# When leaving CRITICAL state
elif current_alert_level == 'critical' and alert_level != 'critical':
    seizure_start_time = None
    seizure_duration = 0.0
```

### Duration Significance

| Duration   | Severity               | Action               |
| ---------- | ---------------------- | -------------------- |
| < 30s      | Normal seizure         | Monitor              |
| 30s - 2min | Prolonged              | Prepare intervention |
| > 2min     | **Status Epilepticus** | **EMERGENCY**        |

---

## 6. Output Format

### `update_prediction()` Output

```python
{
    'raw_confidence': 0.85,           # Input confidence
    'smoothed_confidence': 0.82,      # EMA smoothed
    'alert_level': 'warning',         # 'normal' | 'warning' | 'critical'
    'seizure_detected': False,        # True only for critical
    'alert_message': '⚠️ Seizure Warning: 0.82',
    'temporal_analysis': {
        'trend': 'increasing',
        'volatility': 0.12,
        'peak_confidence': 0.88,
        'sustained_high': True,
        'trend_slope': 0.023
    },
    'timestamp': 1703836800.0,
    'seizure_duration': 0.0
}
```

### `get_current_status()` Output

```python
{
    'alert_level': 'warning',
    'smoothed_confidence': 0.82,
    'seizure_duration': 0.0,
    'buffer_filled': True,            # history >= temporal_window
    'last_alert_time': 1703836795.0,
    'recent_trend': 'increasing'
}
```

---

## 7. Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│           INPUT: raw_confidence (0.0 - 1.0), timestamp           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Add to History Buffer                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │ confidence_history.append(confidence)       │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Exponential Smoothing (EMA)                             │
│  ┌─────────────────────────────────────────────┐                │
│  │ S = 0.8 × raw + 0.2 × S_prev                │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Temporal Analysis                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │ - Trend (linear regression slope)           │                │
│  │ - Volatility (std deviation)                │                │
│  │ - Peak confidence (max)                     │                │
│  │ - Sustained high (mean > 0.80)              │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Determine Alert Level                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ smoothed >= 0.90 OR raw >= 1.00 ?  ────Yes───▶ CRITICAL  │   │
│  │        │ No                                               │   │
│  │        ▼                                                  │   │
│  │ smoothed >= 0.80 ?  ───────────────Yes───▶ WARNING        │   │
│  │        │ No                                               │   │
│  │        ▼                                                  │   │
│  │                                    NORMAL                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Enhanced Check                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ IF sustained_high AND trend == 'increasing':             │   │
│  │    Upgrade NORMAL → WARNING                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Update Statistics & Return Result                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Statistics Tracking

```python
stats = {
    'total_predictions': 1000,        # Tổng số predictions
    'seizures_detected': 5,           # Số lần CRITICAL
    'false_positives': 2,             # Manual tracking
    'average_confidence': 0.35,       # Running average
    'max_confidence': 0.95,           # Highest ever seen
    'alert_history': [                # Recent alerts
        {'timestamp': ..., 'level': 'critical', 'confidence': 0.92},
        ...
    ]
}
```

---

## 9. API Methods Summary

| Method                        | Input        | Output     | Description            |
| ----------------------------- | ------------ | ---------- | ---------------------- |
| `update_prediction(conf, ts)` | float, float | Dict       | Main prediction update |
| `get_current_status()`        | -            | Dict       | Current state summary  |
| `get_statistics()`            | -            | Dict       | Full statistics        |
| `export_history()`            | -            | List[Dict] | All predictions        |
| `reset()`                     | -            | -          | Clear all state        |

---

## 10. Integration với Pipeline

```python
# Trong AdvancedHealthcarePipeline:
seizure_result = seizure_predictor.update_prediction(
    confidence=raw_seizure_confidence,
    timestamp=frame_timestamp
)

if seizure_result['alert_level'] == 'critical':
    # Trigger alarm
    # Create event
    # Capture snapshots
elif seizure_result['alert_level'] == 'warning':
    # Log warning
    # Prepare for potential escalation
```

---

## Notes (VI)

Làm mượt confidence bằng EMA (α=0.8) để phản ứng nhanh với thay đổi. Phân tích temporal pattern (trend, volatility, sustained high) để nâng cao độ chính xác. Ngưỡng: CRITICAL ≥ 0.90, WARNING ≥ 0.80. Theo dõi seizure duration để phát hiện Status Epilepticus (>2 phút = emergency).
