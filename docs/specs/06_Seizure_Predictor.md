# Seizure Predictor – Temporal Analysis

- Module: [src/seizure_detection/seizure_predictor.py](src/seizure_detection/seizure_predictor.py) → class `SeizurePredictor`

## Parameters
- `temporal_window=5`, `smoothing_factor=0.8`, `alert_threshold=0.90`, `warning_threshold=0.80`.

## Temporal Buffers
- Store last N confidences, maintain smoothed value and history.

## Equations
- Exponential smoothing: S_t = α x_t + (1−α) S_{t−1} with α=0.8.
- Trend via linear regression slope on window: `slope = polyfit(range(n), history, 1)[0]` with cutoffs >0.01 increasing, <−0.01 decreasing.
- Volatility: `std(history)`; Peak: `max(history)`; Sustained high: `mean(last 10) > warning_threshold`.

## Alert Logic
- `critical` if `smooth ≥ 0.90` OR `raw ≥ 1.00` (0.90+0.1); track `seizure_duration`.
- `warning` if `smooth ≥ 0.80`.
- `normal` otherwise; resets duration.
- If sustained_high AND trend increasing: at least `warning`.

Notes (VI): Làm mượt mũ suy, phát hiện xu hướng/độ biến động, đưa ra cảnh báo dựa trên ngưỡng.
