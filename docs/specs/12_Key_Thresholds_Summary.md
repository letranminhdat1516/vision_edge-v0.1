# Key Thresholds Summary (Quick Reference)

## Motion
- `motion_pixels > 150` (MOG2).

## Keyframe
- `normalized_diff > 0.01` and `> 1.5×avg(last 5)`.

## YOLOv8-Pose person selection
- `confidence ≥ 0.5`.

## Fall detection
- Rapid: `Δy > 70px` downward; accept if velocity `≥ 600 px/s` OR `(c2y/H ≥ 0.90 ∧ a2 ≥ 1.4)`.
- Sideways: `Δx > 40px ∧ Δa > 1.2 ∧ a2 > 1.4`; conf formula with caps.
- Lying down controlled: final near floor, aspect stable, velocity `< 1500 px/s`.
- Controlled descent reject: `v < 150 px/s` and `duration > 0.5s`.
- Cooldowns: `danger 15s`, `standing-up 3s`.

## Seizure predictor
- Smoothing `α=0.8`; `warning ≥ 0.80`, `critical ≥ 0.90`.

## Pipeline cooldowns
- Global `45s`, fall `10s`, seizure `30s`; normal-log throttle 10s; danger → prevent normal for `180s`.

Giải thích (VI): Bảng tóm tắt nhanh các ngưỡng/điều kiện quan trọng để vận hành và tinh chỉnh.
