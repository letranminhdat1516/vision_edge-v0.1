# Fall Detection – Heuristics + Velocity Model

- Module: [src/fall_detection/simple_fall_detector.py](src/fall_detection/simple_fall_detector.py) → class `SimpleFallDetector`

## Parameters
- `confidence_threshold=0.40`, `min_time_interval=0.15s`, frame buffer `max_buffer_size=5`.
- Cooldowns: `danger_cooldown=15s`, `standing_up_cooldown=3s`.
- Repeated sitting window: `10s`, threshold `3` events.

## Preprocessing
- Safe bbox conversion (validates `[x1,y1,x2,y2]`, positive size, finite floats).
- Movement features from first vs last buffered frame: widths/heights, aspect ratios, centers.

## Core Features
- Aspect ratios: a1 = w1/h1, a2 = w2/h2, Δa = a2/a1.
- Center shifts: Δx = |c2x−c1x|, Δy = |c2y−c1y|.
- BBox size change (depth heuristic): r_size = |w2h2 − w1h1| / (w1h1 + 1).

## Priority/Filters
1. Standing up filter: upward movement Δy > 300px and c2y < c1y → reject as standing-up and start 3s cooldown.
2. Small posture adjustment: downward Δy < 60px → reject, unless sideways pattern (below).
3. Sideways fall pattern (bypass small-down filter): Δx > 40px AND Δa > 1.2 AND a2 > 1.4.
   - Sideways fall confidence: conf = 0.55 + min((Δa−1.2)*0.25,0.15) + min(Δx/150,0.15) + min(Δy/80,0.10) capped at 0.90. Accept if ≥ 0.50.
4. Lying down (controlled): final c2y/H > 0.90 AND a2 > 1.2 AND 0.85 ≤ Δa ≤ 1.15 AND Δy < 600px AND v_y < 1500 px/s → reject as lying-down.
5. Rapid downward movement (Strategy 0): Δy > 70px and downward.
   - Walking filter: if Δx > 0.8×Δy and not Δy > 150px, reject.
   - Depth movement filter: r_size > 1.50 and 150 < Δy < 400 → reject.
   - Already lying filter: if initial a1 > 1.5 and not Δy > 250px, reject.
   - Sitting filter (definitive fall requirement): require (c2y/H ≥ 0.90) AND (a2 ≥ 1.4); else treat as sitting/squatting. Also detect repeated-sitting pattern (≥3 in 10s).
   - Deep bending filter: if a2 < 0.6 → reject as bending.

## Velocity Model
- Track `fall_start_time` and `fall_start_position`. After ≥0.1s, compute `fall_duration = t − t0`, `fall_velocity = (c2y − c1y) / (t − t0)`.
- Controlled descent: if `fall_velocity < 150 px/s` and `fall_duration > 0.5s` → reject.
- Fall types:
  - fast_fall if v > 400 px/s
  - moderate_fall if v ≥ 150 px/s
  - slow_collapse if duration ≥ 1.5s (stroke-like)
- Confidence aggregation:
  - Base: conf0 = min(0.90, 0.50 + Δy/180) then conf = min(0.95, conf0 × severity_multiplier) where severity_multiplier ∈ {1.3 (slow_collapse), 1.1 (moderate), 1.0 (fast)}; accept if conf ≥ 0.50 AND either v ≥ 600 px/s OR (c2y/H ≥ 0.90 ∧ a2 ≥ 1.4).

## Outputs
- `fall_detected`, `confidence`, `fall_type`, `fall_duration`, `fall_velocity`, `category/method` markers.

Notes (VI): Kết hợp hình học bbox + vận tốc + nhiều bộ lọc (đứng dậy, cúi người, đi ngang, chiều sâu, ngồi nhanh). Dùng cooldown 15s để tránh spam.
