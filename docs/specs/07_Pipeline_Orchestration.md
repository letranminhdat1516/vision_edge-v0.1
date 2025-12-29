# Combined Video Pipeline – Orchestration & Cooldowns

- Module: [src/service/advanced_healthcare_pipeline.py](src/service/advanced_healthcare_pipeline.py) → class `AdvancedHealthcarePipeline`

## Buffers/Throttling
- Frame buffer: deque `maxlen=5` to capture pre-event frames.
- Normal log throttle: only log NORMAL every ~10s; block NORMAL for 180s after a DANGER.
- Global event cooldown: 45s between any two events.
- Fall-specific cooldown: 10s between fall detections.
- Seizure-specific cooldown: 30s between seizure detections.

## Keyframe Gating
- Only perform heavier AI steps on keyframes.

## Event Creation Strategy
1. Create event in DB first (get `event_id`).
2. Capture snapshots (5 images immediate using buffer + current frame).
3. Update event with `snapshot_id`.

## Status Mapping
- Five levels used in pipeline paths: danger, warning, suspect, normal (plus internal transitions).

Notes (VI): Pipeline điều phối cooldown toàn cục + riêng loại, tạo event trước rồi chụp ảnh để có `event_id` liên kết snapshot.
