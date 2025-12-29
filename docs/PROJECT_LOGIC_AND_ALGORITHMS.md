# Project Logic, Algorithms, and Formulas (Comprehensive Overview)

Generated from the current codebase (2025-12-29). Each item includes where it lives in the repo and the core logic/thresholds. English titles with Vietnamese notes for clarity.

---

## 0. Index (What’s Covered)
- Camera ingestion & reconnection
- Motion detection (background subtraction)
- Keyframe detection (frame-diff + history)
- YOLOv8-Pose keypoint extraction
- Fall detection (bbox dynamics + velocity/state filters)
- Seizure predictor (temporal smoothing + trend/volatility)
- Pipeline orchestration (cooldowns, snapshots, status levels)
- Event publishing (severity mapping, priority gating, captions)
- Snapshot storage (MinIO, metadata cleaning)
- Alarm runtime (PostgreSQL LISTEN/NOTIFY + audio)

Giải thích (VI): Danh sách toàn bộ thuật toán/chức năng tính toán chính và nơi triển khai trong code.

---

## 1) Camera Ingestion & Health
- Module: `src/camera/simple_camera.py`

### Component Purpose
- Provide a resilient RTSP camera reader for downstream video processing.
- Maintain stream health metrics (FPS, latency, drops) and auto-recovery.

Giải thích (VI): Thành phần đọc camera RTSP, theo dõi sức khỏe luồng và tự khôi phục khi rớt kết nối.

### Responsibilities
- Open and validate RTSP streams; handle credentials and URL formats.
- Read frames continuously; expose latest frame to the pipeline.
- Detect stalls/drops; run reconnection with backoff.
- Report health stats to the pipeline/logger.

### Inputs & Sources
- RTSP URL (e.g., `rtsp://user:pass@ip:port/…`).
- Optional camera configuration from `config/config.json` and validation in `src/main.py` (`load_camera_config()`, `validate_camera_credentials()`).

### Outputs & Events
- Frames (BGR `np.ndarray`).
- Health metrics: `is_open`, `fps`, `last_frame_ts`, `drop_count`, `error_count`.
- Status events (log): `CONNECTING`, `STREAMING`, `RECONNECTING`, `ERROR`.

### Public Interface (Runtime Contract)
- `open()` → attempts connection; sets state to `CONNECTING` → `STREAMING` on success.
- `read()` → returns `(ok: bool, frame: np.ndarray | None, ts: float)`; updates FPS/drops.
- `close()` → releases resources; state `DISCONNECTED`.
- `is_open()` → bool.

Giải thích (VI): Giao diện chạy gồm mở/đọc/đóng luồng và các chỉ số sức khỏe.

### Configuration Parameters
- `rtsp_url`: Camera endpoint with credentials.
- `connect_timeout`: e.g., 5–10s.
- `read_timeout`: e.g., 2–5s before stall considered.
- `retry_policy`: max retries (e.g., 5), `backoff_initial` (2s), `backoff_max` (30s).
- `target_fps`: expected stream FPS (used to flag degradation).
- `resolution_hint`: optional resize for downstream processing.

### State Machine
- States: `DISCONNECTED` → `CONNECTING` → `STREAMING` ↔ `DEGRADED` → `RECONNECTING` → `STREAMING` | `ERROR`.
- Triggers:
  - `CONNECTING→STREAMING`: `isOpened()` and first frame received.
  - `STREAMING→DEGRADED`: FPS < 50% target or read latency > `read_timeout`.
  - `DEGRADED→RECONNECTING`: consecutive stalls > N (e.g., 3).
  - `ANY→ERROR`: unrecoverable auth/URL error.

Giải thích (VI): Trạng thái luồng và điều kiện chuyển trạng thái khi rớt hoặc suy giảm.

### Health Metrics & Telemetry
- `fps = frames_read / elapsed_time` (sliding window 5–10s).
- `read_latency = now − last_frame_ts`.
- `stalls = count(read_latency > read_timeout)`.
- `drops = count(read() returns False)`.

### Failure Modes & Recovery
- Authentication/URL error: validate and mask passwords in logs; suggest alternative creds (see `validate_camera_credentials()` in `src/main.py`).
- Transport drop: exponential backoff reconnect.
- Long stall: force close/reopen capture.
- Codec/format mismatch: fallback to software decode or lower resolution if supported.

Giải thích (VI): Các lỗi thường gặp và cách tự khôi phục để duy trì luồng.

### Security & Privacy
- Never log plain passwords; mask in diagnostics.
- RTSP URLs stored securely; restrict file/system access.

### Monitoring & Alerts
- Warn if `fps < 0.5×target_fps` for > 10s.
- Alert if `stalls ≥ 3` within 30s or `drops` rising continuously.

### Testing Checklist
- Connect to valid RTSP; verify `STREAMING` within `connect_timeout`.
- Simulate disconnect; verify `RECONNECTING` backoff then `STREAMING`.
- Measure FPS against baseline; ensure metrics reported.

Notes (VI): Kịch bản kiểm thử đảm bảo mở luồng, tự khôi phục, và theo dõi chỉ số ổn định.

---

## 2) Motion Detection (Background Subtraction)
- Module: `src/video_processing/simple_processing.py` → class `SimpleMotionDetector`
- Algorithm:
  - Resize frame to `resolution=(256,144)`.
  - Apply MOG2 background subtractor with `history=200`, `varThreshold=32`, `detectShadows=True`.
  - Remove shadows (`mask==127 → 0`), open with 3×3 ellipse kernel.
  - Count white pixels → `motion_pixels`; motion if `motion_pixels > threshold` (default 150) after warmup `start_frames`.
- Output: `{motion_detected, motion_pixels, threshold, frame_count}`.
- Notes (VI): Phát hiện chuyển động bằng nền MOG2, ngưỡng pixel > 150.

---

## 3) Keyframe Detection (Frame Difference + History)
- Module: `src/video_processing/simple_processing.py` → class `SimpleKeyframeDetector`
- Algorithm:
  - Grayscale + Gaussian blur(9×9); compute abs diff with last frame.
  - Normalize `normalized_diff = nonzero(diff) / (H×W)`.
  - Real-time threshold: `is_keyframe ← normalized_diff > min_diff_threshold` (default 0.01).
  - History refinement: require `normalized_diff > 1.5 × mean(last 5 diffs)`.
- Output: `(is_keyframe, normalized_diff)`.
- Notes (VI): Chọn keyframe khi độ khác biệt ảnh vượt ngưỡng và lớn hơn trung bình gần đây.

---

## 4) YOLOv8-Pose Keypoint Extraction
- Module: `src/seizure_detection/yolov8_pose_estimator.py` → class `YOLOv8PoseEstimator`
- Algorithm:
  - Load `yolov8{size}-pose.pt` (`n/s/m/l/x`).
  - Inference → select best person by highest `box.conf`.
  - Threshold: `best_confidence >= confidence_threshold` (default 0.5).
  - Extract 17 COCO keypoints `(x,y,conf)` for best person.
- Data: `keypoint_names` and `skeleton_connections` defined for rendering/consistency.
- Notes (VI): Trích xuất khớp (17 điểm) người có độ tin cậy cao nhất, ngưỡng 0.5.

---

## 5) Fall Detection – Heuristics + Velocity Model
- Module: `src/fall_detection/simple_fall_detector.py` → class `SimpleFallDetector`
- Parameters:
  - `confidence_threshold=0.40`, `min_time_interval=0.15s`, frame buffer `max_buffer_size=5`.
  - Cooldowns: `danger_cooldown=15s`, `standing_up_cooldown=3s`.
  - Repeated sitting window: `10s`, threshold `3` events.
- Preprocessing:
  - Safe bbox conversion (validates `[x1,y1,x2,y2]`, positive size, finite floats).
  - Movement features from first vs last buffered frame: widths/heights, aspect ratios, centers.
- Core features:
  - Aspect ratios: $a_1 = \frac{w_1}{h_1}$, $a_2 = \frac{w_2}{h_2}$, $\Delta a = \frac{a_2}{a_1}$.
  - Center shifts: $\Delta x = |c_{2x}-c_{1x}|$, $\Delta y = |c_{2y}-c_{1y}|$.
  - BBox size change (depth heuristic): $r_{size} = \frac{|w_2h_2-w_1h_1|}{w_1h_1+1}$.
- Priority/filters:
  1. Standing up filter: upward movement `Δy > 300px` and `c2y < c1y` → reject as `standing-up` and start 3s cooldown.
  2. Small posture adjustment: downward `Δy < 60px` → reject, unless sideways pattern (below).
  3. Sideways fall pattern (bypass small-down filter): `Δx > 40px` AND `Δa > 1.2` AND `a2 > 1.4`.
     - Sideways fall confidence: `conf = 0.55 + min((Δa−1.2)*0.25,0.15) + min(Δx/150,0.15) + min(Δy/80,0.10)` capped at `0.90`. Accept if `≥ 0.50`.
  4. Lying down (controlled): final `c2y/H > 0.90` AND `a2 > 1.2` AND `0.85 ≤ Δa ≤ 1.15` AND `Δy < 600px` AND `v_y < 1500 px/s` → reject as `lying-down`.
  5. Rapid downward movement (Strategy 0): `Δy > 70px` and downward.
     - Walking filter: if `Δx > 0.8×Δy` and not `Δy > 150px`, reject.
     - Depth movement filter: `r_size > 1.50` and `150 < Δy < 400` → reject.
     - Already lying filter: if initial `a1 > 1.5` and not `Δy > 250px`, reject.
     - Sitting filter (definitive fall requirement): require `(c2y/H ≥ 0.90) AND (a2 ≥ 1.4)`; else treat as sitting/squatting. Also detect repeated-sitting pattern (≥3 in 10s).
     - Deep bending filter: if `a2 < 0.6` → reject as bending.
- Velocity model:
  - Track `fall_start_time` and `fall_start_position`. After ≥0.1s, compute `fall_duration = t − t0`, `fall_velocity = (c2y−c1y)/(t − t0)`.
  - Controlled descent: if `fall_velocity < 150 px/s` and `fall_duration > 0.5s` → reject.
  - Fall types:
    - `fast_fall` if `v > 400 px/s`
    - `moderate_fall` if `v ≥ 150 px/s`
    - `slow_collapse` if `duration ≥ 1.5s` (stroke-like)
  - Confidence aggregation:
    - Base: `conf0 = min(0.90, 0.50 + Δy/180)` then `conf = min(0.95, conf0 × severity_multiplier)` where `severity_multiplier ∈ {1.3 (slow_collapse), 1.1 (moderate), 1.0 (fast)}`; accept if `conf ≥ 0.50` AND
      either `v ≥ 600 px/s` OR `(c2y/H ≥ 0.90 ∧ a2 ≥ 1.4)`.
- Outputs: `fall_detected`, `confidence`, `fall_type`, `fall_duration`, `fall_velocity`, `category/method` markers.
- Notes (VI): Kết hợp hình học bbox + vận tốc + nhiều bộ lọc (đứng dậy, cúi người, đi ngang, chiều sâu, ngồi nhanh). Dùng cooldown 15s để tránh spam.

---

## 6) Seizure Predictor – Temporal Analysis
- Module: `src/seizure_detection/seizure_predictor.py` → class `SeizurePredictor`
- Parameters: `temporal_window=5`, `smoothing_factor=0.8`, `alert_threshold=0.90`, `warning_threshold=0.80`.
- Temporal buffers: store last N confidences, maintain smoothed value and history.
- Equations:
  - Exponential smoothing: $S_t = \alpha x_t + (1-\alpha) S_{t-1}$ with $\alpha=0.8$.
  - Trend via linear regression slope on window: `slope = polyfit(range(n), history, 1)[0]` with cutoffs `>0.01` increasing, `<−0.01` decreasing.
  - Volatility: `std(history)`; Peak: `max(history)`; Sustained high: `mean(last 10) > warning_threshold`.
- Alert logic:
  - `critical` if `smooth ≥ 0.90` OR `raw ≥ 1.00` (0.90+0.1); track `seizure_duration`.
  - `warning` if `smooth ≥ 0.80`.
  - `normal` otherwise; resets duration.
  - If sustained_high AND trend increasing: at least `warning`.
- Notes (VI): Làm mượt mũ suy, phát hiện xu hướng/độ biến động, đưa ra cảnh báo dựa trên ngưỡng.

---

## 7) Combined Video Pipeline – Orchestration & Cooldowns
- Module: `src/service/advanced_healthcare_pipeline.py` → class `AdvancedHealthcarePipeline`
- Buffers/Throttling:
  - Frame buffer: deque `maxlen=5` to capture pre-event frames.
  - Normal log throttle: only log NORMAL every ~10s; block NORMAL for 180s after a DANGER.
  - Global event cooldown: `45s` between any two events (`_GLOBAL_EVENT_COOLDOWN`).
  - Fall-specific cooldown: `10s` between fall detections.
  - Seizure-specific cooldown: `30s` between seizure detections.
- Keyframe gating: Only perform heavier AI steps on keyframes.
- Event creation strategy:
  1. Create event in DB first (get `event_id`).
  2. Capture snapshots (5 images immediate using buffer + current frame).
  3. Update event with `snapshot_id`.
- Status mapping (5 levels used in pipeline paths): `danger`, `warning`, `suspect`, `normal` (plus internal transitions).
- Notes (VI): Pipeline điều phối cooldown toàn cục + riêng loại, tạo event trước rồi chụp ảnh để có `event_id` liên kết snapshot.

---

## 8) Event Publishing – Severity Mapping, Priority, Captions
- Module: `src/service/emergency_notification_dispatcher.py` → class `HealthcareEventPublisher`
- Severity mapping (fallbacks if config missing):
  - Fall: `high ≥ 0.60`, `medium ≥ 0.40`, else `low`.
  - Seizure: `high ≥ 0.50`, `medium ≥ 0.30`, else `low`.
- Mobile status mapping: `high → danger`, `medium → abnormal_behavior`, `low → normal`.
- Priority level: `high=4`, `medium=3`, `low=2`; `acknowledged` reduces priority; `resolved=0`.
- Creation gating: compare new event priority vs highest active; create only if `≥ current_max` (or `> low` if none).
- Intelligent caption: optional BLIP-based Vietnamese caption; else static template including `(Tin cậy: {confidence:.0%})`.
- Notes (VI): Ánh xạ ngưỡng → mức nguy hiểm, cơ chế ưu tiên để tránh spam, có caption AI tiếng Việt.

---

## 9) Snapshot Storage – MinIO + Metadata Cleaning
- Module: `src/infrastructure/services/snapshot_service.py` → class `SnapshotService`
- Logic:
  - SQLAlchemy session to DB; MinIO integration via `get_minio_service()`.
  - `clean_metadata_for_json`: convert `numpy` types → native Python for JSON (list/float/int/bool).
  - `create_detection_snapshot(camera_id,user_id,event_type,confidence,frame,metadata,event_id)`
    - Upload image to MinIO, create DB rows `snapshots` and `snapshot_images`, return IDs.
- Notes (VI): Ảnh lưu S3/MinIO, metadata chuẩn hóa để JSON hóa trước khi ghi DB.

---

## 10) Alarm Runtime – PostgreSQL LISTEN/NOTIFY + Audio
- Module: `src/infrastructure/services/emergency_alarm_handler_psycopg.py` → class `EmergencyAlarmHandlerPsycopg`
- Channels: `system_alarm_trigger_channel`, `system_alarm_stop_channel` on direct port 5432.
- Deduplication: `processed_events` set; cleanup periodically.
- Trigger processing: `TRIGGER_ALARM` plays alarm indefinitely (duration=0) until stopped.
- Stop processing: `stop_alarm`, then update event `lifecycle_state → RESOLVED` unless it’s an AUTOCALLED transition from DB trigger (ignored for safety).
- Notes (VI): Lắng nghe NOTIFY thời gian thực, bật/tắt còi, cập nhật trạng thái sự kiện an toàn.

---

## 11) Other Supporting Logic
- `src/video_processing/simple_processing.py`:
  - Simple YOLOv8 object detector for persons with `conf` threshold; overlays; aggregates detection stats.
  - Frame saving utilities insert confidence into filenames for traceability.
- `src/service/ai_vision_description_service.py` (if present):
  - BLIP caption → Vietnamese professional caption with context (camera location), used by publisher.
- `src/service/postgresql_healthcare_service.py`:
  - Direct DB operations: `publish_event_detection`, `update_event_snapshot`, queries for alerts.

---

## 12) Key Thresholds Summary (Quick Reference)
- Motion: `motion_pixels > 150` (MOG2).
- Keyframe: `normalized_diff > 0.01` and `> 1.5×avg(last 5)`.
- YOLOv8-Pose person selection: `confidence ≥ 0.5`.
- Fall detection:
  - Rapid: `Δy > 70px` downward; accept if velocity `≥ 600 px/s` OR `(c2y/H ≥ 0.90 ∧ a2 ≥ 1.4)`.
  - Sideways: `Δx > 40px ∧ Δa > 1.2 ∧ a2 > 1.4`; conf formula with caps.
  - Lying down controlled: final near floor, aspect stable, velocity `< 1500 px/s`.
  - Controlled descent reject: `v < 150 px/s` and `duration > 0.5s`.
  - Cooldowns: `danger 15s`, `standing-up 3s`.
- Seizure predictor: smoothing `α=0.8`; `warning ≥ 0.80`, `critical ≥ 0.90`.
- Pipeline cooldowns: global `45s`, fall `10s`, seizure `30s`; normal-log throttle 10s; danger → prevent normal for `180s`.

Giải thích (VI): Bảng tóm tắt nhanh các ngưỡng/điều kiện quan trọng để vận hành và tinh chỉnh.

---

## 13) Event Lifecycle & Snapshots (Operational Flow)
1. Keyframe passes → detectors run.
2. If fall/seizure passes thresholds/cooled down: create DB event (get `event_id`).
3. Capture 5 snapshots (buffer ± current frame) and upload to MinIO.
4. Update event with first `snapshot_id`; publish realtime updates.
5. Alarm control via API or DB triggers; audio handler manages playback/stop.

Notes (VI): Chu trình vận hành khi có sự kiện khẩn: phát hiện → lưu DB → chụp ảnh → phát realtime → còi báo → dừng còi/cập nhật.

---

## 14) Tuning Pointers
- Adjust `SimpleFallDetector` thresholds (Δy, aspect, velocity) per camera height and field-of-view.
- Prefer increasing `global cooldown` for noisy environments; reduce for critical coverage.
- Use caption-based actions to reduce alert fatigue by adding context to DANGER/WARNING.

Giải thích (VI): Gợi ý tinh chỉnh theo môi trường triển khai để giảm báo giả và tối ưu phản ứng.
