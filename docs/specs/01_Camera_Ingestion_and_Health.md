# Camera Ingestion & Health

- Module: [src/camera/simple_camera.py](src/camera/simple_camera.py)

## Component Purpose
- Provide a resilient RTSP camera reader for downstream video processing.
- Maintain stream health metrics (FPS, latency, drops) and auto-recovery.

Giải thích (VI): Thành phần đọc camera RTSP, theo dõi sức khỏe luồng và tự khôi phục khi rớt kết nối.

## Responsibilities
- Open and validate RTSP streams; handle credentials and URL formats.
- Read frames continuously; expose latest frame to the pipeline.
- Detect stalls/drops; run reconnection with backoff.
- Report health stats to the pipeline/logger.

## Inputs & Sources
- RTSP URL (e.g., rtsp://user:pass@ip:port/…).
- Optional camera configuration from config/config.json and validation in src/main.py (`load_camera_config()`, `validate_camera_credentials()`).

## Outputs & Events
- Frames (BGR np.ndarray).
- Health metrics: `is_open`, `fps`, `last_frame_ts`, `drop_count`, `error_count`.
- Status events (log): CONNECTING, STREAMING, RECONNECTING, ERROR.

## Public Interface (Runtime Contract)
- `open()` → attempts connection; sets state to CONNECTING → STREAMING on success.
- `read()` → returns (ok: bool, frame: np.ndarray | None, ts: float); updates FPS/drops.
- `close()` → releases resources; state DISCONNECTED.
- `is_open()` → bool.

Giải thích (VI): Giao diện chạy gồm mở/đọc/đóng luồng và các chỉ số sức khỏe.

## Configuration Parameters
- `rtsp_url`: Camera endpoint with credentials.
- `connect_timeout`: e.g., 5–10s.
- `read_timeout`: e.g., 2–5s before stall considered.
- `retry_policy`: max retries (e.g., 5), `backoff_initial` (2s), `backoff_max` (30s).
- `target_fps`: expected stream FPS (used to flag degradation).
- `resolution_hint`: optional resize for downstream processing.

## State Machine
- States: DISCONNECTED → CONNECTING → STREAMING ↔ DEGRADED → RECONNECTING → STREAMING | ERROR.
- Triggers:
  - CONNECTING→STREAMING: `isOpened()` and first frame received.
  - STREAMING→DEGRADED: FPS < 50% target or read latency > `read_timeout`.
  - DEGRADED→RECONNECTING: consecutive stalls > N (e.g., 3).
  - ANY→ERROR: unrecoverable auth/URL error.

Giải thích (VI): Trạng thái luồng và điều kiện chuyển trạng thái khi rớt hoặc suy giảm.

## Health Metrics & Telemetry
- `fps = frames_read / elapsed_time` (sliding window 5–10s).
- `read_latency = now − last_frame_ts`.
- `stalls = count(read_latency > read_timeout)`.
- `drops = count(read() returns False)`.

## Failure Modes & Recovery
- Authentication/URL error: validate and mask passwords in logs; suggest alternative creds (see `validate_camera_credentials()` in src/main.py).
- Transport drop: exponential backoff reconnect.
- Long stall: force close/reopen capture.
- Codec/format mismatch: fallback to software decode or lower resolution if supported.

Giải thích (VI): Các lỗi thường gặp và cách tự khôi phục để duy trì luồng.

## Security & Privacy
- Never log plain passwords; mask in diagnostics.
- RTSP URLs stored securely; restrict file/system access.

## Monitoring & Alerts
- Warn if `fps < 0.5×target_fps` for > 10s.
- Alert if `stalls ≥ 3` within 30s or `drops` rising continuously.

## Testing Checklist
- Connect to valid RTSP; verify STREAMING within `connect_timeout`.
- Simulate disconnect; verify RECONNECTING backoff then STREAMING.
- Measure FPS against baseline; ensure metrics reported.

Notes (VI): Kịch bản kiểm thử đảm bảo mở luồng, tự khôi phục, và theo dõi chỉ số ổn định.
