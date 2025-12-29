# Event Lifecycle & Snapshots (Operational Flow)

## Flow
1. Keyframe passes → detectors run.
2. If fall/seizure passes thresholds/cooled down: create DB event (get `event_id`).
3. Capture 5 snapshots (buffer ± current frame) and upload to MinIO.
4. Update event with first `snapshot_id`; publish realtime updates.
5. Alarm control via API or DB triggers; audio handler manages playback/stop.

Notes (VI): Chu trình vận hành khi có sự kiện khẩn: phát hiện → lưu DB → chụp ảnh → phát realtime → còi báo → dừng còi/cập nhật.
