# Snapshot Storage – MinIO + Metadata Cleaning

- Module: [src/infrastructure/services/snapshot_service.py](src/infrastructure/services/snapshot_service.py) → class `SnapshotService`

## Logic
- SQLAlchemy session to DB; MinIO integration via `get_minio_service()`.
- `clean_metadata_for_json`: convert numpy types → native Python for JSON (list/float/int/bool).
- `create_detection_snapshot(camera_id,user_id,event_type,confidence,frame,metadata,event_id)`
  - Upload image to MinIO, create DB rows `snapshots` and `snapshot_images`, return IDs.

Notes (VI): Ảnh lưu S3/MinIO, metadata chuẩn hóa để JSON hóa trước khi ghi DB.
