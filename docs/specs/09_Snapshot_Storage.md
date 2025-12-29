# Snapshot Storage – MinIO + Metadata Cleaning

- **Module**: [src/infrastructure/services/snapshot_service.py](src/infrastructure/services/snapshot_service.py) → class `SnapshotService`
- **Mục đích**: Lưu trữ ảnh detection vào MinIO (S3-compatible) và quản lý metadata trong PostgreSQL.

---

## 1. Constructor

```python
def __init__(self, database_url: str):
    # Database connection
    self.engine = create_engine(database_url)
    self.SessionLocal = sessionmaker(bind=self.engine)

    # MinIO service
    self.minio_service = get_minio_service()

    # Test connection
    if self.minio_service.test_connection():
        logger.info("MinIO connection verified")
```

---

## 2. Metadata Cleaning for JSON

### Problem

NumPy types không JSON serializable:

```python
# Fails:
json.dumps({'confidence': np.float64(0.85)})  # TypeError!
json.dumps({'bbox': np.array([1,2,3,4])})     # TypeError!
```

### Solution: `clean_metadata_for_json()`

```python
def clean_metadata_for_json(data: Any) -> Any:
    """Convert numpy types → Python native types"""

    if isinstance(data, dict):
        return {key: clean_metadata_for_json(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [clean_metadata_for_json(item) for item in data]

    elif isinstance(data, np.ndarray):
        return data.tolist()          # np.array → list

    elif isinstance(data, np.floating):
        return float(data)            # np.float64 → float

    elif isinstance(data, np.integer):
        return int(data)              # np.int64 → int

    elif isinstance(data, np.bool_):
        return bool(data)             # np.bool_ → bool

    else:
        return data
```

### Type Conversion Table

| Input Type   | Output Type | Example                   |
| ------------ | ----------- | ------------------------- |
| `np.ndarray` | `list`      | `[1,2,3,4]` → `[1,2,3,4]` |
| `np.float64` | `float`     | `0.85` → `0.85`           |
| `np.int64`   | `int`       | `17` → `17`               |
| `np.bool_`   | `bool`      | `True` → `True`           |
| `dict`       | `dict`      | Recursive clean           |
| `list`       | `list`      | Recursive clean           |

---

## 3. Main Method: `create_detection_snapshot()`

### Signature

```python
def create_detection_snapshot(
    camera_id: str,                    # Camera UUID
    user_id: str,                      # User UUID
    event_type: str,                   # 'fall', 'seizure', etc.
    confidence: float,                 # Detection confidence
    frame: np.ndarray,                 # OpenCV frame (BGR)
    metadata: Optional[Dict] = None,   # Additional metadata
    event_id: Optional[str] = None     # Link to event_detections
) -> Tuple[str, str]:                  # (snapshot_id, image_id)
```

### Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Generate UUIDs                                          │
│  ┌─────────────────────────────────────────────┐                │
│  │ snapshot_id = uuid4()                       │                │
│  │ image_id = uuid4()                          │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Upload Image to MinIO                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │ minio_service.upload_frame_image(           │                │
│  │     frame, camera_id, event_type,           │                │
│  │     confidence, user_id, metadata           │                │
│  │ )                                           │                │
│  │ → Returns: (object_name, cloud_url, size)   │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Prepare Metadata                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │ metadata_dict = {                           │                │
│  │     'event_type': event_type,               │                │
│  │     'confidence': confidence,               │                │
│  │     'detection_time': now.isoformat(),      │                │
│  │     'event_id': event_id,  # ⭐ CRITICAL!   │                │
│  │     ...custom metadata                      │                │
│  │ }                                           │                │
│  │ cleaned = clean_metadata_for_json(metadata) │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Create Database Records                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │ A. Create Snapshots record                  │                │
│  │    - snapshot_id, camera_id, user_id        │                │
│  │    - metadata_ (JSON), capture_type         │                │
│  │    - captured_at, processed_at              │                │
│  │                                             │                │
│  │ B. Create SnapshotImages record             │                │
│  │    - image_id, snapshot_id                  │                │
│  │    - image_path (MinIO object name)         │                │
│  │    - cloud_url, file_size                   │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Return IDs                                              │
│  return (snapshot_id, image_id)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Capture Type Mapping

```python
capture_type_mapping = {
    'seizure': 'alert_triggered',
    'fall': 'alert_triggered',
    'manual': 'manual',
    'motion': 'motion_triggered',
    'scheduled': 'scheduled'
}

db_capture_type = capture_type_mapping.get(event_type, 'alert_triggered')
```

### Database Enum Values

| Capture Type       | When Used              |
| ------------------ | ---------------------- |
| `alert_triggered`  | Fall/Seizure detection |
| `manual`           | User manually captures |
| `motion_triggered` | Motion-based capture   |
| `scheduled`        | Periodic snapshots     |

---

## 5. Database Schema

### `snapshots` Table

```sql
CREATE TABLE snapshots (
    snapshot_id UUID PRIMARY KEY,
    camera_id UUID NOT NULL,
    user_id UUID NOT NULL,
    metadata_ JSONB,              -- Cleaned metadata
    capture_type VARCHAR(50),     -- 'alert_triggered', etc.
    captured_at TIMESTAMP,
    processed_at TIMESTAMP,
    is_processed BOOLEAN DEFAULT TRUE
);
```

### `snapshot_images` Table

```sql
CREATE TABLE snapshot_images (
    image_id UUID PRIMARY KEY,
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    is_primary BOOLEAN DEFAULT TRUE,
    image_path VARCHAR(500),      -- MinIO object name
    cloud_url VARCHAR(1000),      -- Public/Signed URL
    file_size VARCHAR(50),
    created_at TIMESTAMP
);
```

---

## 6. MinIO Object Structure

### Object Naming Convention

```
{bucket}/{user_id}/{camera_id}/{event_type}/{timestamp}_{confidence}.jpg

Example:
healthcare-snapshots/
  ├── user-abc-123/
  │   ├── camera-xyz-789/
  │   │   ├── fall/
  │   │   │   ├── 2024-01-15T10-30-45_0.85.jpg
  │   │   │   └── 2024-01-15T10-30-46_0.82.jpg
  │   │   └── seizure/
  │   │       └── 2024-01-15T11-00-00_0.91.jpg
```

### Upload Result

```python
upload_result = minio_service.upload_frame_image(...)
# Returns:
(
    object_name,  # "user-abc/camera-xyz/fall/2024-01-15T10-30-45_0.85.jpg"
    cloud_url,    # "https://minio.example.com/healthcare/..."
    file_size     # 125430 (bytes)
)
```

---

## 7. Event Linking

### Critical: event_id in Metadata

```python
# ⭐ CRITICAL: Link snapshot to event
if event_id:
    metadata_dict['event_id'] = event_id
    logger.info(f"🔗 Linking snapshot to event: {event_id}")
else:
    logger.warning("⚠️ No event_id - snapshot NOT linked!")
```

### Why Link?

1. **Traceability**: Query all snapshots for an event
2. **Evidence**: View images associated with detection
3. **Mobile App**: Display relevant images when showing alert

### Query Example

```sql
SELECT si.cloud_url, s.metadata_
FROM snapshots s
JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
WHERE s.metadata_->>'event_id' = 'abc-123-xyz';
```

---

## 8. Error Handling

```python
try:
    # Upload + DB operations
    db.add(snapshot)
    db.commit()

    db.add(snapshot_image)
    db.commit()

except Exception as e:
    db.rollback()  # Rollback on any error
    logger.error(f"Error creating snapshot: {e}")
    raise
finally:
    db.close()
```

### Failure Scenarios

| Scenario              | Handling                               |
| --------------------- | -------------------------------------- |
| MinIO upload fails    | Raise exception, no DB records created |
| Snapshot insert fails | Rollback, no image record              |
| Image insert fails    | Rollback snapshot too                  |
| Any error             | Full rollback, log error               |

---

## 9. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Event                               │
│                (Fall/Seizure detected)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline calls:                                                 │
│  snapshot_service.create_detection_snapshot(                     │
│      camera_id, user_id, 'fall', 0.85, frame,                   │
│      metadata={...}, event_id='abc-123'                         │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  MinIO Upload           │     │  PostgreSQL Insert      │
│  - Encode JPEG          │     │  - snapshots table      │
│  - Upload to bucket     │     │  - snapshot_images      │
│  - Get URL              │     │     table               │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Return (snapshot_id, image_id)                                  │
│  - Used to update event_detections.snapshot_id                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Configuration

### Environment Variables

```bash
MINIO_ENDPOINT=minio.example.com:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET=healthcare-snapshots
MINIO_SECURE=true
```

### Connection Test

```python
if minio_service.test_connection():
    logger.info("MinIO ready")
else:
    logger.warning("MinIO test failed, uploads may fail")
```

---

## Notes (VI)

Ảnh detection upload lên MinIO (S3-compatible), metadata chuẩn hóa (numpy → Python native) trước khi lưu JSON vào PostgreSQL. Cấu trúc: `snapshots` (metadata) + `snapshot_images` (paths). **QUAN TRỌNG**: Luôn truyền `event_id` để link snapshot với event detection.
