# Other Supporting Logic

- **Mục đích**: Các module hỗ trợ cho pipeline chính: object detection đơn giản, AI image captioning, và database operations.

---

## 1. simple_processing.py

- **Module**: [src/video_processing/simple_processing.py](src/video_processing/simple_processing.py)
- **Classes**: `SimpleObjectDetector`, `SimpleKeyframeDetector`, `SimpleFrameSaver`

### 1.1 SimpleObjectDetector

```python
class SimpleObjectDetector:
    """YOLOv8 object detector for persons"""

    def __init__(self, confidence_threshold=0.5):
        self.model = YOLO('yolov8n.pt')  # Nano model for speed
        self.confidence_threshold = confidence_threshold
```

#### Key Methods

```python
def detect_persons(frame) -> List[Dict]:
    """Detect all persons in frame"""
    results = model(frame, classes=[0])  # class 0 = person

    return [
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': 'person'
        }
        for box in results...
    ]
```

#### Overlay Drawing

```python
def draw_detections(frame, detections) -> np.ndarray:
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"Person {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10), ...)

    return frame
```

### 1.2 Detection Statistics

```python
def get_stats() -> Dict:
    return {
        'total_detections': 1500,
        'avg_confidence': 0.72,
        'max_confidence': 0.95,
        'min_confidence': 0.51,
        'persons_per_frame_avg': 1.2
    }
```

---

## 2. SimpleFrameSaver

```python
class SimpleFrameSaver:
    """Save important frames to disk"""

    def __init__(self, base_path="data/saved_frames", max_files_per_folder=1000):
        self.keyframes_path = os.path.join(base_path, "keyframes")
        self.detections_path = os.path.join(base_path, "detections")
        self.alerts_path = os.path.join(base_path, "alerts")
```

### Folder Structure

```
data/saved_frames/
├── keyframes/      # Important frames (motion detected)
├── detections/     # Frames with person detected
└── alerts/         # Fall/Seizure alert frames
```

### Filename Convention

```python
# Include confidence in filename for traceability
filename = f"{timestamp}_{event_type}_conf{confidence:.2f}.jpg"
# Example: "2024-01-15_10-30-45_fall_conf0.85.jpg"
```

### Auto-Cleanup

```python
def _cleanup_old_files(folder_path):
    """Keep only max_files_per_folder most recent files"""
    files = sorted(folder_path.glob("*.jpg"), key=lambda p: p.stat().st_ctime)

    if len(files) > max_files_per_folder:
        for f in files[:-max_files_per_folder]:
            f.unlink()  # Delete oldest
```

---

## 3. ai_vision_description_service.py

- **Module**: [src/service/ai_vision_description_service.py](src/service/ai_vision_description_service.py) (if present)
- **Class**: `ProfessionalCaptionPipeline`

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Image Path + Context (event_type, camera_name)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BLIP Model (Image Captioning)                                   │
│  "A person lying on the floor near a sofa"                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Translation (English → Vietnamese)                              │
│  "Một người nằm trên sàn gần ghế sofa"                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Context Enhancement                                             │
│  Add camera location, event type, confidence                     │
│  "📍 Phòng khách: Một người nằm trên sàn..."                     │
└─────────────────────────────────────────────────────────────────┘
```

### API

```python
def generate_professional_caption(
    image_path: str,
    event_type: str = None,
    camera_name: str = None
) -> Tuple[str, Dict]:
    """
    Returns:
        vietnamese_caption: "Một người nằm trên sàn..."
        metadata: {
            'success': True,
            'english_caption': "A person lying...",
            'model': 'BLIP-base'
        }
    """
```

### Fallback Handling

```python
if not IMAGE_CAPTION_AVAILABLE:
    # Fallback to static template
    return f"Phát hiện {event_type} tại camera {camera_name}"
```

---

## 4. postgresql_healthcare_service.py

- **Module**: [src/service/postgresql_healthcare_service.py](src/service/postgresql_healthcare_service.py)
- **Class**: `PostgreSQLHealthcareService`

### Core Methods

#### Event Publishing

```python
def publish_event_detection(event_data: Dict) -> Dict:
    """Insert new event into event_detections table"""

    event_id = str(uuid.uuid4())

    cursor.execute("""
        INSERT INTO event_detections
        (event_id, event_type, confidence_score, camera_id, user_id,
         detection_data, lifecycle_state, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, 'DETECTED', NOW())
        RETURNING event_id
    """, (event_id, event_type, confidence, camera_id, user_id, json.dumps(data)))

    return {'event_id': event_id, 'success': True}
```

#### Snapshot Linking

```python
def update_event_snapshot(event_id: str, snapshot_id: str) -> bool:
    """Link snapshot to event after creation"""

    cursor.execute("""
        UPDATE event_detections
        SET snapshot_id = %s, updated_at = NOW()
        WHERE event_id = %s
    """, (snapshot_id, event_id))

    return cursor.rowcount > 0
```

#### Alert Queries

```python
def get_user_alerts(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent alerts for user"""

    cursor.execute("""
        SELECT ed.*, s.metadata_, si.cloud_url
        FROM event_detections ed
        LEFT JOIN snapshots s ON ed.snapshot_id = s.snapshot_id
        LEFT JOIN snapshot_images si ON s.snapshot_id = si.snapshot_id
        WHERE ed.user_id = %s
          AND ed.lifecycle_state NOT IN ('RESOLVED', 'EXPIRED')
        ORDER BY ed.created_at DESC
        LIMIT %s
    """, (user_id, limit))

    return cursor.fetchall()
```

#### Camera Name Lookup

```python
def _get_camera_name(camera_id: str) -> Optional[str]:
    """Get camera name for context in captions"""

    cursor.execute("""
        SELECT camera_name FROM cameras WHERE camera_id = %s
    """, (camera_id,))

    result = cursor.fetchone()
    return result['camera_name'] if result else None
```

### Connection Pool

```python
class PostgreSQLHealthcareService:
    def __init__(self):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=database_url
        )

    def get_connection(self):
        return self.pool.getconn()

    def return_connection(self, conn):
        self.pool.putconn(conn)
```

---

## 5. Integration Points

### Pipeline → Supporting Services

```
AdvancedHealthcarePipeline
    │
    ├──▶ SimpleObjectDetector.detect_persons()
    │    └── YOLOv8 person detection
    │
    ├──▶ SimpleKeyframeDetector.is_keyframe()
    │    └── Frame difference analysis
    │
    ├──▶ SimpleFrameSaver.save_alert()
    │    └── Local backup of alert images
    │
    ├──▶ ProfessionalCaptionPipeline.generate_caption()
    │    └── BLIP → Vietnamese caption
    │
    └──▶ PostgreSQLHealthcareService.publish_event_detection()
         └── Database operations
```

---

## 6. Error Handling Patterns

### Detection Service

```python
try:
    results = model(frame)
except Exception as e:
    logger.error(f"Detection failed: {e}")
    return []  # Empty list, pipeline continues
```

### Database Service

```python
try:
    cursor.execute(query, params)
    conn.commit()
except psycopg2.Error as e:
    conn.rollback()
    logger.error(f"DB error: {e}")
    raise
finally:
    self.return_connection(conn)
```

### Caption Service

```python
try:
    caption, metadata = pipeline.generate_caption(image_path)
except Exception as e:
    logger.warning(f"Caption failed: {e}")
    return static_fallback_message()  # Graceful degradation
```

---

## Notes (VI)

Các module hỗ trợ bao gồm: (1) YOLOv8 object detector đơn giản cho person detection, (2) Frame saver với filename có confidence để traceability, (3) BLIP caption pipeline để mô tả ảnh bằng tiếng Việt, (4) PostgreSQL service cho các thao tác database trực tiếp. Tất cả đều có error handling để không làm crash pipeline chính.
