# IV. Software Design Document – 1.1 System Architecture (Drop‑in)

This section is written to be pasted directly into your master document. It summarizes the running architecture derived from the current codebase and includes copy‑ready diagrams.

---

## A. Architectural Overview

The system is an edge‑first, AI‑powered video monitoring pipeline that ingests RTSP camera streams, runs fall and seizure detection, and publishes alerts with snapshots to a PostgreSQL‑backed realtime stack. Images are stored in MinIO/S3, while a lightweight REST API exposes alarm control.

Giải thích (VI): Kiến trúc chạy tại biên (edge), đọc camera RTSP, phát hiện té ngã/co giật, lưu sự kiện vào PostgreSQL, ảnh vào MinIO, và điều khiển cảnh báo qua API.

Key implementation modules:
- Camera ingestion: src/camera/simple_camera.py
- Processing pipeline: src/service/advanced_healthcare_pipeline.py
- Fall detection: src/fall_detection/simple_fall_detector.py
- Seizure predictor: src/seizure_detection/seizure_predictor.py
- Event publisher: src/service/emergency_notification_dispatcher.py
- Snapshot/MinIO: src/infrastructure/services/snapshot_service.py
- Alarm API: src/alarm_fastapi_server.py
- PG LISTEN/NOTIFY handler: src/infrastructure/services/emergency_alarm_handler_psycopg.py

---

## B. Component View

```mermaid
flowchart LR
    CAM[IP Cameras (RTSP)] --> ING[Camera Ingestion\n(simple_camera)]
    ING --> VP[Video Processing]
    VP --> FD[Fall Detector\n(simple_fall_detector)]
    VP --> SD[Seizure Predictor\n(seizure_predictor)]
    FD --> PIPE[Advanced Healthcare Pipeline]
    SD --> PIPE
    PIPE --> PUB[Event Publisher]
    PIPE --> SNAP[Snapshot Service]

    SNAP -->|upload| MINIO[(MinIO/S3)]
    PUB -->|create/update| PG[(PostgreSQL)]

    APIS[Alarm FastAPI] -->|control/status| PUB
    HANDLER[PG LISTEN/NOTIFY Handler] -->|realtime| PG
    HANDLER --> AUDIO[Audio Alert Service]

    PG --> APPS[Mobile/Web Apps\n(FCM/Realtime)]
```

Giải thích (VI): Camera → Ingestion → Xử lý khung hình → Bộ phát hiện (ngã/co giật) → Pipeline tổng hợp → Lưu sự kiện/snapshot → Gửi realtime/FCM → Ứng dụng. API dùng để bật/tắt alarm; handler lắng nghe PG và phát âm thanh.

---

## C. Runtime Flow (Detection → Alert)

```mermaid
sequenceDiagram
    participant CAM as Camera (RTSP)
    participant PIPE as Advanced Pipeline
    participant FD as Fall Detector
    participant SD as Seizure Predictor
    participant SNAP as Snapshot Service
    participant PUB as Event Publisher
    participant DB as PostgreSQL
    participant MIN as MinIO/S3
    participant HND as PG Handler (LISTEN)
    participant APP as Caregiver App (FCM)

    CAM->>PIPE: Frame
    PIPE->>FD: Analyze posture/motion
    PIPE->>SD: Temporal confidence update
    alt Fall/Seizure exceeds thresholds
        PIPE->>SNAP: Create snapshot(s)
        SNAP->>MIN: Upload images
        PIPE->>PUB: Publish event (severity, metadata)
        PUB->>DB: Insert/Update event records
        DB-->>HND: NOTIFY (trigger/stop)
        HND->>APP: Push/notify; play audio if needed
    else No critical event
        PIPE-->>PIPE: Normal log/throttled updates
    end
```

Giải thích (VI): Khi vượt ngưỡng, pipeline chụp ảnh, lưu MinIO, ghi sự kiện vào PostgreSQL; handler nhận NOTIFY để đẩy thông báo và âm thanh.

---

## D. Deployment Topology

```mermaid
flowchart TB
    subgraph Edge Node
      PROC[Detection Services\n(main.py / pipeline)]
      API[Alarm FastAPI]
      HND[PG Handler]
    end

    PROC -->|RTSP| CAMS[IP Cameras]
    PROC -->|JDBC| PG[(PostgreSQL)]
    PROC -->|S3 API| MINIO[(MinIO/S3)]
    API --> PG
    HND --> PG
    HND --> AUDIO[Speaker/Buzzer]
    APPS[Mobile/Web] -->|Realtime/FCM| PG
```

Giải thích (VI): Các dịch vụ chạy trên máy edge; kết nối tới PostgreSQL và MinIO; ứng dụng nhận realtime/FCM.

---

## E. Data and Storage

Primary entities (based on models/generated):
- Events and event_detections: core detection records with severity/status.
- Snapshots and snapshot_images: link images to detections; images stored in MinIO.
- Activity logs and notifications: audit trail and outbound messages.
- Cameras, users, rooms: configuration and ownership.

Giải thích (VI): Dữ liệu gồm sự kiện, ảnh, log, thông báo và cấu hình camera/người dùng; ảnh lưu trên MinIO, bản ghi trong PostgreSQL.

---

## F. Key Configurations and Thresholds

- Fall detection (`simple_fall_detector`): tuned for fewer false positives; velocity and cooldown heuristics; default confidence threshold ≈ 0.40.
- Seizure predictor (`seizure_predictor`): temporal window 5; smoothing 0.8; warning ≥ 0.80; alert ≥ 0.90.
- Severity mapping (`emergency_notification_dispatcher`): maps confidence → {low, medium, high} → mobile status.
- Normal event throttling and danger cooldowns are enforced in the pipeline to reduce alert fatigue.

Giải thích (VI): Ngưỡng và cooldown được điều chỉnh để giảm báo giả; co giật dùng làm mượt theo thời gian; mức độ nặng được ánh xạ ra trạng thái di động.

---

## G. External Interfaces

- Alarm REST API: POST /api/alarm/control, GET /api/alarm/status (see src/alarm_fastapi_server.py)
- PostgreSQL LISTEN/NOTIFY channels: trigger/stop (see emergency_alarm_handler_psycopg.py)
- Object storage: S3‑compatible (MinIO) for snapshot images
- FCM/Realtime: downstream delivery to caregiver apps

Giải thích (VI): API điều khiển alarm, kênh PG realtime, kho S3 cho ảnh và FCM cho thông báo.

---

## H. Architectural Constraints / Non‑Goals

- Assumes RTSP‑capable cameras and stable LAN; intermittent drops handled via reconnect.
- Not a full EMR/clinical records system; focuses on event detection and alerting.
- On‑device recognition is prioritized; cloud offload optional and deployment‑specific.

Giải thích (VI): Yêu cầu camera RTSP và mạng ổn định; không thay thế hệ thống hồ sơ y tế; ưu tiên xử lý tại biên.
