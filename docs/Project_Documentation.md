# Project Documentation for Vision Edge Healthcare System

> Note: This document is derived from the current codebase and repository structure. Some planning details (timeline, budget) are proposed and should be validated with stakeholders.

---

## 1. Project Overview

Vision Edge Healthcare System is an AI-powered, real-time monitoring platform that detects critical patient events such as falls and seizures using computer vision. It integrates camera streams, lightweight AI models, and a healthcare event pipeline to generate alerts, capture snapshots, store data, and notify caregivers via real-time channels and push notifications.

- Purpose: Provide continuous, automated patient safety monitoring with low latency and high reliability in healthcare environments (hospitals, nursing homes, home care).
- Scope: Multi-camera ingestion, fall and seizure detection, event lifecycle handling, snapshot storage (MinIO/S3), alerting (audio/FCM), REST API for alarm control, and PostgreSQL-based realtime integrations.
- Intended Outcome: Reduce response time to emergencies (< 3 seconds), lower staff workload, and maintain structured records of critical events with visual evidence.

Vietnamese explanation: Hệ thống giám sát dùng AI để phát hiện té ngã/co giật theo thời gian thực từ camera, lưu ảnh bằng chứng, ghi sự kiện vào CSDL và gửi cảnh báo ngay cho người chăm sóc. Mục tiêu là giảm thời gian phản ứng, giảm tải nhân lực và tăng độ an toàn cho bệnh nhân.

---

## 2. Project Goals and Objectives (SMART)

- Reduce detection-to-alert latency: Detect and publish critical alerts within ≤ 3 seconds in typical deployment conditions.
- Achieve reliable fall detection: ≥ 85% precision/recall on representative test data and live pilots, measured monthly.
- Improve seizure detection reliability: ≥ 72% model-level accuracy with temporal smoothing; operational false alarm rate ≤ 5% per 24h per camera.
- Ensure system uptime: ≥ 99% for core services (ingestion, detection, alert dispatch) measured monthly.
- Support multi-camera deployments: Scale to at least 8 concurrent camera streams on a mid-range edge device/server.
- Deliver actionable alerts: Provide snapshots and a human-readable caption for ≥ 95% of DANGER events.

Vietnamese explanation: Mục tiêu cụ thể gồm độ trễ cảnh báo ≤ 3 giây, độ chính xác phát hiện ngã ≥ 85%, co giật ≥ 72%, độ ổn định ≥ 99%, hỗ trợ nhiều camera, và mỗi cảnh báo nguy hiểm có ảnh/caption rõ ràng.

---

## 3. Stakeholders

- Patients and Care Recipients: Primary beneficiaries; safety monitoring and rapid response.
- Caregivers and Healthcare Staff: Receive and act on alerts; confirm, acknowledge, or resolve events.
- Clinical Administrators: Oversee deployment, policy, and compliance; analyze event logs.
- Engineering Team: Develop and maintain vision pipeline, models, services, and integrations.
- IT/Operations: Provision hardware, networks, databases, storage, and monitoring.
- Compliance and Security Officers: Ensure privacy, access control, and data protection requirements.
- External Vendors/Partners: Camera providers, cloud/infra providers (S3/MinIO, PostgreSQL).

Vietnamese explanation: Bên liên quan gồm người bệnh, người chăm sóc, quản trị y tế, team kỹ thuật, IT/ops, bộ phận tuân thủ/bảo mật và nhà cung cấp thiết bị/hạ tầng.

---

## 4. System Requirements

### 4.1 Functional Requirements

- Camera ingestion: Connect to RTSP/stream sources; support multiple concurrent cameras.
- Fall detection: Analyze frames to detect falls; categorize severity and type (e.g., slow collapse).
- Seizure detection: Temporal analysis across recent frames using pose/keypoint sequences.
- Event lifecycle: Create, update, and resolve events with severity and status levels.
- Snapshot capture: Store images associated with detections; multi-angle/frame-buffer support.
- Alerting:
  - Real-time notifications via PostgreSQL channels and/or FCM push.
  - Audio alerts for on-prem environments.
- Alarm control API: REST endpoints to trigger/stop alarms and query status.
- Storage and persistence: Write events, snapshots, and metadata to PostgreSQL; upload images to MinIO/S3.
- Configuration: Load runtime thresholds and system parameters from config.

Code references: core pipeline and services live in src/service and src/infrastructure (e.g., advanced pipeline, event publisher, alarm API, snapshot service, MinIO integration).

Vietnamese explanation: Chức năng chính gồm đọc camera, phát hiện ngã/co giật, quản lý vòng đời sự kiện, chụp ảnh, gửi cảnh báo realtime/FCM, API điều khiển alarm, và lưu trữ vào PostgreSQL/MinIO.

### 4.2 Non-Functional Requirements

- Performance: End-to-end alert latency ≤ 3s for DANGER events; processing ≥ 15–30 FPS per stream depending on model and hardware.
- Reliability: Auto-reconnect to camera streams; resilient to transient failures; idempotent event publishing.
- Scalability: Horizontal camera scaling; modular model backends; configurable thresholds per camera.
- Security: Encrypted transport where applicable; credentials protected; least-privilege DB/storage access.
- Privacy: Limit retention of snapshots; PII handling aligned with local regulations; access control for data.
- Observability: Structured logging, health checks, and metrics for core services.
- Portability: Deployable via Docker/docker-compose on edge or server; minimal platform dependencies.

Vietnamese explanation: Phi chức năng gồm hiệu năng (độ trễ thấp), độ tin cậy (tự khôi phục), mở rộng (nhiều camera), bảo mật/riêng tư, khả năng quan sát (log/metrics), và triển khai linh hoạt.

---

## 5. Project Timeline (Proposed)

- Phase 1 – Requirements & Architecture (2 weeks): Confirm use cases, thresholds, compliance constraints; finalize architecture and KPIs.
- Phase 2 – Prototype & Single-Camera MVP (3 weeks): Single stream, fall detection + snapshots + local alerts.
- Phase 3 – Multi-Camera & Seizure Module (3 weeks): Add seizure predictor and multi-stream orchestration.
- Phase 4 – Integrations & Storage (3 weeks): PostgreSQL events, MinIO snapshots, FCM push, alarm API hardening.
- Phase 5 – Testing & Pilot (4 weeks): Benchmarks, edge cases, security review; pilot in a limited ward/home.
- Phase 6 – Rollout & Training (1 week): Deployment playbooks, caregiver training, monitoring 
  dashboards.

Total: ~16 weeks. Adjust per resource availability and regulatory review.

Vietnamese explanation: Lộ trình dự kiến 16 tuần qua các giai đoạn: yêu cầu/kiến trúc, MVP 1 camera, mở rộng nhiều camera + co giật, tích hợp CSDL/MinIO/FCM, kiểm thử/pilot, rồi triển khai chính thức.

---

## 6. Risk Assessment

- False positives/negatives: Misclassification due to occlusion, lighting, or unusual postures.
  - Mitigation: Temporal smoothing, per-camera threshold tuning, human-in-the-loop acknowledgement.
- Camera reliability: RTSP disconnects or authentication issues.
  - Mitigation: Reconnect logic, credential validation, camera health checks.
- Privacy and compliance: Storing identifiable images and event data.
  - Mitigation: Retention policies, access controls, encryption at rest/in transit.
- Performance bottlenecks: Model inference latency at higher camera counts.
  - Mitigation: Model selection (YOLOv8-pose vs MoveNet), batching, GPU/accelerator utilization.
- Infrastructure failures: DB/storage downtime.
  - Mitigation: Backoff/retry, local buffering, monitoring/alerts, DR procedures.
- Operational burden: Alert fatigue in busy environments.
  - Mitigation: Cooldowns, severity mapping, contextual captions, better UX.

Vietnamese explanation: Rủi ro gồm sai nhầm phân loại, mất kết nối camera, vấn đề riêng tư, nghẽn hiệu năng, downtime hạ tầng, và quá tải cảnh báo; có chiến lược giảm thiểu tương ứng.

---

## 7. Resources

- Human Resources: CV/ML engineers, backend engineers, DevOps/infra, QA, pilot site coordinators, support.
- Hardware: IP cameras with RTSP; edge device or server with CPU/GPU/TPU as needed.
- Software: Python 3.10+, OpenCV, FastAPI, PostgreSQL, MinIO/S3, YOLOv8/MoveNet/MediaPipe/VSViG, Docker.
- Services/Accounts: PostgreSQL instance, MinIO/S3 bucket, FCM project, secure secrets management.
- Documentation & Ops: Runbooks, monitoring dashboards, incident response procedures.

Vietnamese explanation: Nguồn lực gồm nhân sự kỹ thuật/QA/vận hành, camera và máy chủ/edge, phần mềm/thư viện, dịch vụ DB/MinIO/FCM và tài liệu vận hành.

---

## 8. Project Budget (Estimates)

- Hardware:
  - Cameras: $100–$300 per unit × N units
  - Edge server (CPU/GPU): $1,500–$4,000 per site (spec-dependent)
- Infrastructure:
  - PostgreSQL hosting: $50–$200/month (or self-hosted)
  - Object storage (MinIO/S3): $20–$150/month depending on volume
  - Push notifications (FCM): typically free; associated backend costs apply
- Personnel (initial 4 months):
  - 1–2 CV/ML engineers, 1 backend, 1 DevOps, 0.5 QA: $80k–$180k (region-dependent)
- Contingency: 10–20%

Vietnamese explanation: Ngân sách ước tính gồm thiết bị camera, máy chủ/edge, hạ tầng DB/lưu trữ, nhân sự trong 4 tháng và khoản dự phòng. Số liệu cần điều chỉnh theo quy mô/thị trường.

---

## 9. Quality Assurance

- Testing Strategy:
  - Unit/integration tests for services (pipeline, event publisher, alarm API, snapshot uploads).
  - Performance tests: FPS/latency under load; multi-camera scenarios.
  - Dataset/field validation for fall and seizure cases; confusion matrix tracking.
- Processes:
  - Code reviews; CI for linting/tests; reproducible builds with Docker.
  - Staging environment matching production parameters.
- Monitoring:
  - Health checks, structured logs, metrics, alerting on service degradation.
  - Periodic accuracy audits and threshold recalibration per camera/site.

Vietnamese explanation: Đảm bảo chất lượng gồm test đơn vị/tích hợp, benchmark hiệu năng, xác thực trên dữ liệu thực tế, quy trình review/CI, theo dõi sức khỏe dịch vụ và hiệu chỉnh định kỳ.

---

## 10. Conclusion

Vision Edge Healthcare System delivers real-time, AI-driven detection of critical patient events with an end-to-end pipeline for alerts, storage, and integrations. It is designed to be modular, scalable, and privacy-conscious, improving patient safety and operational efficiency. With appropriate deployment, monitoring, and continuous calibration, the system can significantly reduce response times and enhance caregiver effectiveness.

Vietnamese explanation: Hệ thống mang lại phát hiện thời gian thực, quy trình cảnh báo đầy đủ và tích hợp hạ tầng, giúp tăng an toàn và hiệu quả chăm sóc. Khi triển khai đúng kèm giám sát/hiệu chỉnh, hệ thống giảm thời gian phản ứng và hỗ trợ người chăm sóc tốt hơn.

---

Change management: All updates to scope, requirements, or thresholds should be versioned, reviewed, and communicated to stakeholders via release notes and operational bulletins.
