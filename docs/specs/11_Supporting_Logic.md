# Other Supporting Logic

## simple_processing.py
- Module: [src/video_processing/simple_processing.py](src/video_processing/simple_processing.py)
- Simple YOLOv8 object detector for persons with `conf` threshold; overlays; aggregates detection stats.
- Frame saving utilities insert confidence into filenames for traceability.

## ai_vision_description_service.py
- Module: [src/service/ai_vision_description_service.py](src/service/ai_vision_description_service.py) (if present)
- BLIP caption → Vietnamese professional caption with context (camera location), used by publisher.

## postgresql_healthcare_service.py
- Module: [src/service/postgresql_healthcare_service.py](src/service/postgresql_healthcare_service.py)
- Direct DB operations: `publish_event_detection`, `update_event_snapshot`, queries for alerts.

Notes (VI): Các logic hỗ trợ nhận diện, mô tả ảnh, và thao tác DB trực tiếp.
