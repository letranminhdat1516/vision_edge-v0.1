from dataclasses import dataclass
from typing import Optional

@dataclass
class Statistics:
    frames_processed: int
    persons_detected: int
    total_frames: int
    keyframes_detected: int
    motion_frames: int
    start_time: float
    last_detection_time: Optional[float]
    fall_detections: int
    last_fall_time: Optional[float]
    fall_confidence_avg: float
    fall_false_positives: int
    seizure_detections: int
    last_seizure_time: Optional[float]
    seizure_confidence_avg: float
    seizure_warnings: int
    pose_extraction_failures: int
    critical_alerts: int
    total_alerts: int
    last_alert_time: Optional[float]
    alert_type: str
