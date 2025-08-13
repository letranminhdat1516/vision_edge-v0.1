from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class DetectionResult:
    fall_detected: bool
    fall_confidence: float
    seizure_detected: bool
    seizure_confidence: float
    seizure_ready: bool
    alert_level: str
    keypoints: Optional[Any]
    emergency_type: Optional[str]
