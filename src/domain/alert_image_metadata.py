from dataclasses import dataclass
from datetime import datetime

@dataclass
class AlertImageMetadata:
    alert_type: str
    confidence: float
    keyframe_confidence: float
    timestamp: str
