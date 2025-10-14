# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class EventDetections:
    event_id: Optional[str] = None
    snapshot_id: str
    user_id: str
    camera_id: str
    event_type: str
    detected_at: datetime
    created_at: datetime
    confirmation_state: str
    notes: Optional[str] = None
    event_description: Optional[str] = None
    detection_data: Optional[dict[str, Any]] = None
    ai_analysis_result: Optional[dict[str, Any]] = None
    confidence_score: Optional[float] = None
    bounding_boxes: Optional[dict[str, Any]] = None
    context_data: Optional[dict[str, Any]] = None
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    dismissed_at: Optional[datetime] = None
    confirm_status: Optional[bool] = None
    status: Optional[str] = None
    pending_until: Optional[datetime] = None
    proposed_status: Optional[str] = None
    proposed_event_type: Optional[str] = None
    proposed_reason: Optional[str] = None
    proposed_by: Optional[str] = None

