# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Notifications:
    notification_id: Optional[str] = None
    user_id: str
    notification_type: str
    severity: str
    message: str
    status: str
    created_at: datetime
    event_id: Optional[str] = None
    title: Optional[str] = None
    delivery_data: Optional[dict[str, Any]] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    retry_count: Optional[int] = None
    error_message: Optional[str] = None
    read_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

