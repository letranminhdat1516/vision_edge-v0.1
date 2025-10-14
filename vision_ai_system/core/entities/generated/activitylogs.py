# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class ActivityLogs:
    id: Optional[str] = None
    timestamp: datetime
    action: str
    severity: str
    actor_id: Optional[str] = None
    actor_name: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    message: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    ip: Optional[str] = None
    action_enum: Optional[str] = None
    resource_name: Optional[str] = None

