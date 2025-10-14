# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class SharedPermissions:
    id: Optional[str] = None
    customer_id: str
    caregiver_id: str
    stream_view: bool
    alert_read: bool
    alert_ack: bool
    profile_view: bool
    created_at: datetime
    updated_at: datetime
    log_access_days: Optional[int] = None
    report_access_days: Optional[int] = None
    notification_channel: Optional[dict[str, Any]] = None
    permission_requests: Optional[dict[str, Any]] = None
    permission_scopes: Optional[dict[str, Any]] = None

