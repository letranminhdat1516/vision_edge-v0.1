# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class FcmTokens:
    token_id: Optional[str] = None
    user_id: str
    token: str
    platform: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    device_id: Optional[str] = None
    app_version: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    topics: Optional[dict[str, Any]] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

