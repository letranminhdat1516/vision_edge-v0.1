# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class EmergencyContacts:
    id: Optional[str] = None
    user_id: str
    name: str
    relation: str
    phone: str
    alert_level: int
    created_at: datetime
    updated_at: datetime
    is_deleted: bool

