# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Uploads:
    upload_id: Optional[str] = None
    user_id: str
    filename: str
    mime: str
    size: int
    url: str
    upload_type: str
    created_at: datetime
    metadata: Optional[dict[str, Any]] = None

