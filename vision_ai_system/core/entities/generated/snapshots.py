# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Snapshots:
    snapshot_id: Optional[str] = None
    camera_id: str
    capture_type: str
    captured_at: datetime
    is_processed: bool
    user_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    processed_at: Optional[datetime] = None

