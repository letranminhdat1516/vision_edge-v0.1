# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class SnapshotImages:
    image_id: Optional[str] = None
    snapshot_id: str
    created_at: datetime
    image_path: Optional[str] = None
    cloud_url: Optional[str] = None
    file_size: Optional[int] = None

