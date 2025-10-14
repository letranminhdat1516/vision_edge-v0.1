# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Roles:
    id: Optional[str] = None
    name: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None

