# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Ticket:
    ticket_id: Optional[str] = None
    user_id: str
    type: str
    created_at: datetime
    updated_at: datetime
    status: str
    title: Optional[str] = None
    description: Optional[str] = None

