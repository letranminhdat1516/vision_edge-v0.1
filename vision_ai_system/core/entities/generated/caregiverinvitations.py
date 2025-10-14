# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class CaregiverInvitations:
    assignment_id: Optional[str] = None
    caregiver_id: str
    customer_id: str
    assigned_at: datetime
    is_active: bool
    unassigned_at: Optional[datetime] = None
    assigned_by: Optional[str] = None
    assignment_notes: Optional[str] = None
    status: Optional[str] = None

