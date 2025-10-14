# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class PatientSupplements:
    id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    name: Optional[str] = None
    dob: Optional[datetime] = None
    customer_id: Optional[str] = None
    call_confirmed_until: Optional[datetime] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[str] = None

