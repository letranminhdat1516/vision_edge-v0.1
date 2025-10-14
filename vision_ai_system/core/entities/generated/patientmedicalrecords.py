# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class PatientMedicalRecords:
    id: Optional[str] = None
    history: dict[str, Any]
    updated_at: datetime
    supplement_id: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None

