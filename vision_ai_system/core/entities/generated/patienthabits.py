# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class PatientHabits:
    habit_id: Optional[str] = None
    habit_type: str
    habit_name: str
    frequency: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    user_id: str
    description: Optional[str] = None
    days_of_week: Optional[dict[str, Any]] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    supplement_id: Optional[str] = None
    sleep_start: Optional[str] = None
    sleep_end: Optional[str] = None

