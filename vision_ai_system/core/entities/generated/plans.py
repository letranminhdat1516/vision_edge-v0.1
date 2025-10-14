# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Plans:
    id: Optional[str] = None
    code: str
    name: str
    price: int
    currency: str
    billing_period: str
    is_active: bool
    is_current: bool
    camera_quota: int
    retention_days: int
    caregiver_seats: int
    sites: int
    major_updates_months: int
    status: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    storage_size: Optional[str] = None
    version: Optional[str] = None
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    is_recommended: Optional[bool] = None
    successor_plan_code: Optional[str] = None
    successor_plan_version: Optional[str] = None
    tier: Optional[int] = None

