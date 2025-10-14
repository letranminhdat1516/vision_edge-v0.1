# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Subscriptions:
    subscription_id: Optional[str] = None
    user_id: str
    plan_code: str
    status: str
    billing_period: str
    started_at: datetime
    current_period_start: datetime
    auto_renew: bool
    extra_camera_quota: int
    extra_caregiver_seats: int
    extra_sites: int
    extra_storage_gb: int
    cancel_at_period_end: bool
    plan_id: Optional[str] = None
    current_period_end: Optional[datetime] = None
    trial_end_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    notes: Optional[str] = None
    last_payment_at: Optional[datetime] = None
    version: Optional[str] = None
    offer_start_date: Optional[datetime] = None
    offer_end_date: Optional[datetime] = None

