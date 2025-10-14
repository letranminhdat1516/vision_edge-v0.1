# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class SubscriptionEvents:
    id: Optional[int] = None
    subscription_id: str
    event_type: str
    created_at: datetime
    event_data: Optional[dict[str, Any]] = None
    old_plan_code: Optional[str] = None
    new_plan_code: Optional[str] = None
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    triggered_by: Optional[str] = None
    reason: Optional[str] = None
    tx_id: Optional[str] = None
    payment_id: Optional[str] = None

