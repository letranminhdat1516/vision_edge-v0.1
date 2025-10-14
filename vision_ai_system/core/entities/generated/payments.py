# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Payments:
    payment_id: Optional[str] = None
    user_id: str
    amount: int
    currency: str
    provider: str
    status: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    delivery_data: Optional[dict[str, Any]] = None
    vnp_txn_ref: Optional[str] = None
    vnp_create_date: Optional[int] = None
    vnp_expire_date: Optional[int] = None
    vnp_order_info: Optional[str] = None
    version: Optional[str] = None

