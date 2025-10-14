# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Transactions:
    tx_id: Optional[str] = None
    subscription_id: str
    plan_code: str
    plan_snapshot: dict[str, Any]
    amount_subtotal: int
    amount_discount: int
    amount_tax: int
    amount_total: int
    currency: str
    period_start: datetime
    period_end: datetime
    status: str
    effective_action: str
    provider: str
    created_at: datetime
    updated_at: datetime
    is_proration: bool
    proration_charge: int
    proration_credit: int
    plan_id: Optional[str] = None
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    payment_id: Optional[str] = None
    provider_payment_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    related_tx_id: Optional[str] = None
    notes: Optional[str] = None
    plan_snapshot_new: Optional[dict[str, Any]] = None
    plan_snapshot_old: Optional[dict[str, Any]] = None
    version: Optional[str] = None

