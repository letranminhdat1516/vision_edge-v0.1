"""
Transactions Model
Generated from table: transactions
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Transactions(Base):
    __tablename__ = 'transactions'
    
    tx_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    plan_code = Column(Text, nullable=False)
    plan_snapshot = Column(String, nullable=False)
    amount_subtotal = Column(String, nullable=False)
    amount_discount = Column(String, nullable=False)
    amount_tax = Column(String, nullable=False)
    amount_total = Column(String, nullable=False)
    currency = Column(String(3), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    status = Column(String(7), nullable=False)
    due_date = Column(DateTime)
    paid_at = Column(DateTime)
    payment_id = Column(UUID(as_uuid=True))
    effective_action = Column(String(10), nullable=False)
    provider = Column(String(6), nullable=False)
    provider_payment_id = Column(String(100))
    idempotency_key = Column(String(100))
    related_tx_id = Column(UUID(as_uuid=True))
    notes = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_proration = Column(Boolean, nullable=False)
    plan_snapshot_new = Column(String)
    plan_snapshot_old = Column(String)
    proration_charge = Column(String, nullable=False)
    proration_credit = Column(String, nullable=False)
    version = Column(String(20))

    def __repr__(self):
        return f"<Transactions(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
