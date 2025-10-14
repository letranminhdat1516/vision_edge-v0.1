"""
Subscriptions Model
Generated from table: subscriptions
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    plan_code = Column(Text, nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    status = Column(String(9), nullable=False)
    billing_period = Column(String(7), nullable=False)
    started_at = Column(DateTime, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime)
    trial_end_at = Column(DateTime)
    canceled_at = Column(DateTime)
    ended_at = Column(DateTime)
    auto_renew = Column(Boolean, nullable=False)
    extra_camera_quota = Column(Integer, nullable=False)
    extra_caregiver_seats = Column(Integer, nullable=False)
    extra_sites = Column(Integer, nullable=False)
    extra_storage_gb = Column(Integer, nullable=False)
    notes = Column(Text)
    last_payment_at = Column(DateTime)
    version = Column(String(20))
    cancel_at_period_end = Column(Boolean, nullable=False)
    offer_start_date = Column(DateTime)
    offer_end_date = Column(DateTime)

    def __repr__(self):
        return f"<Subscriptions(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
