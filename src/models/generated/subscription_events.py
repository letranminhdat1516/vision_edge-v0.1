"""
SubscriptionEvents Model
Generated from table: subscription_events
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class SubscriptionEvents(Base):
    __tablename__ = 'subscription_events'
    
    id = Column(String, primary_key=True, nullable=False)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String(10), nullable=False)
    event_data = Column(String)
    old_plan_code = Column(String(50))
    new_plan_code = Column(String(50))
    old_status = Column(String(9))
    new_status = Column(String(9))
    triggered_by = Column(UUID(as_uuid=True))
    reason = Column(String(255))
    tx_id = Column(UUID(as_uuid=True))
    payment_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<SubscriptionEvents(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
