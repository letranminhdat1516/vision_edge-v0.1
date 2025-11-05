"""
Notifications Model
Generated from table: notifications
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Notifications(Base):
    __tablename__ = 'notifications'
    
    notification_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    event_id = Column(UUID(as_uuid=True))
    severity = Column(String(8), nullable=False)
    title = Column(String(255))
    message = Column(Text, nullable=False)
    delivery_data = Column(String)
    status = Column(String(9), nullable=False)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    retry_count = Column(Integer)
    error_message = Column(Text)
    read_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)
    channel = Column(String(7), nullable=False)
    business_type = Column(String(20))

    def __repr__(self):
        return f"<Notifications(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
