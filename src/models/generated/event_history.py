"""
EventHistory Model
Generated from table: event_history
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class EventHistory(Base):
    __tablename__ = 'event_history'
    
    history_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    event_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(18), nullable=False)
    actor_id = Column(UUID(as_uuid=True))
    actor_name = Column(String(255))
    actor_role = Column(String(50))
    previous_status = Column(String(50))
    new_status = Column(String(50))
    previous_event_type = Column(String(17))
    new_event_type = Column(String(17))
    previous_confirmation_state = Column(String(21))
    new_confirmation_state = Column(String(21))
    reason = Column(Text)
    metadata = Column(String)
    created_at = Column(DateTime, nullable=False)
    response_time_minutes = Column(Integer)
    is_first_action = Column(Boolean)

    def __repr__(self):
        return f"<EventHistory(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
