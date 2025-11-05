"""
EventDetections Model
Generated from table: event_detections
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class EventDetections(Base):
    __tablename__ = 'event_detections'
    
    notes = Column(Text)
    event_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String(17), nullable=False)
    event_description = Column(Text)
    detection_data = Column(String)
    ai_analysis_result = Column(String)
    confidence_score = Column(String)
    bounding_boxes = Column(String)
    context_data = Column(String)
    detected_at = Column(DateTime, nullable=False)
    verified_at = Column(DateTime)
    verified_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    dismissed_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    confirm_status = Column(Boolean)
    status = Column(String(8))
    confirmation_state = Column(String(21), nullable=False)
    pending_until = Column(DateTime)
    proposed_reason = Column(Text)
    proposed_by = Column(UUID(as_uuid=True))
    proposed_status = Column(String(8))
    proposed_event_type = Column(String(17))
    auto_escalation_reason = Column(Text)
    escalated_at = Column(DateTime)
    escalation_count = Column(Integer, nullable=False)
    is_canceled = Column(Boolean, nullable=False)
    last_action_at = Column(DateTime)
    last_action_by = Column(UUID(as_uuid=True))
    pending_since = Column(DateTime)
    verification_status = Column(String(9), nullable=False)
    notification_attempts = Column(Integer, nullable=False)
    lifecycle_state = Column(String(15), nullable=False)
    reliability_score = Column(String)

    def __repr__(self):
        return f"<EventDetections(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
