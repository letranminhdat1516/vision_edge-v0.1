"""
CaregiverInvitations Model
Generated from table: caregiver_invitations
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class CaregiverInvitations(Base):
    __tablename__ = 'caregiver_invitations'
    
    assignment_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    assigned_at = Column(DateTime, nullable=False)
    unassigned_at = Column(DateTime)
    is_active = Column(Boolean, nullable=False)
    assigned_by = Column(UUID(as_uuid=True))
    assignment_notes = Column(Text)
    responded_at = Column(DateTime)
    expires_at = Column(DateTime)
    response_reason = Column(String(255))
    status = Column(String(9), nullable=False)

    def __repr__(self):
        return f"<CaregiverInvitations(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
