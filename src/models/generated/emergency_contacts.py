"""
EmergencyContacts Model
Generated from table: emergency_contacts
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class EmergencyContacts(Base):
    __tablename__ = 'emergency_contacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(100), nullable=False)
    relation = Column(String(50), nullable=False)
    phone = Column(String(20), nullable=False)
    alert_level = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_deleted = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"<EmergencyContacts(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
