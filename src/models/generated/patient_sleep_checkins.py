"""
PatientSleepCheckins Model
Generated from table: patient_sleep_checkins
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class PatientSleepCheckins(Base):
    __tablename__ = 'patient_sleep_checkins'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    state = Column(String(100), nullable=False)
    meta = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    habit_id = Column(UUID(as_uuid=True))
    medical_history_id = Column(UUID(as_uuid=True))
    supplement_id = Column(UUID(as_uuid=True))
    checkin_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<PatientSleepCheckins(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
