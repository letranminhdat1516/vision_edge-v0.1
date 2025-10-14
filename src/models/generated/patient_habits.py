"""
PatientHabits Model
Generated from table: patient_habits
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class PatientHabits(Base):
    __tablename__ = 'patient_habits'
    
    habit_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    habit_type = Column(String(10), nullable=False)
    habit_name = Column(String(200), nullable=False)
    description = Column(Text)
    frequency = Column(String(6), nullable=False)
    days_of_week = Column(String)
    location = Column(String(100))
    notes = Column(Text)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    user_id = Column(UUID(as_uuid=True), nullable=False)
    sleep_start = Column(String)
    sleep_end = Column(String)

    def __repr__(self):
        return f"<PatientHabits(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
