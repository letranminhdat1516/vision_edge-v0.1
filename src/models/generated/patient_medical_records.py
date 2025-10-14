"""
PatientMedicalRecords Model
Generated from table: patient_medical_records
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class PatientMedicalRecords(Base):
    __tablename__ = 'patient_medical_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    history = Column(String, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    name = Column(String(200))
    notes = Column(Text)

    def __repr__(self):
        return f"<PatientMedicalRecords(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
