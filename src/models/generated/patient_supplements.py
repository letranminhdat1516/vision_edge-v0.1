"""
PatientSupplements Model
Generated from table: patient_supplements
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class PatientSupplements(Base):
    __tablename__ = 'patient_supplements'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(Text)
    dob = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    customer_id = Column(UUID(as_uuid=True))
    call_confirmed_until = Column(DateTime)
    height_cm = Column(Integer)
    weight_kg = Column(String)

    def __repr__(self):
        return f"<PatientSupplements(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
