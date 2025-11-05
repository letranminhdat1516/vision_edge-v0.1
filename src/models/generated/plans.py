"""
Plans Model
Generated from table: plans
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Plans(Base):
    __tablename__ = 'plans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    code = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(String, nullable=False)
    currency = Column(String(10), nullable=False)
    billing_period = Column(String(10), nullable=False)
    is_active = Column(Boolean, nullable=False)
    is_current = Column(Boolean, nullable=False)
    camera_quota = Column(Integer, nullable=False)
    storage_size = Column(Text)
    retention_days = Column(Integer, nullable=False)
    caregiver_seats = Column(Integer, nullable=False)
    sites = Column(Integer, nullable=False)
    major_updates_months = Column(Integer, nullable=False)
    version = Column(String(20))
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    is_recommended = Column(Boolean)
    successor_plan_code = Column(String(50))
    successor_plan_version = Column(String(20))
    tier = Column(Integer)
    status = Column(String(10), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<Plans(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
