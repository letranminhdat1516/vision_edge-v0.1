"""
SystemConfig Model
Generated from table: system_config
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class SystemConfig(Base):
    __tablename__ = 'system_config'
    
    setting_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text, nullable=False)
    description = Column(Text)
    data_type = Column(String(7), nullable=False)
    category = Column(String(50))
    is_encrypted = Column(Boolean)
    updated_at = Column(DateTime, nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=False)

    def __repr__(self):
        return f"<SystemConfig(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
