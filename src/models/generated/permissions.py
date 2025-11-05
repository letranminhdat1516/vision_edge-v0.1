"""
Permissions Model
Generated from table: permissions
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Permissions(Base):
    __tablename__ = 'permissions'
    
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    permission_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)

    def __repr__(self):
        return f"<Permissions(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
