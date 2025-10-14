"""
Uploads Model
Generated from table: uploads
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Uploads(Base):
    __tablename__ = 'uploads'
    
    upload_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    filename = Column(String(255), nullable=False)
    mime = Column(String(100), nullable=False)
    size = Column(Integer, nullable=False)
    url = Column(String(500), nullable=False)
    upload_type = Column(String(12), nullable=False)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(String)

    def __repr__(self):
        return f"<Uploads(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
