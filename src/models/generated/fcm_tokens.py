"""
FcmTokens Model
Generated from table: fcm_tokens
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class FcmTokens(Base):
    __tablename__ = 'fcm_tokens'
    
    token_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    device_id = Column(String(100))
    token = Column(Text, nullable=False)
    platform = Column(String(7), nullable=False)
    app_version = Column(String(50))
    device_model = Column(String(100))
    os_version = Column(String(50))
    topics = Column(String)
    is_active = Column(Boolean, nullable=False)
    last_used_at = Column(DateTime)
    revoked_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<FcmTokens(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
