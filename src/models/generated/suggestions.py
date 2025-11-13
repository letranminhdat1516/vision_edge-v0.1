"""
Suggestions Model
Generated from table: suggestions
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Suggestions(Base):
    __tablename__ = 'suggestions'
    
    suggestion_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True))
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    type = Column(String(50))
    message = Column(Text)
    skip_until = Column(DateTime)
    skip_scope = Column(String(4))
    skip_type = Column(String(50))
    skip_reason = Column(Text)
    last_notified_at = Column(DateTime)
    next_notify_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)
    meta = Column(String)
    status = Column(String(8), nullable=False)
    title = Column(String(255))

    def __repr__(self):
        return f"<Suggestions(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
