"""
ActivityLogs Model
Generated from table: activity_logs
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class ActivityLogs(Base):
    __tablename__ = 'activity_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    actor_id = Column(UUID(as_uuid=True))
    actor_name = Column(String(255))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    message = Column(Text)
    severity = Column(String(8), nullable=False)
    meta = Column(String)
    ip = Column(String(50))
    action_enum = Column(String(11))
    resource_name = Column(String)

    def __repr__(self):
        return f"<ActivityLogs(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
