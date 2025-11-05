"""
AccessGrants Model
Generated from table: access_grants
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class AccessGrants(Base):
    __tablename__ = 'access_grants'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    stream_view = Column(Boolean, nullable=False)
    alert_read = Column(Boolean, nullable=False)
    alert_ack = Column(Boolean, nullable=False)
    profile_view = Column(Boolean, nullable=False)
    log_access_days = Column(Integer)
    report_access_days = Column(Integer)
    notification_channel = Column(String)
    permission_requests = Column(String)
    permission_scopes = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<AccessGrants(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
