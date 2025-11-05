"""
Snapshots Model
Generated from table: snapshots
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Snapshots(Base):
    __tablename__ = 'snapshots'
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    snapshot_metadata = Column('metadata', String)  # Mapped to 'metadata' column in DB
    capture_type = Column(String(16), nullable=False)
    captured_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime)
    is_processed = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"<Snapshots(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
