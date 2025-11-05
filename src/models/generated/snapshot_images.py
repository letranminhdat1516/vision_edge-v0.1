"""
SnapshotImages Model
Generated from table: snapshot_images
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'
    
    image_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    is_primary = Column(Boolean, nullable=False)
    image_path = Column(Text)
    cloud_url = Column(Text)
    created_at = Column(DateTime, nullable=False)
    file_size = Column(String)

    def __repr__(self):
        return f"<SnapshotImages(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
