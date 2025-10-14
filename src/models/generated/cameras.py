"""
Cameras Model
Generated from table: cameras
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Cameras(Base):
    __tablename__ = 'cameras'
    
    camera_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_name = Column(String(100), nullable=False)
    camera_type = Column(String(4), nullable=False)
    ip_address = Column(String(45))
    port = Column(Integer)
    rtsp_url = Column(String(255))
    username = Column(String(50))
    password = Column(String(100))
    location_in_room = Column(String(50))
    resolution = Column(String(20))
    fps = Column(Integer)
    status = Column(String(8), nullable=False)
    last_ping = Column(DateTime)
    is_online = Column(Boolean, nullable=False)
    last_heartbeat_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<Cameras(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
