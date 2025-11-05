"""
PrismaMigrations Model
Generated from table: _prisma_migrations
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class PrismaMigrations(Base):
    __tablename__ = '_prisma_migrations'
    
    id = Column(String(36), primary_key=True, nullable=False)
    checksum = Column(String(64), nullable=False)
    finished_at = Column(DateTime)
    migration_name = Column(String(255), nullable=False)
    logs = Column(Text)
    rolled_back_at = Column(DateTime)
    started_at = Column(DateTime, nullable=False)
    applied_steps_count = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<PrismaMigrations(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
