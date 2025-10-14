"""
RolePermissions Model
Generated from table: role_permissions
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class RolePermissions(Base):
    __tablename__ = 'role_permissions'
    
    role_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    permission_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    assigned_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<RolePermissions(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
