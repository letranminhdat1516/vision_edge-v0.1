"""
Ticket Model
Generated from table: ticket
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Ticket(Base):
    __tablename__ = 'ticket'
    
    ticket_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    type = Column(String(7), nullable=False)
    title = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status = Column(String(12), nullable=False)
    category = Column(String(50))
    priority = Column(String(20))
    assigned_to = Column(UUID(as_uuid=True))
    tags = Column(String)
    metadata = Column(String)
    due_date = Column(DateTime)
    resolved_at = Column(DateTime)
    closed_at = Column(DateTime)

    def __repr__(self):
        return f"<Ticket(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
