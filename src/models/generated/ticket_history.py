"""
TicketHistory Model
Generated from table: ticket_history
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class TicketHistory(Base):
    __tablename__ = 'ticket_history'
    
    history_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    ticket_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(30), nullable=False)
    old_values = Column(String)
    new_values = Column(String)
    description = Column(Text)
    metadata = Column(String)
    created_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<TicketHistory(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
