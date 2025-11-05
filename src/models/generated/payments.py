"""
Payments Model
Generated from table: payments
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Payments(Base):
    __tablename__ = 'payments'
    
    payment_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    amount = Column(String, nullable=False)
    currency = Column(String(3), nullable=False)
    provider = Column(String(6), nullable=False)
    status = Column(String(20), nullable=False)
    description = Column(Text)
    delivery_data = Column(String)
    vnp_txn_ref = Column(String(50))
    idempotency_key = Column(String(128))
    vnp_create_date = Column(String)
    vnp_expire_date = Column(String)
    vnp_order_info = Column(Text)
    version = Column(String(20))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status_enum = Column(String(10))

    def __repr__(self):
        return f"<Payments(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
