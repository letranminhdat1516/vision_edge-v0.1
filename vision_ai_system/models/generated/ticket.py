from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Ticket(Base):
    __tablename__ = 'ticket'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='ticket_user_id_fkey'),
        PrimaryKeyConstraint('ticket_id', name='ticket_pkey'),
        Index('idx_customer_requests_created_at', 'created_at'),
        Index('idx_customer_requests_status', 'status'),
        Index('idx_customer_requests_type', 'type'),
        Index('idx_customer_requests_user', 'user_id')
    )

    ticket_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    type: Mapped[str] = mapped_column(Enum('report', 'support', name='customer_request_type_enum'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    status: Mapped[str] = mapped_column(Enum('new', 'acknowledged', 'in_progress', 'resolved', 'rejected', name='customer_request_status_enum'), nullable=False, server_default=text("'new'::customer_request_status_enum"))
    title: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)

    user: Mapped['Users'] = relationship('Users', back_populates='ticket')
