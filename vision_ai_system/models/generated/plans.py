from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Plans(Base):
    __tablename__ = 'plans'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='plans_pkey'),
        Index('plans_code_is_current_idx', 'code', 'is_current'),
        Index('plans_code_key', 'code', unique=True),
        Index('plans_is_active_idx', 'is_active')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    code: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    price: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, server_default=text("'VND'::character varying"))
    billing_period: Mapped[str] = mapped_column(Enum('monthly', 'yearly', 'none', name='billing_period_enum'), nullable=False, server_default=text("'monthly'::billing_period_enum"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    camera_quota: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('30'))
    caregiver_seats: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    sites: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('1'))
    major_updates_months: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('12'))
    status: Mapped[str] = mapped_column(Enum('draft', 'available', 'deprecated', 'archived', name='plan_status_enum'), nullable=False, server_default=text("'available'::plan_status_enum"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(Text)
    storage_size: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[Optional[str]] = mapped_column(String(20))
    effective_from: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(precision=3))
    effective_to: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(precision=3))
    is_recommended: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('false'))
    successor_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    successor_plan_version: Mapped[Optional[str]] = mapped_column(String(20))
    tier: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('1'))

    subscriptions: Mapped[list['Subscriptions']] = relationship('Subscriptions', back_populates='plan')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='plan')
