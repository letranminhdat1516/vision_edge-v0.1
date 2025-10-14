from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    __table_args__ = (
        ForeignKeyConstraint(['plan_id'], ['plans.id'], ondelete='SET NULL', onupdate='CASCADE', name='subscriptions_plan_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='subscriptions_user_id_fkey'),
        PrimaryKeyConstraint('subscription_id', name='subscriptions_pkey'),
        Index('idx_sub_current_end', 'current_period_end'),
        Index('idx_sub_plan', 'plan_code'),
        Index('idx_sub_plan_id', 'plan_id'),
        Index('idx_sub_status', 'status'),
        Index('idx_sub_user', 'user_id'),
        Index('idx_sub_user_status', 'user_id', 'status')
    )

    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    plan_code: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'), nullable=False, server_default=text("'active'::subscription_status_enum"))
    billing_period: Mapped[str] = mapped_column(Enum('monthly', 'yearly', 'none', name='billing_period_enum'), nullable=False, server_default=text("'none'::billing_period_enum"))
    started_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    current_period_start: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    extra_camera_quota: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_caregiver_seats: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_sites: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_storage_gb: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    plan_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    current_period_end: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    trial_end_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    canceled_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    ended_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    last_payment_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    version: Mapped[Optional[str]] = mapped_column(String(20))
    offer_start_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    offer_end_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    plan: Mapped[Optional['Plans']] = relationship('Plans', back_populates='subscriptions')
    user: Mapped['Users'] = relationship('Users', back_populates='subscriptions')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='subscription')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='subscription')
