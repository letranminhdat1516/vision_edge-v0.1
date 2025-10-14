from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class SubscriptionEvents(Base):
    __tablename__ = 'subscription_events'
    __table_args__ = (
        ForeignKeyConstraint(['payment_id'], ['payments.payment_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_payment_id_fkey'),
        ForeignKeyConstraint(['subscription_id'], ['subscriptions.subscription_id'], ondelete='CASCADE', onupdate='CASCADE', name='subscription_events_subscription_id_fkey'),
        ForeignKeyConstraint(['triggered_by'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_triggered_by_fkey'),
        ForeignKeyConstraint(['tx_id'], ['transactions.tx_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_tx_id_fkey'),
        PrimaryKeyConstraint('id', name='subscription_events_pkey'),
        Index('idx_sub_events_composite', 'subscription_id', 'event_type', 'created_at'),
        Index('idx_sub_events_new_plan', 'new_plan_code'),
        Index('idx_sub_events_old_plan', 'old_plan_code'),
        Index('idx_sub_events_payment', 'payment_id'),
        Index('idx_sub_events_sub', 'subscription_id'),
        Index('idx_sub_events_triggered_by', 'triggered_by'),
        Index('idx_sub_events_tx', 'tx_id'),
        Index('idx_sub_events_type', 'event_type')
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    event_type: Mapped[str] = mapped_column(Enum('created', 'activated', 'renewed', 'upgraded', 'downgraded', 'paused', 'resumed', 'canceled', 'expired', name='subscription_event_type_enum'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    event_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    old_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    new_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    old_status: Mapped[Optional[str]] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'))
    new_status: Mapped[Optional[str]] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'))
    triggered_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    reason: Mapped[Optional[str]] = mapped_column(String(255))
    tx_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    payment_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)

    payment: Mapped[Optional['Payments']] = relationship('Payments', back_populates='subscription_events')
    subscription: Mapped['Subscriptions'] = relationship('Subscriptions', back_populates='subscription_events')
    users: Mapped[Optional['Users']] = relationship('Users', back_populates='subscription_events')
    tx: Mapped[Optional['Transactions']] = relationship('Transactions', back_populates='subscription_events')
