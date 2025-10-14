from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Transactions(Base):
    __tablename__ = 'transactions'
    __table_args__ = (
        ForeignKeyConstraint(['payment_id'], ['payments.payment_id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_payment_id_fkey'),
        ForeignKeyConstraint(['plan_id'], ['plans.id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_plan_id_fkey'),
        ForeignKeyConstraint(['related_tx_id'], ['transactions.tx_id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_related_tx_id_fkey'),
        ForeignKeyConstraint(['subscription_id'], ['subscriptions.subscription_id'], ondelete='CASCADE', onupdate='CASCADE', name='transactions_subscription_id_fkey'),
        PrimaryKeyConstraint('tx_id', name='transactions_pkey'),
        Index('idx_tx_period_end', 'period_end'),
        Index('idx_tx_period_start', 'period_start'),
        Index('idx_tx_plan_code', 'plan_code'),
        Index('idx_tx_plan_id', 'plan_id'),
        Index('idx_tx_provider_payment_id', 'provider_payment_id'),
        Index('idx_tx_status', 'status'),
        Index('idx_tx_sub', 'subscription_id'),
        Index('uq_tx_idem_per_sub', 'subscription_id', 'idempotency_key', unique=True),
        Index('uq_tx_payment_id', 'payment_id', unique=True)
    )

    tx_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    plan_code: Mapped[str] = mapped_column(Text, nullable=False)
    plan_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    amount_subtotal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount_discount: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    amount_tax: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    amount_total: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default=text("'VND'::character varying"))
    period_start: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    period_end: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    status: Mapped[str] = mapped_column(Enum('draft', 'open', 'paid', 'void', 'overdue', name='invoice_status_enum'), nullable=False, server_default=text("'draft'::invoice_status_enum"))
    effective_action: Mapped[str] = mapped_column(Enum('new', 'renew', 'upgrade', 'downgrade', 'adjustment', name='TransactionAction'), nullable=False)
    provider: Mapped[str] = mapped_column(Enum('vn_pay', 'stripe', 'manual', name='PaymentProvider'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    is_proration: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    proration_charge: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    proration_credit: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    plan_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    due_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    paid_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    payment_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    provider_payment_id: Mapped[Optional[str]] = mapped_column(String(100))
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(100))
    related_tx_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    plan_snapshot_new: Mapped[Optional[dict]] = mapped_column(JSONB)
    plan_snapshot_old: Mapped[Optional[dict]] = mapped_column(JSONB)
    version: Mapped[Optional[str]] = mapped_column(String(20))

    payment: Mapped[Optional['Payments']] = relationship('Payments', back_populates='transactions')
    plan: Mapped[Optional['Plans']] = relationship('Plans', back_populates='transactions')
    related_tx: Mapped[Optional['Transactions']] = relationship('Transactions', remote_side=[tx_id], back_populates='related_tx_reverse')
    related_tx_reverse: Mapped[list['Transactions']] = relationship('Transactions', remote_side=[related_tx_id], back_populates='related_tx')
    subscription: Mapped['Subscriptions'] = relationship('Subscriptions', back_populates='transactions')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='tx')
