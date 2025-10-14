from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Payments(Base):
    __tablename__ = 'payments'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='payments_user_id_fkey'),
        PrimaryKeyConstraint('payment_id', name='payments_pkey'),
        Index('idx_pay_status', 'status'),
        Index('idx_pay_user', 'user_id'),
        Index('payments_vnp_txn_ref_key', 'vnp_txn_ref', unique=True)
    )

    payment_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default=text("'VND'::character varying"))
    provider: Mapped[str] = mapped_column(Enum('vn_pay', 'stripe', 'manual', name='PaymentProvider'), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'pending'::character varying"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(Text)
    delivery_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    vnp_txn_ref: Mapped[Optional[str]] = mapped_column(String(50))
    vnp_create_date: Mapped[Optional[int]] = mapped_column(BigInteger)
    vnp_expire_date: Mapped[Optional[int]] = mapped_column(BigInteger)
    vnp_order_info: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[Optional[str]] = mapped_column(String(20))

    user: Mapped['Users'] = relationship('Users', back_populates='payments')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='payment')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='payment')
