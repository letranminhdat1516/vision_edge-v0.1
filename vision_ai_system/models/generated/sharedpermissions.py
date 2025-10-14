from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class SharedPermissions(Base):
    __tablename__ = 'shared_permissions'
    __table_args__ = (
        ForeignKeyConstraint(['caregiver_id'], ['users.user_id'], name='shared_permissions_caregiver_id_fkey'),
        ForeignKeyConstraint(['customer_id'], ['users.user_id'], name='shared_permissions_customer_id_fkey'),
        PrimaryKeyConstraint('id', name='shared_permissions_pkey'),
        Index('idx_shared_permissions_caregiver', 'caregiver_id'),
        Index('idx_unique_shared_permission_pair', 'customer_id', 'caregiver_id', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    customer_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    caregiver_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    stream_view: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    alert_read: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    alert_ack: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    profile_view: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    log_access_days: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('0'))
    report_access_days: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('0'))
    notification_channel: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    permission_requests: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    permission_scopes: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))

    caregiver: Mapped['Users'] = relationship('Users', foreign_keys=[caregiver_id], back_populates='shared_permissions')
    customer: Mapped['Users'] = relationship('Users', foreign_keys=[customer_id], back_populates='shared_permissions_')
