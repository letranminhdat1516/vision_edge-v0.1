from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class FcmTokens(Base):
    __tablename__ = 'fcm_tokens'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='fcm_tokens_user_id_fkey'),
        PrimaryKeyConstraint('token_id', name='fcm_tokens_pkey'),
        Index('fcm_tokens_token_key', 'token', unique=True),
        Index('idx_fcm_active', 'is_active'),
        Index('idx_fcm_last_used', 'last_used_at'),
        Index('idx_fcm_platform', 'platform'),
        Index('idx_fcm_user', 'user_id'),
        Index('unique_fcm_user_device', 'user_id', 'device_id', unique=True)
    )

    token_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    token: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(Enum('ios', 'android', 'web', name='push_platform_enum'), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    device_id: Mapped[Optional[str]] = mapped_column(String(100))
    app_version: Mapped[Optional[str]] = mapped_column(String(50))
    device_model: Mapped[Optional[str]] = mapped_column(String(100))
    os_version: Mapped[Optional[str]] = mapped_column(String(50))
    topics: Mapped[Optional[dict]] = mapped_column(JSONB)
    last_used_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    revoked_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    user: Mapped['Users'] = relationship('Users', back_populates='fcm_tokens')
