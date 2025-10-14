from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class UserSettings(Base):
    __tablename__ = 'user_settings'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', name='user_settings_user_id_fkey'),
        PrimaryKeyConstraint('id', name='user_settings_pkey'),
        Index('uq_user_setting', 'user_id', 'category', 'setting_key', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)
    setting_value: Mapped[str] = mapped_column(Text, nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    overridden_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    user: Mapped['Users'] = relationship('Users', back_populates='user_settings')
