from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class SystemSettings(Base):
    __tablename__ = 'system_settings'
    __table_args__ = (
        ForeignKeyConstraint(['updated_by'], ['users.user_id'], ondelete='RESTRICT', onupdate='CASCADE', name='system_settings_updated_by_fkey'),
        PrimaryKeyConstraint('setting_id', name='system_settings_pkey'),
        Index('idx_ss_category', 'category'),
        Index('idx_ss_dtype', 'data_type'),
        Index('idx_ss_key', 'setting_key'),
        Index('idx_ss_updated_by', 'updated_by'),
        Index('idx_system_settings_key', 'setting_key', unique=True)
    )

    setting_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)
    setting_value: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(Enum('string', 'int', 'float', 'boolean', 'json', name='data_type_enum'), nullable=False, server_default=text("'string'::data_type_enum"))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(50), server_default=text("'general'::character varying"))
    is_encrypted: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('false'))

    users: Mapped['Users'] = relationship('Users', back_populates='system_settings')
