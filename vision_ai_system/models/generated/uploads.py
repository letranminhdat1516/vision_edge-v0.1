from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Uploads(Base):
    __tablename__ = 'uploads'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='uploads_user_id_fkey'),
        PrimaryKeyConstraint('upload_id', name='uploads_pkey'),
        Index('idx_uploads_created', 'created_at'),
        Index('idx_uploads_user', 'user_id')
    )

    upload_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime: Mapped[str] = mapped_column(String(100), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    upload_type: Mapped[str] = mapped_column(Enum('camera_error', 'other', name='upload_type_enum'), nullable=False, server_default=text("'other'::upload_type_enum"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    metadata_: Mapped[Optional[dict]] = mapped_column('metadata', JSONB)

    user: Mapped['Users'] = relationship('Users', back_populates='uploads')
