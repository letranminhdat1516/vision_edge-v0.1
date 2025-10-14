from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'
    __table_args__ = (
        ForeignKeyConstraint(['snapshot_id'], ['snapshots.snapshot_id'], ondelete='CASCADE', name='fk_snapshot_images_snapshot'),
        PrimaryKeyConstraint('image_id', name='snapshot_images_pkey'),
        Index('idx_snapshot_images_created_at', 'created_at'),
        Index('idx_snapshot_images_snapshot_id', 'snapshot_id')
    )

    image_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    snapshot_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    image_path: Mapped[Optional[str]] = mapped_column(Text)
    cloud_url: Mapped[Optional[str]] = mapped_column(Text)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)

    snapshot: Mapped['Snapshots'] = relationship('Snapshots', back_populates='snapshot_images')
