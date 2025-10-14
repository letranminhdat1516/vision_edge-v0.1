from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Snapshots(Base):
    __tablename__ = 'snapshots'
    __table_args__ = (
        ForeignKeyConstraint(['camera_id'], ['cameras.camera_id'], ondelete='CASCADE', onupdate='CASCADE', name='snapshots_camera_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='snapshots_user_id_fkey'),
        PrimaryKeyConstraint('snapshot_id', name='snapshots_pkey'),
        Index('idx_sn_camera', 'camera_id'),
        Index('idx_sn_captured', 'captured_at'),
        Index('idx_sn_processed', 'is_processed'),
        Index('idx_sn_type', 'capture_type'),
        Index('idx_sn_user', 'user_id'),
        Index('idx_sn_user_captured', 'user_id', 'captured_at'),
        Index('idx_snaps_camera_date', 'camera_id', 'captured_at')
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    camera_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    capture_type: Mapped[str] = mapped_column(Enum('scheduled', 'motion_triggered', 'manual', 'alert_triggered', name='capture_type_enum'), nullable=False, server_default=text("'scheduled'::capture_type_enum"))
    captured_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    is_processed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    metadata_: Mapped[Optional[dict]] = mapped_column('metadata', JSONB)
    processed_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    camera: Mapped['Cameras'] = relationship('Cameras', back_populates='snapshots')
    user: Mapped[Optional['Users']] = relationship('Users', back_populates='snapshots')
    event_detections: Mapped[list['EventDetections']] = relationship('EventDetections', back_populates='snapshot')
    snapshot_images: Mapped[list['SnapshotImages']] = relationship('SnapshotImages', back_populates='snapshot')
