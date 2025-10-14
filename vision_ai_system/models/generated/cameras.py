from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Cameras(Base):
    __tablename__ = 'cameras'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='cameras_user_id_fkey'),
        PrimaryKeyConstraint('camera_id', name='cameras_pkey'),
        Index('cameras_ip_address_key', 'ip_address', unique=True),
        Index('idx_cameras_last_ping', 'last_ping'),
        Index('idx_cameras_status', 'status'),
        Index('idx_cameras_type', 'camera_type')
    )

    camera_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    camera_name: Mapped[str] = mapped_column(String(100), nullable=False)
    camera_type: Mapped[str] = mapped_column(Enum('ip', 'usb', 'rtsp', name='camera_type_enum'), nullable=False, server_default=text("'ip'::camera_type_enum"))
    status: Mapped[str] = mapped_column(Enum('active', 'inactive', 'error', name='camera_status_enum'), nullable=False, server_default=text("'active'::camera_status_enum"))
    is_online: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    port: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('80'))
    rtsp_url: Mapped[Optional[str]] = mapped_column(String(255))
    username: Mapped[Optional[str]] = mapped_column(String(50))
    password: Mapped[Optional[str]] = mapped_column(String(100))
    location_in_room: Mapped[Optional[str]] = mapped_column(String(50))
    resolution: Mapped[Optional[str]] = mapped_column(String(20), server_default=text("'1920x1080'::character varying"))
    fps: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('30'))
    last_ping: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    last_heartbeat_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    user: Mapped['Users'] = relationship('Users', back_populates='cameras')
    snapshots: Mapped[list['Snapshots']] = relationship('Snapshots', back_populates='camera')
    event_detections: Mapped[list['EventDetections']] = relationship('EventDetections', back_populates='camera')
