from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class EventDetections(Base):
    __tablename__ = 'event_detections'
    __table_args__ = (
        ForeignKeyConstraint(['camera_id'], ['cameras.camera_id'], ondelete='CASCADE', onupdate='CASCADE', name='event_detections_camera_id_fkey'),
        ForeignKeyConstraint(['snapshot_id'], ['snapshots.snapshot_id'], ondelete='CASCADE', onupdate='CASCADE', name='event_detections_snapshot_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='event_detections_user_id_fkey'),
        ForeignKeyConstraint(['verified_by'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='event_detections_verified_by_fkey'),
        PrimaryKeyConstraint('event_id', name='event_detections_pkey'),
        Index('idx_ed_ack_at', 'acknowledged_at'),
        Index('idx_ed_camera', 'camera_id'),
        Index('idx_ed_conf', 'confidence_score'),
        Index('idx_ed_confstate_pending', 'confirmation_state', 'pending_until'),
        Index('idx_ed_detected', 'detected_at'),
        Index('idx_ed_dismissed_at', 'dismissed_at'),
        Index('idx_ed_snapshot', 'snapshot_id'),
        Index('idx_ed_type', 'event_type'),
        Index('idx_ed_user', 'user_id'),
        Index('idx_ed_verified_by', 'verified_by'),
        Index('idx_events_user_date', 'user_id', 'detected_at')
    )

    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    snapshot_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    camera_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    event_type: Mapped[str] = mapped_column(Enum('fall', 'abnormal_behavior', 'emergency', 'normal_activity', 'sleep', name='event_type_enum'), nullable=False)
    detected_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    confirmation_state: Mapped[str] = mapped_column(Enum('DETECTED', 'CAREGIVER_UPDATED', 'CONFIRMED_BY_CUSTOMER', 'REJECTED_BY_CUSTOMER', 'AUTO_APPROVED', name='confirmation_state_enum'), nullable=False, server_default=text("'DETECTED'::confirmation_state_enum"))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    event_description: Mapped[Optional[str]] = mapped_column(Text)
    detection_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    ai_analysis_result: Mapped[Optional[dict]] = mapped_column(JSONB)
    confidence_score: Mapped[Optional[decimal.Decimal]] = mapped_column(Numeric(5, 2), server_default=text('0.00'))
    bounding_boxes: Mapped[Optional[dict]] = mapped_column(JSONB)
    context_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    verified_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    verified_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    acknowledged_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    acknowledged_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    dismissed_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    confirm_status: Mapped[Optional[bool]] = mapped_column(Boolean)
    status: Mapped[Optional[str]] = mapped_column(Enum('danger', 'warning', 'normal', name='event_status_enum'))
    pending_until: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    proposed_status: Mapped[Optional[str]] = mapped_column(Text)
    proposed_event_type: Mapped[Optional[str]] = mapped_column(Text)
    proposed_reason: Mapped[Optional[str]] = mapped_column(Text)
    proposed_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)

    camera: Mapped['Cameras'] = relationship('Cameras', back_populates='event_detections')
    snapshot: Mapped['Snapshots'] = relationship('Snapshots', back_populates='event_detections')
    user: Mapped['Users'] = relationship('Users', foreign_keys=[user_id], back_populates='event_detections')
    users: Mapped[Optional['Users']] = relationship('Users', foreign_keys=[verified_by], back_populates='event_detections_')
    notifications: Mapped[list['Notifications']] = relationship('Notifications', back_populates='event')
