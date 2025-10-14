from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Notifications(Base):
    __tablename__ = 'notifications'
    __table_args__ = (
        ForeignKeyConstraint(['acknowledged_by'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='notifications_acknowledged_by_fkey'),
        ForeignKeyConstraint(['event_id'], ['event_detections.event_id'], ondelete='CASCADE', onupdate='CASCADE', name='notifications_event_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='notifications_user_id_fkey'),
        PrimaryKeyConstraint('notification_id', name='notifications_pkey'),
        Index('idx_notif_ack_by', 'acknowledged_by'),
        Index('idx_notif_created', 'created_at'),
        Index('idx_notif_event', 'event_id'),
        Index('idx_notif_read', 'read_at'),
        Index('idx_notif_retry', 'retry_count'),
        Index('idx_notif_sent', 'sent_at'),
        Index('idx_notif_severity', 'severity'),
        Index('idx_notif_status', 'status'),
        Index('idx_notif_type', 'notification_type'),
        Index('idx_notif_type_status', 'notification_type', 'status'),
        Index('idx_notif_user', 'user_id')
    )

    notification_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    notification_type: Mapped[str] = mapped_column(Enum('email', 'sms', 'push', 'in_app', 'webhook', name='notif_type_enum'), nullable=False)
    severity: Mapped[str] = mapped_column(Enum('critical', 'high', 'medium', 'low', name='severity_enum'), nullable=False, server_default=text("'medium'::severity_enum"))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Enum('pending', 'sent', 'delivered', 'failed', 'bounced', name='notif_status_enum'), nullable=False, server_default=text("'pending'::notif_status_enum"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    delivery_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    sent_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    delivered_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    retry_count: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('0'))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    read_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    acknowledged_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    acknowledged_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    resolved_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    users: Mapped[Optional['Users']] = relationship('Users', foreign_keys=[acknowledged_by], back_populates='notifications')
    event: Mapped[Optional['EventDetections']] = relationship('EventDetections', back_populates='notifications')
    user: Mapped['Users'] = relationship('Users', foreign_keys=[user_id], back_populates='notifications_')
