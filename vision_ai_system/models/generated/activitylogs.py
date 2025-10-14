from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class ActivityLogs(Base):
    __tablename__ = 'activity_logs'
    __table_args__ = (
        ForeignKeyConstraint(['actor_id'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='fk_activity_logs_actor'),
        PrimaryKeyConstraint('id', name='activity_logs_pkey'),
        Index('idx_al_action', 'action'),
        Index('idx_al_actor', 'actor_id'),
        Index('idx_al_resource_type', 'resource_type'),
        Index('idx_al_severity', 'severity'),
        Index('idx_al_timestamp', 'timestamp')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(Enum('critical', 'high', 'medium', 'low', 'info', name='activity_severity_enum'), nullable=False, server_default=text("'info'::activity_severity_enum"))
    actor_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    actor_name: Mapped[Optional[str]] = mapped_column(String(255))
    resource_type: Mapped[Optional[str]] = mapped_column(String(100))
    resource_id: Mapped[Optional[str]] = mapped_column(String(100))
    message: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB)
    ip: Mapped[Optional[str]] = mapped_column(String(50))
    action_enum: Mapped[Optional[str]] = mapped_column(Enum('create', 'update', 'delete', 'login', 'logout', 'acknowledge', 'assign', 'unassign', name='activity_action_enum'))
    resource_name: Mapped[Optional[str]] = mapped_column(String)

    actor: Mapped[Optional['Users']] = relationship('Users', back_populates='activity_logs')
