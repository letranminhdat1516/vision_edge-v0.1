from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class CaregiverInvitations(Base):
    __tablename__ = 'caregiver_invitations'
    __table_args__ = (
        ForeignKeyConstraint(['assigned_by'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='caregiver_assignments_assigned_by_fkey'),
        PrimaryKeyConstraint('assignment_id', name='caregiver_patient_assignments_pkey')
    )

    assignment_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    caregiver_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    customer_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    assigned_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    unassigned_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    assigned_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    assignment_notes: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[Optional[str]] = mapped_column(String(20), server_default=text("'pending'::character varying"))

    users: Mapped[Optional['Users']] = relationship('Users', back_populates='caregiver_invitations')
