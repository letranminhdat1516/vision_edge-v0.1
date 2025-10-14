from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class PatientHabits(Base):
    __tablename__ = 'patient_habits'
    __table_args__ = (
        ForeignKeyConstraint(['supplement_id'], ['patient_supplements.id'], ondelete='CASCADE', onupdate='CASCADE', name='patient_habits_supplement_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='patient_habits_user_id_fkey'),
        PrimaryKeyConstraint('habit_id', name='patient_habits_pkey'),
        Index('idx_ph_frequency', 'frequency'),
        Index('idx_ph_habit_type', 'habit_type'),
        Index('idx_ph_is_active', 'is_active'),
        Index('idx_ph_supplement', 'supplement_id'),
        Index('idx_ph_user_id', 'user_id')
    )

    habit_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    habit_type: Mapped[str] = mapped_column(Enum('sleep', 'meal', 'medication', 'activity', 'bathroom', 'therapy', 'social', name='habit_type_enum'), nullable=False)
    habit_name: Mapped[str] = mapped_column(String(200), nullable=False)
    frequency: Mapped[str] = mapped_column(Enum('daily', 'weekly', 'custom', name='frequency_enum'), nullable=False, server_default=text("'daily'::frequency_enum"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    days_of_week: Mapped[Optional[dict]] = mapped_column(JSONB)
    location: Mapped[Optional[str]] = mapped_column(String(100))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    supplement_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    sleep_start: Mapped[Optional[datetime.time]] = mapped_column(TIME(precision=6))
    sleep_end: Mapped[Optional[datetime.time]] = mapped_column(TIME(precision=6))

    supplement: Mapped[Optional['PatientSupplements']] = relationship('PatientSupplements', back_populates='patient_habits')
    user: Mapped['Users'] = relationship('Users', back_populates='patient_habits')
