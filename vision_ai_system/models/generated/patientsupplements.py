from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class PatientSupplements(Base):
    __tablename__ = 'patient_supplements'
    __table_args__ = (
        ForeignKeyConstraint(['customer_id'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='patient_supplements_customer_id_fkey'),
        PrimaryKeyConstraint('id', name='patient_supplements_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    name: Mapped[Optional[str]] = mapped_column(Text)
    dob: Mapped[Optional[datetime.date]] = mapped_column(Date)
    customer_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    call_confirmed_until: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    height_cm: Mapped[Optional[int]] = mapped_column(Integer)
    weight_kg: Mapped[Optional[float]] = mapped_column(Double(53))

    customer: Mapped[Optional['Users']] = relationship('Users', back_populates='patient_supplements')
    patient_habits: Mapped[list['PatientHabits']] = relationship('PatientHabits', back_populates='supplement')
    patient_medical_records: Mapped[list['PatientMedicalRecords']] = relationship('PatientMedicalRecords', back_populates='supplement')
