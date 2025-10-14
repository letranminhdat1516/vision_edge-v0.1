from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class PatientMedicalRecords(Base):
    __tablename__ = 'patient_medical_records'
    __table_args__ = (
        ForeignKeyConstraint(['supplement_id'], ['patient_supplements.id'], ondelete='CASCADE', onupdate='CASCADE', name='patient_medical_records_supplement_id_fkey'),
        PrimaryKeyConstraint('id', name='patient_medical_records_pkey'),
        Index('idx_pmr_supplement', 'supplement_id')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    history: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    supplement_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    name: Mapped[Optional[str]] = mapped_column(String(200))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    supplement: Mapped[Optional['PatientSupplements']] = relationship('PatientSupplements', back_populates='patient_medical_records')
