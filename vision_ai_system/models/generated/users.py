from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Users(Base):
    __tablename__ = 'users'
    __table_args__ = (
        PrimaryKeyConstraint('user_id', name='users_pkey'),
        Index('idx_users_created_at', 'created_at'),
        Index('idx_users_email', 'email'),
        Index('idx_users_is_active', 'is_active'),
        Index('idx_users_phone_number', 'phone_number'),
        Index('idx_users_role', 'role'),
        Index('idx_users_username', 'username'),
        Index('users_email_key', 'email', unique=True),
        Index('users_username_key', 'username', unique=True)
    )

    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(Enum('customer', 'caregiver', 'admin', name='role_enum'), nullable=False, server_default=text("'customer'::role_enum"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    date_of_birth: Mapped[Optional[datetime.date]] = mapped_column(Date)
    phone_number: Mapped[Optional[str]] = mapped_column(String(20))
    otp_code: Mapped[Optional[str]] = mapped_column(Text)
    otp_expires_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    activity_logs: Mapped[list['ActivityLogs']] = relationship('ActivityLogs', back_populates='actor')
    cameras: Mapped[list['Cameras']] = relationship('Cameras', back_populates='user')
    caregiver_invitations: Mapped[list['CaregiverInvitations']] = relationship('CaregiverInvitations', back_populates='users')
    emergency_contacts: Mapped[list['EmergencyContacts']] = relationship('EmergencyContacts', back_populates='user')
    fcm_tokens: Mapped[list['FcmTokens']] = relationship('FcmTokens', back_populates='user')
    patient_supplements: Mapped[list['PatientSupplements']] = relationship('PatientSupplements', back_populates='customer')
    payments: Mapped[list['Payments']] = relationship('Payments', back_populates='user')
    shared_permissions: Mapped[list['SharedPermissions']] = relationship('SharedPermissions', foreign_keys='[SharedPermissions.caregiver_id]', back_populates='caregiver')
    shared_permissions_: Mapped[list['SharedPermissions']] = relationship('SharedPermissions', foreign_keys='[SharedPermissions.customer_id]', back_populates='customer')
    subscriptions: Mapped[list['Subscriptions']] = relationship('Subscriptions', back_populates='user')
    system_settings: Mapped[list['SystemSettings']] = relationship('SystemSettings', back_populates='users')
    ticket: Mapped[list['Ticket']] = relationship('Ticket', back_populates='user')
    uploads: Mapped[list['Uploads']] = relationship('Uploads', back_populates='user')
    user_settings: Mapped[list['UserSettings']] = relationship('UserSettings', back_populates='user')
    patient_habits: Mapped[list['PatientHabits']] = relationship('PatientHabits', back_populates='user')
    snapshots: Mapped[list['Snapshots']] = relationship('Snapshots', back_populates='user')
    event_detections: Mapped[list['EventDetections']] = relationship('EventDetections', foreign_keys='[EventDetections.user_id]', back_populates='user')
    event_detections_: Mapped[list['EventDetections']] = relationship('EventDetections', foreign_keys='[EventDetections.verified_by]', back_populates='users')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='users')
    notifications: Mapped[list['Notifications']] = relationship('Notifications', foreign_keys='[Notifications.acknowledged_by]', back_populates='users')
    notifications_: Mapped[list['Notifications']] = relationship('Notifications', foreign_keys='[Notifications.user_id]', back_populates='user')
