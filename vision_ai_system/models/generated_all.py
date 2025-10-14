from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


class EmailTemplates(Base):
    __tablename__ = 'email_templates'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='email_templates_pkey'),
        Index('email_templates_name_key', 'name', unique=True),
        Index('idx_email_templates_active', 'is_active'),
        Index('idx_email_templates_type', 'type')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    subject_template: Mapped[str] = mapped_column(String(255), nullable=False)
    html_template: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    text_template: Mapped[Optional[str]] = mapped_column(Text)
    variables: Mapped[Optional[dict]] = mapped_column(JSONB)


class Permissions(Base):
    __tablename__ = 'permissions'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='permissions_pkey'),
        Index('idx_permissions_name', 'name'),
        Index('permissions_name_key', 'name', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(String(255))

    role_permissions: Mapped[list['RolePermissions']] = relationship('RolePermissions', back_populates='permission')


class Plans(Base):
    __tablename__ = 'plans'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='plans_pkey'),
        Index('plans_code_is_current_idx', 'code', 'is_current'),
        Index('plans_code_key', 'code', unique=True),
        Index('plans_is_active_idx', 'is_active')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    code: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    price: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, server_default=text("'VND'::character varying"))
    billing_period: Mapped[str] = mapped_column(Enum('monthly', 'yearly', 'none', name='billing_period_enum'), nullable=False, server_default=text("'monthly'::billing_period_enum"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    camera_quota: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('30'))
    caregiver_seats: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    sites: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('1'))
    major_updates_months: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('12'))
    status: Mapped[str] = mapped_column(Enum('draft', 'available', 'deprecated', 'archived', name='plan_status_enum'), nullable=False, server_default=text("'available'::plan_status_enum"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(Text)
    storage_size: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[Optional[str]] = mapped_column(String(20))
    effective_from: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(precision=3))
    effective_to: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(precision=3))
    is_recommended: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('false'))
    successor_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    successor_plan_version: Mapped[Optional[str]] = mapped_column(String(20))
    tier: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('1'))

    subscriptions: Mapped[list['Subscriptions']] = relationship('Subscriptions', back_populates='plan')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='plan')


class Roles(Base):
    __tablename__ = 'roles'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='roles_pkey'),
        Index('idx_roles_name', 'name'),
        Index('roles_name_key', 'name', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(String(255))

    role_permissions: Mapped[list['RolePermissions']] = relationship('RolePermissions', back_populates='role')


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


class EmergencyContacts(Base):
    __tablename__ = 'emergency_contacts'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='emergency_contacts_user_id_fkey'),
        PrimaryKeyConstraint('id', name='emergency_contacts_pkey'),
        Index('idx_emergency_contacts_alert_level', 'alert_level'),
        Index('idx_emergency_contacts_phone', 'phone'),
        Index('idx_emergency_contacts_user_id', 'user_id'),
        Index('unique_user_phone', 'user_id', 'phone', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    relation: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    alert_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))

    user: Mapped['Users'] = relationship('Users', back_populates='emergency_contacts')


class FcmTokens(Base):
    __tablename__ = 'fcm_tokens'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='fcm_tokens_user_id_fkey'),
        PrimaryKeyConstraint('token_id', name='fcm_tokens_pkey'),
        Index('fcm_tokens_token_key', 'token', unique=True),
        Index('idx_fcm_active', 'is_active'),
        Index('idx_fcm_last_used', 'last_used_at'),
        Index('idx_fcm_platform', 'platform'),
        Index('idx_fcm_user', 'user_id'),
        Index('unique_fcm_user_device', 'user_id', 'device_id', unique=True)
    )

    token_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    token: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(Enum('ios', 'android', 'web', name='push_platform_enum'), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    device_id: Mapped[Optional[str]] = mapped_column(String(100))
    app_version: Mapped[Optional[str]] = mapped_column(String(50))
    device_model: Mapped[Optional[str]] = mapped_column(String(100))
    os_version: Mapped[Optional[str]] = mapped_column(String(50))
    topics: Mapped[Optional[dict]] = mapped_column(JSONB)
    last_used_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    revoked_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    user: Mapped['Users'] = relationship('Users', back_populates='fcm_tokens')


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


class Payments(Base):
    __tablename__ = 'payments'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='payments_user_id_fkey'),
        PrimaryKeyConstraint('payment_id', name='payments_pkey'),
        Index('idx_pay_status', 'status'),
        Index('idx_pay_user', 'user_id'),
        Index('payments_vnp_txn_ref_key', 'vnp_txn_ref', unique=True)
    )

    payment_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default=text("'VND'::character varying"))
    provider: Mapped[str] = mapped_column(Enum('vn_pay', 'stripe', 'manual', name='PaymentProvider'), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'pending'::character varying"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    description: Mapped[Optional[str]] = mapped_column(Text)
    delivery_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    vnp_txn_ref: Mapped[Optional[str]] = mapped_column(String(50))
    vnp_create_date: Mapped[Optional[int]] = mapped_column(BigInteger)
    vnp_expire_date: Mapped[Optional[int]] = mapped_column(BigInteger)
    vnp_order_info: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[Optional[str]] = mapped_column(String(20))

    user: Mapped['Users'] = relationship('Users', back_populates='payments')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='payment')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='payment')


class RolePermissions(Base):
    __tablename__ = 'role_permissions'
    __table_args__ = (
        ForeignKeyConstraint(['permission_id'], ['permissions.id'], ondelete='CASCADE', onupdate='CASCADE', name='role_permissions_permission_id_fkey'),
        ForeignKeyConstraint(['role_id'], ['roles.id'], ondelete='CASCADE', onupdate='CASCADE', name='role_permissions_role_id_fkey'),
        PrimaryKeyConstraint('role_id', 'permission_id', name='role_permissions_pkey'),
        Index('idx_role_permissions_permission', 'permission_id'),
        Index('idx_role_permissions_role', 'role_id')
    )

    role_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    permission_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    assigned_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))

    permission: Mapped['Permissions'] = relationship('Permissions', back_populates='role_permissions')
    role: Mapped['Roles'] = relationship('Roles', back_populates='role_permissions')


class SharedPermissions(Base):
    __tablename__ = 'shared_permissions'
    __table_args__ = (
        ForeignKeyConstraint(['caregiver_id'], ['users.user_id'], name='shared_permissions_caregiver_id_fkey'),
        ForeignKeyConstraint(['customer_id'], ['users.user_id'], name='shared_permissions_customer_id_fkey'),
        PrimaryKeyConstraint('id', name='shared_permissions_pkey'),
        Index('idx_shared_permissions_caregiver', 'caregiver_id'),
        Index('idx_unique_shared_permission_pair', 'customer_id', 'caregiver_id', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    customer_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    caregiver_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    stream_view: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    alert_read: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    alert_ack: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    profile_view: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    log_access_days: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('0'))
    report_access_days: Mapped[Optional[int]] = mapped_column(Integer, server_default=text('0'))
    notification_channel: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    permission_requests: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    permission_scopes: Mapped[Optional[dict]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))

    caregiver: Mapped['Users'] = relationship('Users', foreign_keys=[caregiver_id], back_populates='shared_permissions')
    customer: Mapped['Users'] = relationship('Users', foreign_keys=[customer_id], back_populates='shared_permissions_')


class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    __table_args__ = (
        ForeignKeyConstraint(['plan_id'], ['plans.id'], ondelete='SET NULL', onupdate='CASCADE', name='subscriptions_plan_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='subscriptions_user_id_fkey'),
        PrimaryKeyConstraint('subscription_id', name='subscriptions_pkey'),
        Index('idx_sub_current_end', 'current_period_end'),
        Index('idx_sub_plan', 'plan_code'),
        Index('idx_sub_plan_id', 'plan_id'),
        Index('idx_sub_status', 'status'),
        Index('idx_sub_user', 'user_id'),
        Index('idx_sub_user_status', 'user_id', 'status')
    )

    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    plan_code: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'), nullable=False, server_default=text("'active'::subscription_status_enum"))
    billing_period: Mapped[str] = mapped_column(Enum('monthly', 'yearly', 'none', name='billing_period_enum'), nullable=False, server_default=text("'none'::billing_period_enum"))
    started_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    current_period_start: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    extra_camera_quota: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_caregiver_seats: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_sites: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    extra_storage_gb: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text('0'))
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    plan_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    current_period_end: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    trial_end_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    canceled_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    ended_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    last_payment_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    version: Mapped[Optional[str]] = mapped_column(String(20))
    offer_start_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    offer_end_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    plan: Mapped[Optional['Plans']] = relationship('Plans', back_populates='subscriptions')
    user: Mapped['Users'] = relationship('Users', back_populates='subscriptions')
    transactions: Mapped[list['Transactions']] = relationship('Transactions', back_populates='subscription')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='subscription')


class SystemSettings(Base):
    __tablename__ = 'system_settings'
    __table_args__ = (
        ForeignKeyConstraint(['updated_by'], ['users.user_id'], ondelete='RESTRICT', onupdate='CASCADE', name='system_settings_updated_by_fkey'),
        PrimaryKeyConstraint('setting_id', name='system_settings_pkey'),
        Index('idx_ss_category', 'category'),
        Index('idx_ss_dtype', 'data_type'),
        Index('idx_ss_key', 'setting_key'),
        Index('idx_ss_updated_by', 'updated_by'),
        Index('idx_system_settings_key', 'setting_key', unique=True)
    )

    setting_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)
    setting_value: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(Enum('string', 'int', 'float', 'boolean', 'json', name='data_type_enum'), nullable=False, server_default=text("'string'::data_type_enum"))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(50), server_default=text("'general'::character varying"))
    is_encrypted: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('false'))

    users: Mapped['Users'] = relationship('Users', back_populates='system_settings')


class Ticket(Base):
    __tablename__ = 'ticket'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='ticket_user_id_fkey'),
        PrimaryKeyConstraint('ticket_id', name='ticket_pkey'),
        Index('idx_customer_requests_created_at', 'created_at'),
        Index('idx_customer_requests_status', 'status'),
        Index('idx_customer_requests_type', 'type'),
        Index('idx_customer_requests_user', 'user_id')
    )

    ticket_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    type: Mapped[str] = mapped_column(Enum('report', 'support', name='customer_request_type_enum'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    status: Mapped[str] = mapped_column(Enum('new', 'acknowledged', 'in_progress', 'resolved', 'rejected', name='customer_request_status_enum'), nullable=False, server_default=text("'new'::customer_request_status_enum"))
    title: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)

    user: Mapped['Users'] = relationship('Users', back_populates='ticket')


class Uploads(Base):
    __tablename__ = 'uploads'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', onupdate='CASCADE', name='uploads_user_id_fkey'),
        PrimaryKeyConstraint('upload_id', name='uploads_pkey'),
        Index('idx_uploads_created', 'created_at'),
        Index('idx_uploads_user', 'user_id')
    )

    upload_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime: Mapped[str] = mapped_column(String(100), nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    upload_type: Mapped[str] = mapped_column(Enum('camera_error', 'other', name='upload_type_enum'), nullable=False, server_default=text("'other'::upload_type_enum"))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    metadata_: Mapped[Optional[dict]] = mapped_column('metadata', JSONB)

    user: Mapped['Users'] = relationship('Users', back_populates='uploads')


class UserSettings(Base):
    __tablename__ = 'user_settings'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE', name='user_settings_user_id_fkey'),
        PrimaryKeyConstraint('id', name='user_settings_pkey'),
        Index('uq_user_setting', 'user_id', 'category', 'setting_key', unique=True)
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)
    setting_value: Mapped[str] = mapped_column(Text, nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('true'))
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    overridden_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))

    user: Mapped['Users'] = relationship('Users', back_populates='user_settings')


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


class Transactions(Base):
    __tablename__ = 'transactions'
    __table_args__ = (
        ForeignKeyConstraint(['payment_id'], ['payments.payment_id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_payment_id_fkey'),
        ForeignKeyConstraint(['plan_id'], ['plans.id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_plan_id_fkey'),
        ForeignKeyConstraint(['related_tx_id'], ['transactions.tx_id'], ondelete='SET NULL', onupdate='CASCADE', name='transactions_related_tx_id_fkey'),
        ForeignKeyConstraint(['subscription_id'], ['subscriptions.subscription_id'], ondelete='CASCADE', onupdate='CASCADE', name='transactions_subscription_id_fkey'),
        PrimaryKeyConstraint('tx_id', name='transactions_pkey'),
        Index('idx_tx_period_end', 'period_end'),
        Index('idx_tx_period_start', 'period_start'),
        Index('idx_tx_plan_code', 'plan_code'),
        Index('idx_tx_plan_id', 'plan_id'),
        Index('idx_tx_provider_payment_id', 'provider_payment_id'),
        Index('idx_tx_status', 'status'),
        Index('idx_tx_sub', 'subscription_id'),
        Index('uq_tx_idem_per_sub', 'subscription_id', 'idempotency_key', unique=True),
        Index('uq_tx_payment_id', 'payment_id', unique=True)
    )

    tx_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    plan_code: Mapped[str] = mapped_column(Text, nullable=False)
    plan_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    amount_subtotal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount_discount: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    amount_tax: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    amount_total: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default=text("'VND'::character varying"))
    period_start: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    period_end: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    status: Mapped[str] = mapped_column(Enum('draft', 'open', 'paid', 'void', 'overdue', name='invoice_status_enum'), nullable=False, server_default=text("'draft'::invoice_status_enum"))
    effective_action: Mapped[str] = mapped_column(Enum('new', 'renew', 'upgrade', 'downgrade', 'adjustment', name='TransactionAction'), nullable=False)
    provider: Mapped[str] = mapped_column(Enum('vn_pay', 'stripe', 'manual', name='PaymentProvider'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False)
    is_proration: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text('false'))
    proration_charge: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    proration_credit: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text('0'))
    plan_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    due_date: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    paid_at: Mapped[Optional[datetime.datetime]] = mapped_column(TIMESTAMP(True, 6))
    payment_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    provider_payment_id: Mapped[Optional[str]] = mapped_column(String(100))
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(100))
    related_tx_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    plan_snapshot_new: Mapped[Optional[dict]] = mapped_column(JSONB)
    plan_snapshot_old: Mapped[Optional[dict]] = mapped_column(JSONB)
    version: Mapped[Optional[str]] = mapped_column(String(20))

    payment: Mapped[Optional['Payments']] = relationship('Payments', back_populates='transactions')
    plan: Mapped[Optional['Plans']] = relationship('Plans', back_populates='transactions')
    related_tx: Mapped[Optional['Transactions']] = relationship('Transactions', remote_side=[tx_id], back_populates='related_tx_reverse')
    related_tx_reverse: Mapped[list['Transactions']] = relationship('Transactions', remote_side=[related_tx_id], back_populates='related_tx')
    subscription: Mapped['Subscriptions'] = relationship('Subscriptions', back_populates='transactions')
    subscription_events: Mapped[list['SubscriptionEvents']] = relationship('SubscriptionEvents', back_populates='tx')


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


class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'
    __table_args__ = (
        ForeignKeyConstraint(['snapshot_id'], ['snapshots.snapshot_id'], ondelete='CASCADE', name='fk_snapshot_images_snapshot'),
        PrimaryKeyConstraint('image_id', name='snapshot_images_pkey'),
        Index('idx_snapshot_images_created_at', 'created_at'),
        Index('idx_snapshot_images_snapshot_id', 'snapshot_id')
    )

    image_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    snapshot_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    image_path: Mapped[Optional[str]] = mapped_column(Text)
    cloud_url: Mapped[Optional[str]] = mapped_column(Text)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)

    snapshot: Mapped['Snapshots'] = relationship('Snapshots', back_populates='snapshot_images')


class SubscriptionEvents(Base):
    __tablename__ = 'subscription_events'
    __table_args__ = (
        ForeignKeyConstraint(['payment_id'], ['payments.payment_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_payment_id_fkey'),
        ForeignKeyConstraint(['subscription_id'], ['subscriptions.subscription_id'], ondelete='CASCADE', onupdate='CASCADE', name='subscription_events_subscription_id_fkey'),
        ForeignKeyConstraint(['triggered_by'], ['users.user_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_triggered_by_fkey'),
        ForeignKeyConstraint(['tx_id'], ['transactions.tx_id'], ondelete='SET NULL', onupdate='CASCADE', name='subscription_events_tx_id_fkey'),
        PrimaryKeyConstraint('id', name='subscription_events_pkey'),
        Index('idx_sub_events_composite', 'subscription_id', 'event_type', 'created_at'),
        Index('idx_sub_events_new_plan', 'new_plan_code'),
        Index('idx_sub_events_old_plan', 'old_plan_code'),
        Index('idx_sub_events_payment', 'payment_id'),
        Index('idx_sub_events_sub', 'subscription_id'),
        Index('idx_sub_events_triggered_by', 'triggered_by'),
        Index('idx_sub_events_tx', 'tx_id'),
        Index('idx_sub_events_type', 'event_type')
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subscription_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    event_type: Mapped[str] = mapped_column(Enum('created', 'activated', 'renewed', 'upgraded', 'downgraded', 'paused', 'resumed', 'canceled', 'expired', name='subscription_event_type_enum'), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    event_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    old_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    new_plan_code: Mapped[Optional[str]] = mapped_column(String(50))
    old_status: Mapped[Optional[str]] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'))
    new_status: Mapped[Optional[str]] = mapped_column(Enum('trialing', 'active', 'past_due', 'paused', 'suspended', 'canceled', 'expired', name='subscription_status_enum'))
    triggered_by: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    reason: Mapped[Optional[str]] = mapped_column(String(255))
    tx_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)
    payment_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid)

    payment: Mapped[Optional['Payments']] = relationship('Payments', back_populates='subscription_events')
    subscription: Mapped['Subscriptions'] = relationship('Subscriptions', back_populates='subscription_events')
    users: Mapped[Optional['Users']] = relationship('Users', back_populates='subscription_events')
    tx: Mapped[Optional['Transactions']] = relationship('Transactions', back_populates='subscription_events')


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
