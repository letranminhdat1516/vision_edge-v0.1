"""
Generated SQLAlchemy Models
Auto-generated from database schema
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()


class SharedPermissions(Base):
    __tablename__ = 'shared_permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    stream_view = Column(Boolean, nullable=False)
    alert_read = Column(Boolean, nullable=False)
    alert_ack = Column(Boolean, nullable=False)
    profile_view = Column(Boolean, nullable=False)
    log_access_days = Column(Integer)
    report_access_days = Column(Integer)
    notification_channel = Column(String)
    permission_requests = Column(String)
    permission_scopes = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class Users(Base):
    __tablename__ = 'users'
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    username = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(9), nullable=False)
    date_of_birth = Column(String)
    phone_number = Column(String(20))
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    otp_code = Column(Text)
    otp_expires_at = Column(DateTime)


class SystemSettings(Base):
    __tablename__ = 'system_settings'
    
    setting_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text, nullable=False)
    description = Column(Text)
    data_type = Column(String(7), nullable=False)
    category = Column(String(50))
    is_encrypted = Column(Boolean)
    updated_at = Column(DateTime, nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=False)


class SubscriptionEvents(Base):
    __tablename__ = 'subscription_events'
    
    id = Column(String, primary_key=True, nullable=False)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String(10), nullable=False)
    event_data = Column(String)
    old_plan_code = Column(String(50))
    new_plan_code = Column(String(50))
    old_status = Column(String(9))
    new_status = Column(String(9))
    triggered_by = Column(UUID(as_uuid=True))
    reason = Column(String(255))
    tx_id = Column(UUID(as_uuid=True))
    payment_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime, nullable=False)


class Payments(Base):
    __tablename__ = 'payments'
    
    payment_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    amount = Column(String, nullable=False)
    currency = Column(String(3), nullable=False)
    provider = Column(String(6), nullable=False)
    status = Column(String(20), nullable=False)
    description = Column(Text)
    delivery_data = Column(String)
    vnp_txn_ref = Column(String(50))
    vnp_create_date = Column(String)
    vnp_expire_date = Column(String)
    vnp_order_info = Column(Text)
    version = Column(String(20))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status_enum = Column(String(10))


class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    plan_code = Column(Text, nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    status = Column(String(9), nullable=False)
    billing_period = Column(String(7), nullable=False)
    started_at = Column(DateTime, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime)
    trial_end_at = Column(DateTime)
    canceled_at = Column(DateTime)
    ended_at = Column(DateTime)
    auto_renew = Column(Boolean, nullable=False)
    extra_camera_quota = Column(Integer, nullable=False)
    extra_caregiver_seats = Column(Integer, nullable=False)
    extra_sites = Column(Integer, nullable=False)
    extra_storage_gb = Column(Integer, nullable=False)
    notes = Column(Text)
    last_payment_at = Column(DateTime)
    version = Column(String(20))
    cancel_at_period_end = Column(Boolean, nullable=False)
    offer_start_date = Column(DateTime)
    offer_end_date = Column(DateTime)


class Plans(Base):
    __tablename__ = 'plans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    code = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(String, nullable=False)
    currency = Column(String(10), nullable=False)
    billing_period = Column(String(7), nullable=False)
    is_active = Column(Boolean, nullable=False)
    is_current = Column(Boolean, nullable=False)
    camera_quota = Column(Integer, nullable=False)
    storage_size = Column(Text)
    retention_days = Column(Integer, nullable=False)
    caregiver_seats = Column(Integer, nullable=False)
    sites = Column(Integer, nullable=False)
    major_updates_months = Column(Integer, nullable=False)
    version = Column(String(20))
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    is_recommended = Column(Boolean)
    successor_plan_code = Column(String(50))
    successor_plan_version = Column(String(20))
    tier = Column(Integer)
    status = Column(String(10), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class Transactions(Base):
    __tablename__ = 'transactions'
    
    tx_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    plan_code = Column(Text, nullable=False)
    plan_snapshot = Column(String, nullable=False)
    amount_subtotal = Column(String, nullable=False)
    amount_discount = Column(String, nullable=False)
    amount_tax = Column(String, nullable=False)
    amount_total = Column(String, nullable=False)
    currency = Column(String(3), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    status = Column(String(7), nullable=False)
    due_date = Column(DateTime)
    paid_at = Column(DateTime)
    payment_id = Column(UUID(as_uuid=True))
    effective_action = Column(String(10), nullable=False)
    provider = Column(String(6), nullable=False)
    provider_payment_id = Column(String(100))
    idempotency_key = Column(String(100))
    related_tx_id = Column(UUID(as_uuid=True))
    notes = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_proration = Column(Boolean, nullable=False)
    plan_snapshot_new = Column(String)
    plan_snapshot_old = Column(String)
    proration_charge = Column(String, nullable=False)
    proration_credit = Column(String, nullable=False)
    version = Column(String(20))


class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'
    
    image_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    image_path = Column(Text)
    cloud_url = Column(Text)
    created_at = Column(DateTime, nullable=False)
    file_size = Column(String)


class Snapshots(Base):
    __tablename__ = 'snapshots'
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    metadata = Column(String)
    capture_type = Column(String(16), nullable=False)
    captured_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime)
    is_processed = Column(Boolean, nullable=False)


class Cameras(Base):
    __tablename__ = 'cameras'
    
    camera_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_name = Column(String(100), nullable=False)
    camera_type = Column(String(4), nullable=False)
    ip_address = Column(String(45))
    port = Column(Integer)
    rtsp_url = Column(String(255))
    username = Column(String(50))
    password = Column(String(100))
    location_in_room = Column(String(50))
    resolution = Column(String(20))
    fps = Column(Integer)
    status = Column(String(8), nullable=False)
    last_ping = Column(DateTime)
    is_online = Column(Boolean, nullable=False)
    last_heartbeat_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class EmailTemplates(Base):
    __tablename__ = 'email_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    subject_template = Column(String(255), nullable=False)
    html_template = Column(Text, nullable=False)
    text_template = Column(Text)
    variables = Column(String)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class PatientMedicalRecords(Base):
    __tablename__ = 'patient_medical_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    history = Column(String, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    name = Column(String(200))
    notes = Column(Text)


class PatientSupplements(Base):
    __tablename__ = 'patient_supplements'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(Text)
    dob = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    customer_id = Column(UUID(as_uuid=True))
    call_confirmed_until = Column(DateTime)
    height_cm = Column(Integer)
    weight_kg = Column(String)


class Notifications(Base):
    __tablename__ = 'notifications'
    
    notification_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    event_id = Column(UUID(as_uuid=True))
    notification_type = Column(String(7), nullable=False)
    severity = Column(String(8), nullable=False)
    title = Column(String(255))
    message = Column(Text, nullable=False)
    delivery_data = Column(String)
    status = Column(String(9), nullable=False)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    retry_count = Column(Integer)
    error_message = Column(Text)
    read_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)
    channel = Column(String(7))
    business_type = Column(String(20))


class EventDetections(Base):
    __tablename__ = 'event_detections'
    
    notes = Column(Text)
    event_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String(17), nullable=False)
    event_description = Column(Text)
    detection_data = Column(String)
    ai_analysis_result = Column(String)
    confidence_score = Column(String)
    bounding_boxes = Column(String)
    context_data = Column(String)
    detected_at = Column(DateTime, nullable=False)
    verified_at = Column(DateTime)
    verified_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    dismissed_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    confirm_status = Column(Boolean)
    status = Column(String(7))
    confirmation_state = Column(String(21), nullable=False)
    pending_until = Column(DateTime)
    proposed_status = Column(Text)
    proposed_event_type = Column(Text)
    proposed_reason = Column(Text)
    proposed_by = Column(UUID(as_uuid=True))


class PatientHabits(Base):
    __tablename__ = 'patient_habits'
    
    habit_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    habit_type = Column(String(10), nullable=False)
    habit_name = Column(String(200), nullable=False)
    description = Column(Text)
    frequency = Column(String(6), nullable=False)
    days_of_week = Column(String)
    location = Column(String(100))
    notes = Column(Text)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    user_id = Column(UUID(as_uuid=True), nullable=False)
    sleep_start = Column(String)
    sleep_end = Column(String)


class FcmTokens(Base):
    __tablename__ = 'fcm_tokens'
    
    token_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    device_id = Column(String(100))
    token = Column(Text, nullable=False)
    platform = Column(String(7), nullable=False)
    app_version = Column(String(50))
    device_model = Column(String(100))
    os_version = Column(String(50))
    topics = Column(String)
    is_active = Column(Boolean, nullable=False)
    last_used_at = Column(DateTime)
    revoked_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class EmergencyContacts(Base):
    __tablename__ = 'emergency_contacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(100), nullable=False)
    relation = Column(String(50), nullable=False)
    phone = Column(String(20), nullable=False)
    alert_level = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_deleted = Column(Boolean, nullable=False)


class CaregiverInvitations(Base):
    __tablename__ = 'caregiver_invitations'
    
    assignment_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    assigned_at = Column(DateTime, nullable=False)
    unassigned_at = Column(DateTime)
    is_active = Column(Boolean, nullable=False)
    assigned_by = Column(UUID(as_uuid=True))
    assignment_notes = Column(Text)
    status = Column(String(20))


class ActivityLogs(Base):
    __tablename__ = 'activity_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    actor_id = Column(UUID(as_uuid=True))
    actor_name = Column(String(255))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    message = Column(Text)
    severity = Column(String(8), nullable=False)
    meta = Column(String)
    ip = Column(String(50))
    action_enum = Column(String(11))
    resource_name = Column(String)


class Roles(Base):
    __tablename__ = 'roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class RolePermissions(Base):
    __tablename__ = 'role_permissions'
    
    role_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    permission_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    assigned_at = Column(DateTime, nullable=False)


class Permissions(Base):
    __tablename__ = 'permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class Uploads(Base):
    __tablename__ = 'uploads'
    
    upload_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    filename = Column(String(255), nullable=False)
    mime = Column(String(100), nullable=False)
    size = Column(Integer, nullable=False)
    url = Column(String(500), nullable=False)
    upload_type = Column(String(12), nullable=False)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(String)


class UserSettings(Base):
    __tablename__ = 'user_settings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    category = Column(Text, nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text, nullable=False)
    is_enabled = Column(Boolean, nullable=False)
    is_overridden = Column(Boolean, nullable=False)
    overridden_at = Column(DateTime)


class Ticket(Base):
    __tablename__ = 'ticket'
    
    ticket_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    type = Column(String(7), nullable=False)
    title = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status = Column(String(12), nullable=False)

