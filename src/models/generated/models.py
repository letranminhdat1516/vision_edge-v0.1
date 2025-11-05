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
    idempotency_key = Column(String(128))
    vnp_create_date = Column(String)
    vnp_expire_date = Column(String)
    vnp_order_info = Column(Text)
    version = Column(String(20))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status_enum = Column(String(10))


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
    default_payment_method_id = Column(UUID(as_uuid=True))


class Plans(Base):
    __tablename__ = 'plans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    code = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(String, nullable=False)
    currency = Column(String(10), nullable=False)
    billing_period = Column(String(10), nullable=False)
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


class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    plan_code = Column(Text, nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    status = Column(String(9), nullable=False)
    billing_period = Column(String(10), nullable=False)
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
    renewal_attempt_count = Column(Integer, nullable=False)
    next_renew_attempt_at = Column(DateTime)
    dunning_stage = Column(String(50))
    last_renewal_error = Column(Text)
    plan_snapshot = Column(String)
    unit_amount_minor = Column(String, nullable=False)


class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'
    
    image_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    is_primary = Column(Boolean, nullable=False)
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


class PrismaMigrations(Base):
    __tablename__ = '_prisma_migrations'
    
    id = Column(String(36), primary_key=True, nullable=False)
    checksum = Column(String(64), nullable=False)
    finished_at = Column(DateTime)
    migration_name = Column(String(255), nullable=False)
    logs = Column(Text)
    rolled_back_at = Column(DateTime)
    started_at = Column(DateTime, nullable=False)
    applied_steps_count = Column(Integer, nullable=False)


class AccessGrants(Base):
    __tablename__ = 'access_grants'
    
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
    category = Column(String(50))
    priority = Column(String(20))
    assigned_to = Column(UUID(as_uuid=True))
    tags = Column(String)
    metadata = Column(String)
    due_date = Column(DateTime)
    resolved_at = Column(DateTime)
    closed_at = Column(DateTime)


class SystemConfig(Base):
    __tablename__ = 'system_config'
    
    setting_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text, nullable=False)
    description = Column(Text)
    data_type = Column(String(7), nullable=False)
    category = Column(String(50))
    is_encrypted = Column(Boolean)
    updated_at = Column(DateTime, nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=False)


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
    responded_at = Column(DateTime)
    expires_at = Column(DateTime)
    response_reason = Column(String(255))
    status = Column(String(9), nullable=False)


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
    status = Column(String(8))
    confirmation_state = Column(String(21), nullable=False)
    pending_until = Column(DateTime)
    proposed_reason = Column(Text)
    proposed_by = Column(UUID(as_uuid=True))
    proposed_status = Column(String(8))
    proposed_event_type = Column(String(17))
    auto_escalation_reason = Column(Text)
    escalated_at = Column(DateTime)
    escalation_count = Column(Integer, nullable=False)
    is_canceled = Column(Boolean, nullable=False)
    last_action_at = Column(DateTime)
    last_action_by = Column(UUID(as_uuid=True))
    pending_since = Column(DateTime)
    verification_status = Column(String(9), nullable=False)
    notification_attempts = Column(Integer, nullable=False)
    lifecycle_state = Column(String(15), nullable=False)
    reliability_score = Column(String)


class Notifications(Base):
    __tablename__ = 'notifications'
    
    notification_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    event_id = Column(UUID(as_uuid=True))
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
    channel = Column(String(7), nullable=False)
    business_type = Column(String(20))


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


class PatientMedicalRecords(Base):
    __tablename__ = 'patient_medical_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    history = Column(String, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    name = Column(String(200))
    notes = Column(Text)


class Permissions(Base):
    __tablename__ = 'permissions'
    
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    permission_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)


class Roles(Base):
    __tablename__ = 'roles'
    
    name = Column(String(50), nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    role_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)


class RolePermissions(Base):
    __tablename__ = 'role_permissions'
    
    role_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    permission_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    assigned_at = Column(DateTime, nullable=False)


class UserRoles(Base):
    __tablename__ = 'user_roles'
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    role_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    assigned_at = Column(DateTime, nullable=False)


class EventHistory(Base):
    __tablename__ = 'event_history'
    
    history_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    event_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(18), nullable=False)
    actor_id = Column(UUID(as_uuid=True))
    actor_name = Column(String(255))
    actor_role = Column(String(50))
    previous_status = Column(String(50))
    new_status = Column(String(50))
    previous_event_type = Column(String(17))
    new_event_type = Column(String(17))
    previous_confirmation_state = Column(String(21))
    new_confirmation_state = Column(String(21))
    reason = Column(Text)
    metadata = Column(String)
    created_at = Column(DateTime, nullable=False)
    response_time_minutes = Column(Integer)
    is_first_action = Column(Boolean)


class TicketHistory(Base):
    __tablename__ = 'ticket_history'
    
    history_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    ticket_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(30), nullable=False)
    old_values = Column(String)
    new_values = Column(String)
    description = Column(Text)
    metadata = Column(String)
    created_at = Column(DateTime, nullable=False)


class UserPreferences(Base):
    __tablename__ = 'user_preferences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    category = Column(Text, nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text, nullable=False)
    is_enabled = Column(Boolean, nullable=False)
    is_overridden = Column(Boolean, nullable=False)
    overridden_at = Column(DateTime)


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

