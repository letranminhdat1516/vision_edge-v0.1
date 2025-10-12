from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey, JSON, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class SharedPermissions(Base):
    __tablename__ = 'shared_permissions'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    stream_view = Column(Boolean, nullable=False)
    alert_read = Column(Boolean, nullable=False)
    alert_ack = Column(Boolean, nullable=False)
    profile_view = Column(Boolean, nullable=False)
    log_access_days = Column(Integer)
    report_access_days = Column(Integer)
    notification_channel = Column(JSON)
    permission_requests = Column(JSON)
    permission_scopes = Column(JSON)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class SystemSettings(Base):
    __tablename__ = 'system_settings'

    setting_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    setting_key = Column(String, nullable=False)
    setting_value = Column(Text, nullable=False)
    description = Column(Text)
    data_type = Column(String, nullable=False)
    category = Column(String)
    is_encrypted = Column(Boolean)
    updated_at = Column(DateTime, nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=False)

class SubscriptionEvents(Base):
    __tablename__ = 'subscription_events'

    id = Column(Integer, nullable=False)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String, nullable=False)
    event_data = Column(JSON)
    old_plan_code = Column(String)
    new_plan_code = Column(String)
    old_status = Column(String)
    new_status = Column(String)
    triggered_by = Column(UUID(as_uuid=True))
    reason = Column(String)
    tx_id = Column(UUID(as_uuid=True))
    payment_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime, nullable=False)

class SnapshotImages(Base):
    __tablename__ = 'snapshot_images'

    image_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    image_path = Column(Text)
    cloud_url = Column(Text)
    created_at = Column(DateTime, nullable=False)
    file_size = Column(Integer)

class Subscriptions(Base):
    __tablename__ = 'subscriptions'

    subscription_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    plan_code = Column(Text, nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    status = Column(String, nullable=False)
    billing_period = Column(String, nullable=False)
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
    version = Column(String)
    cancel_at_period_end = Column(Boolean, nullable=False)
    offer_start_date = Column(DateTime)
    offer_end_date = Column(DateTime)

class EmailTemplates(Base):
    __tablename__ = 'email_templates'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    subject_template = Column(String, nullable=False)
    html_template = Column(Text, nullable=False)
    text_template = Column(Text)
    variables = Column(JSON)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class Transactions(Base):
    __tablename__ = 'transactions'

    tx_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), nullable=False)
    plan_id = Column(UUID(as_uuid=True))
    plan_code = Column(Text, nullable=False)
    plan_snapshot = Column(JSON, nullable=False)
    amount_subtotal = Column(Integer, nullable=False)
    amount_discount = Column(Integer, nullable=False)
    amount_tax = Column(Integer, nullable=False)
    amount_total = Column(Integer, nullable=False)
    currency = Column(String, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    due_date = Column(DateTime)
    paid_at = Column(DateTime)
    payment_id = Column(UUID(as_uuid=True))
    effective_action = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    provider_payment_id = Column(String)
    idempotency_key = Column(String)
    related_tx_id = Column(UUID(as_uuid=True))
    notes = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_proration = Column(Boolean, nullable=False)
    plan_snapshot_new = Column(JSON)
    plan_snapshot_old = Column(JSON)
    proration_charge = Column(Integer, nullable=False)
    proration_credit = Column(Integer, nullable=False)
    version = Column(String)

class Snapshots(Base):
    __tablename__ = 'snapshots'

    snapshot_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    metadata = Column(JSON)
    capture_type = Column(String, nullable=False)
    captured_at = Column(DateTime, nullable=False)
    processed_at = Column(DateTime)
    is_processed = Column(Boolean, nullable=False)

class Plans(Base):
    __tablename__ = 'plans'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    code = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    price = Column(Integer, nullable=False)
    currency = Column(String, nullable=False)
    billing_period = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False)
    is_current = Column(Boolean, nullable=False)
    camera_quota = Column(Integer, nullable=False)
    storage_size = Column(Text)
    retention_days = Column(Integer, nullable=False)
    caregiver_seats = Column(Integer, nullable=False)
    sites = Column(Integer, nullable=False)
    major_updates_months = Column(Integer, nullable=False)
    version = Column(String)
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    is_recommended = Column(Boolean)
    successor_plan_code = Column(String)
    successor_plan_version = Column(String)
    tier = Column(Integer)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class Payments(Base):
    __tablename__ = 'payments'

    payment_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    status = Column(String, nullable=False)
    description = Column(Text)
    delivery_data = Column(JSON)
    vnp_txn_ref = Column(String)
    vnp_create_date = Column(Integer)
    vnp_expire_date = Column(Integer)
    vnp_order_info = Column(Text)
    version = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class PatientMedicalRecords(Base):
    __tablename__ = 'patient_medical_records'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    history = Column(JSON, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    name = Column(String)
    notes = Column(Text)

class PatientSupplements(Base):
    __tablename__ = 'patient_supplements'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    name = Column(Text)
    dob = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    customer_id = Column(UUID(as_uuid=True))
    call_confirmed_until = Column(DateTime)
    height_cm = Column(Integer)
    weight_kg = Column(String)

class EventDetections(Base):
    __tablename__ = 'event_detections'

    notes = Column(Text)
    event_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    snapshot_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String, nullable=False)
    event_description = Column(Text)
    detection_data = Column(JSON)
    ai_analysis_result = Column(JSON)
    confidence_score = Column(DECIMAL)
    bounding_boxes = Column(JSON)
    context_data = Column(JSON)
    detected_at = Column(DateTime, nullable=False)
    verified_at = Column(DateTime)
    verified_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    dismissed_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    confirm_status = Column(Boolean)
    status = Column(String)
    confirmation_state = Column(String, nullable=False)
    pending_until = Column(DateTime)
    proposed_status = Column(Text)
    proposed_event_type = Column(Text)
    proposed_reason = Column(Text)
    proposed_by = Column(UUID(as_uuid=True))

class PatientHabits(Base):
    __tablename__ = 'patient_habits'

    habit_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    habit_type = Column(String, nullable=False)
    habit_name = Column(String, nullable=False)
    description = Column(Text)
    frequency = Column(String, nullable=False)
    days_of_week = Column(JSON)
    location = Column(String)
    notes = Column(Text)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    supplement_id = Column(UUID(as_uuid=True))
    user_id = Column(UUID(as_uuid=True), nullable=False)
    sleep_start = Column(String)
    sleep_end = Column(String)

class Notifications(Base):
    __tablename__ = 'notifications'

    notification_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    event_id = Column(UUID(as_uuid=True))
    notification_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String)
    message = Column(Text, nullable=False)
    delivery_data = Column(JSON)
    status = Column(String, nullable=False)
    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)
    retry_count = Column(Integer)
    error_message = Column(Text)
    read_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True))
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)

class FcmTokens(Base):
    __tablename__ = 'fcm_tokens'

    token_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    device_id = Column(String)
    token = Column(Text, nullable=False)
    platform = Column(String, nullable=False)
    app_version = Column(String)
    device_model = Column(String)
    os_version = Column(String)
    topics = Column(JSON)
    is_active = Column(Boolean, nullable=False)
    last_used_at = Column(DateTime)
    revoked_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class EmergencyContacts(Base):
    __tablename__ = 'emergency_contacts'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String, nullable=False)
    relation = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    alert_level = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_deleted = Column(Boolean, nullable=False)

class CaregiverInvitations(Base):
    __tablename__ = 'caregiver_invitations'

    assignment_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    caregiver_id = Column(UUID(as_uuid=True), nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    assigned_at = Column(DateTime, nullable=False)
    unassigned_at = Column(DateTime)
    is_active = Column(Boolean, nullable=False)
    assigned_by = Column(UUID(as_uuid=True))
    assignment_notes = Column(Text)
    status = Column(String)

class Cameras(Base):
    __tablename__ = 'cameras'

    camera_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    camera_name = Column(String, nullable=False)
    camera_type = Column(String, nullable=False)
    ip_address = Column(String)
    port = Column(Integer)
    rtsp_url = Column(String)
    username = Column(String)
    password = Column(String)
    location_in_room = Column(String)
    resolution = Column(String)
    fps = Column(Integer)
    status = Column(String, nullable=False)
    last_ping = Column(DateTime)
    is_online = Column(Boolean, nullable=False)
    last_heartbeat_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class ActivityLogs(Base):
    __tablename__ = 'activity_logs'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False)
    actor_id = Column(UUID(as_uuid=True))
    actor_name = Column(String)
    action = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    message = Column(Text)
    severity = Column(String, nullable=False)
    meta = Column(JSON)
    ip = Column(String)
    action_enum = Column(String)
    resource_name = Column(String)

class Users(Base):
    __tablename__ = 'users'

    user_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    username = Column(String, nullable=False)
    email = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    date_of_birth = Column(String)
    phone_number = Column(String)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    otp_code = Column(Text)
    otp_expires_at = Column(DateTime)

class Roles(Base):
    __tablename__ = 'roles'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class RolePermissions(Base):
    __tablename__ = 'role_permissions'

    role_id = Column(UUID(as_uuid=True), nullable=False)
    permission_id = Column(UUID(as_uuid=True), nullable=False)
    assigned_at = Column(DateTime, nullable=False)

class Permissions(Base):
    __tablename__ = 'permissions'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class Uploads(Base):
    __tablename__ = 'uploads'

    upload_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    filename = Column(String, nullable=False)
    mime = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    url = Column(String, nullable=False)
    upload_type = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)

class UserSettings(Base):
    __tablename__ = 'user_settings'

    id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    category = Column(Text, nullable=False)
    setting_key = Column(String, nullable=False)
    setting_value = Column(Text, nullable=False)
    is_enabled = Column(Boolean, nullable=False)
    is_overridden = Column(Boolean, nullable=False)
    overridden_at = Column(DateTime)

class Ticket(Base):
    __tablename__ = 'ticket'

    ticket_id = Column(UUID(as_uuid=True), nullable=False, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    type = Column(String, nullable=False)
    title = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)

