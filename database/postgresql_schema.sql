-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.activity_logs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  timestamp timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  actor_id uuid,
  actor_name character varying,
  action character varying NOT NULL,
  resource_type character varying,
  resource_id character varying,
  message text,
  severity USER-DEFINED NOT NULL DEFAULT 'info'::activity_severity_enum,
  meta jsonb,
  ip character varying,
  action_enum USER-DEFINED,
  resource_name character varying,
  CONSTRAINT activity_logs_pkey PRIMARY KEY (id),
  CONSTRAINT fk_activity_logs_actor FOREIGN KEY (actor_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.ai_configurations (
  config_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  patient_profile_context jsonb,
  behavior_rules jsonb,
  model_settings jsonb,
  detection_thresholds jsonb,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_by uuid NOT NULL,
  CONSTRAINT ai_configurations_pkey PRIMARY KEY (config_id),
  CONSTRAINT ai_configurations_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(user_id),
  CONSTRAINT ai_configurations_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.ai_processing_logs (
  log_id uuid NOT NULL DEFAULT gen_random_uuid(),
  snapshot_id uuid NOT NULL,
  user_id uuid NOT NULL,
  processing_stage USER-DEFINED NOT NULL,
  input_data jsonb,
  output_data jsonb,
  processing_time_ms integer DEFAULT 0,
  result_status USER-DEFINED NOT NULL,
  error_message text,
  model_version character varying,
  processed_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT ai_processing_logs_pkey PRIMARY KEY (log_id),
  CONSTRAINT ai_processing_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT ai_processing_logs_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.snapshots(snapshot_id)
);
CREATE TABLE public.alert_settings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  key character varying NOT NULL,
  value text,
  is_enabled boolean NOT NULL DEFAULT true,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_by uuid,
  CONSTRAINT alert_settings_pkey PRIMARY KEY (id),
  CONSTRAINT alert_settings_user_fk FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT alert_settings_updated_by_fkey FOREIGN KEY (updated_by) REFERENCES public.users(user_id)
);
CREATE TABLE public.alerts (
  alert_id uuid NOT NULL DEFAULT gen_random_uuid(),
  event_id uuid NOT NULL,
  user_id uuid NOT NULL,
  alert_type USER-DEFINED NOT NULL,
  severity USER-DEFINED NOT NULL DEFAULT 'medium'::severity_enum,
  alert_message text NOT NULL,
  alert_data jsonb,
  status USER-DEFINED NOT NULL DEFAULT 'active'::alert_status_enum,
  acknowledged_by uuid,
  acknowledged_at timestamp with time zone,
  resolution_notes text,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolved_at timestamp with time zone,
  CONSTRAINT alerts_pkey PRIMARY KEY (alert_id),
  CONSTRAINT alerts_event_id_fkey FOREIGN KEY (event_id) REFERENCES public.event_detections(event_id),
  CONSTRAINT alerts_acknowledged_by_fkey FOREIGN KEY (acknowledged_by) REFERENCES public.users(user_id),
  CONSTRAINT alerts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.app_client_roles (
  app_id uuid NOT NULL,
  role USER-DEFINED NOT NULL,
  CONSTRAINT app_client_roles_pkey PRIMARY KEY (app_id, role),
  CONSTRAINT app_client_roles_app_id_fkey FOREIGN KEY (app_id) REFERENCES public.app_clients(app_id)
);
CREATE TABLE public.app_clients (
  app_id uuid NOT NULL DEFAULT gen_random_uuid(),
  code character varying NOT NULL,
  name text NOT NULL,
  CONSTRAINT app_clients_pkey PRIMARY KEY (app_id)
);
CREATE TABLE public.camera_settings (
  setting_id uuid NOT NULL DEFAULT gen_random_uuid(),
  camera_id uuid NOT NULL,
  setting_name character varying NOT NULL,
  setting_value text NOT NULL,
  data_type USER-DEFINED NOT NULL DEFAULT 'string'::data_type_enum,
  description text,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT camera_settings_pkey PRIMARY KEY (setting_id),
  CONSTRAINT camera_settings_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id)
);
CREATE TABLE public.cameras (
  camera_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  room_id uuid,
  camera_name character varying NOT NULL,
  camera_type USER-DEFINED NOT NULL DEFAULT 'ip'::camera_type_enum,
  ip_address character varying,
  port integer DEFAULT 80,
  rtsp_url character varying,
  username character varying,
  password character varying,
  location_in_room character varying,
  resolution character varying DEFAULT '1920x1080'::character varying,
  fps integer DEFAULT 30,
  status USER-DEFINED NOT NULL DEFAULT 'active'::camera_status_enum,
  last_ping timestamp with time zone,
  is_online boolean NOT NULL DEFAULT true,
  last_heartbeat_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT cameras_pkey PRIMARY KEY (camera_id),
  CONSTRAINT cameras_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT cameras_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id)
);
CREATE TABLE public.caregiver_assignments (
  assignment_id uuid NOT NULL DEFAULT gen_random_uuid(),
  caregiver_id uuid NOT NULL,
  customer_id uuid NOT NULL,
  assigned_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  unassigned_at timestamp with time zone,
  is_active boolean NOT NULL DEFAULT true,
  assigned_by uuid,
  assignment_notes text,
  status character varying DEFAULT 'pending'::character varying,
  CONSTRAINT caregiver_assignments_pkey PRIMARY KEY (assignment_id),
  CONSTRAINT caregiver_assignments_assigned_by_fkey FOREIGN KEY (assigned_by) REFERENCES public.users(user_id)
);
CREATE TABLE public.caregivers (
  caregiver_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  CONSTRAINT caregivers_pkey PRIMARY KEY (caregiver_id),
  CONSTRAINT caregivers_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.customer_requests (
  request_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  type USER-DEFINED NOT NULL,
  status USER-DEFINED NOT NULL,
  title text,
  description text,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT customer_requests_pkey PRIMARY KEY (request_id),
  CONSTRAINT customer_requests_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.customers (
  customer_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  CONSTRAINT customers_pkey PRIMARY KEY (customer_id),
  CONSTRAINT customers_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.daily_summaries (
  summary_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  summary_date date NOT NULL,
  activity_summary jsonb,
  habit_compliance jsonb,
  event_summary jsonb,
  behavior_patterns jsonb,
  total_snapshots integer DEFAULT 0,
  total_events integer DEFAULT 0,
  total_alerts integer DEFAULT 0,
  sleep_quality_score numeric DEFAULT 0.00,
  activity_level_score numeric DEFAULT 0.00,
  overall_status USER-DEFINED NOT NULL DEFAULT 'good'::overall_status_enum,
  notes text,
  generated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT daily_summaries_pkey PRIMARY KEY (summary_id),
  CONSTRAINT daily_summaries_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.emergency_contacts (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name text NOT NULL,
  relation text NOT NULL,
  phone text NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  alert_level smallint,
  supplement_id uuid,
  CONSTRAINT emergency_contacts_pkey PRIMARY KEY (id),
  CONSTRAINT emergency_contacts_supplement_id_fkey FOREIGN KEY (supplement_id) REFERENCES public.patient_supplements(id)
);
CREATE TABLE public.event_detections (
  notes text,
  event_id uuid NOT NULL DEFAULT gen_random_uuid(),
  snapshot_id uuid NOT NULL,
  user_id uuid NOT NULL,
  camera_id uuid NOT NULL,
  room_id uuid NOT NULL,
  event_type USER-DEFINED NOT NULL,
  event_description text,
  detection_data jsonb,
  ai_analysis_result jsonb,
  confidence_score numeric DEFAULT 0.00,
  bounding_boxes jsonb,
  context_data jsonb,
  detected_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  verified_at timestamp with time zone,
  verified_by uuid,
  acknowledged_at timestamp with time zone,
  acknowledged_by uuid,
  dismissed_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  confirm_status boolean,
  status USER-DEFINED,
  CONSTRAINT event_detections_pkey PRIMARY KEY (event_id),
  CONSTRAINT event_detections_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT event_detections_verified_by_fkey FOREIGN KEY (verified_by) REFERENCES public.users(user_id),
  CONSTRAINT event_detections_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.snapshots(snapshot_id),
  CONSTRAINT event_detections_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id),
  CONSTRAINT event_detections_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id)
);
CREATE TABLE public.fall_detection_settings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  abnormal_unconfirmed_streak integer NOT NULL DEFAULT 5,
  abnormal_streak_window_minutes integer NOT NULL DEFAULT 30,
  only_trigger_if_unconfirmed boolean DEFAULT true,
  enabled boolean DEFAULT true,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  user_id uuid UNIQUE,
  CONSTRAINT fall_detection_settings_pkey PRIMARY KEY (id),
  CONSTRAINT fk_fall_user FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.fcm_tokens (
  token_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  device_id character varying,
  token text NOT NULL,
  platform USER-DEFINED NOT NULL,
  app_version character varying,
  device_model character varying,
  os_version character varying,
  topics jsonb,
  is_active boolean NOT NULL DEFAULT true,
  last_used_at timestamp with time zone,
  revoked_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fcm_tokens_pkey PRIMARY KEY (token_id),
  CONSTRAINT fcm_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.image_settings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  key character varying NOT NULL,
  value text,
  is_enabled boolean NOT NULL DEFAULT true,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_by uuid,
  CONSTRAINT image_settings_pkey PRIMARY KEY (id),
  CONSTRAINT image_settings_user_fk FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT image_settings_updated_by_fkey FOREIGN KEY (updated_by) REFERENCES public.users(user_id)
);
CREATE TABLE public.license_activations (
  id bigint NOT NULL DEFAULT nextval('license_activations_id_seq'::regclass),
  license_id uuid NOT NULL,
  site_id text NOT NULL,
  activated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT license_activations_pkey PRIMARY KEY (id),
  CONSTRAINT license_activations_license_id_fkey FOREIGN KEY (license_id) REFERENCES public.licenses(license_id)
);
CREATE TABLE public.licenses (
  license_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid,
  payment_id uuid,
  plan_code text NOT NULL,
  key text NOT NULL,
  major_updates_until timestamp with time zone NOT NULL,
  created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT licenses_pkey PRIMARY KEY (license_id),
  CONSTRAINT fk_licenses_users FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT licenses_plan_code_fkey FOREIGN KEY (plan_code) REFERENCES public.plans(code),
  CONSTRAINT licenses_payment_id_fkey FOREIGN KEY (payment_id) REFERENCES public.payments(payment_id)
);
CREATE TABLE public.notification_preferences (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  system_events_enabled boolean NOT NULL DEFAULT true,
  actor_messages_enabled boolean NOT NULL DEFAULT true,
  push_notifications_enabled boolean NOT NULL DEFAULT true,
  email_notifications_enabled boolean NOT NULL DEFAULT false,
  quiet_hours_start time without time zone,
  quiet_hours_end time without time zone,
  fall_detection_enabled boolean NOT NULL DEFAULT true,
  seizure_detection_enabled boolean NOT NULL DEFAULT true,
  abnormal_behavior_enabled boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT notification_preferences_pkey PRIMARY KEY (id)
);
CREATE TABLE public.notifications (
  notification_id uuid NOT NULL DEFAULT gen_random_uuid(),
  alert_id uuid NOT NULL,
  user_id uuid NOT NULL,
  notification_type USER-DEFINED NOT NULL,
  message text NOT NULL,
  delivery_data jsonb,
  status USER-DEFINED NOT NULL DEFAULT 'pending'::notif_status_enum,
  sent_at timestamp with time zone,
  delivered_at timestamp with time zone,
  retry_count integer DEFAULT 0,
  error_message text,
  CONSTRAINT notifications_pkey PRIMARY KEY (notification_id),
  CONSTRAINT notifications_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT notifications_alert_id_fkey FOREIGN KEY (alert_id) REFERENCES public.alerts(alert_id)
);
CREATE TABLE public.patient_habits (
  habit_id uuid NOT NULL DEFAULT gen_random_uuid(),
  habit_type USER-DEFINED NOT NULL,
  habit_name character varying NOT NULL,
  description text,
  typical_time time without time zone,
  duration_minutes integer DEFAULT 30,
  frequency USER-DEFINED NOT NULL DEFAULT 'daily'::frequency_enum,
  days_of_week jsonb,
  location character varying,
  notes text,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  supplement_id uuid,
  usersUser_id uuid,
  CONSTRAINT patient_habits_pkey PRIMARY KEY (habit_id),
  CONSTRAINT patient_habits_supplement_id_fkey FOREIGN KEY (supplement_id) REFERENCES public.patient_supplements(id),
  CONSTRAINT patient_habits_usersUser_id_fkey FOREIGN KEY (usersUser_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.patient_medical_records (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  conditions jsonb NOT NULL DEFAULT '[]'::jsonb,
  medications jsonb NOT NULL DEFAULT '[]'::jsonb,
  history jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  supplement_id uuid,
  CONSTRAINT patient_medical_records_pkey PRIMARY KEY (id),
  CONSTRAINT patient_medical_records_supplement_id_fkey FOREIGN KEY (supplement_id) REFERENCES public.patient_supplements(id)
);
CREATE TABLE public.patient_supplements (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name text,
  dob date,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  customer_id uuid,
  avatar_url text,
  call_confirmed_until timestamp with time zone,
  CONSTRAINT patient_supplements_pkey PRIMARY KEY (id),
  CONSTRAINT patient_supplements_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.payments (
  payment_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  amount bigint NOT NULL,
  description text,
  vnp_txn_ref character varying,
  status character varying NOT NULL DEFAULT 'pending'::character varying,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  plan_code text,
  vnp_create_date bigint,
  vnp_expire_date bigint,
  vnp_order_info text,
  CONSTRAINT payments_pkey PRIMARY KEY (payment_id),
  CONSTRAINT fk_payments_plan_code FOREIGN KEY (plan_code) REFERENCES public.plans(code),
  CONSTRAINT payments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.plans (
  code text NOT NULL,
  name text NOT NULL,
  price bigint NOT NULL,
  camera_quota integer NOT NULL,
  retention_days integer NOT NULL,
  caregiver_seats integer NOT NULL,
  sites integer NOT NULL DEFAULT 1,
  major_updates_months integer NOT NULL DEFAULT 24,
  created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  storage_size character varying,
  is_recommended boolean NOT NULL DEFAULT false,
  tier integer NOT NULL DEFAULT 1,
  currency character varying NOT NULL DEFAULT 'VND'::character varying,
  CONSTRAINT plans_pkey PRIMARY KEY (code)
);
CREATE TABLE public.rooms (
  room_id uuid NOT NULL DEFAULT gen_random_uuid(),
  room_number character varying NOT NULL,
  room_name character varying NOT NULL,
  room_type USER-DEFINED NOT NULL DEFAULT 'single'::room_type_enum,
  floor_number character varying,
  building character varying,
  description text,
  max_capacity integer DEFAULT 1,
  room_settings jsonb,
  status USER-DEFINED NOT NULL DEFAULT 'available'::room_status_enum,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT rooms_pkey PRIMARY KEY (room_id)
);
CREATE TABLE public.shared_permissions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  customer_id uuid NOT NULL,
  caregiver_id uuid NOT NULL,
  stream_view boolean NOT NULL DEFAULT false,
  alert_read boolean NOT NULL DEFAULT false,
  alert_ack boolean NOT NULL DEFAULT false,
  profile_view boolean NOT NULL DEFAULT false,
  log_access_days integer DEFAULT 0,
  report_access_days integer DEFAULT 0,
  notification_channel jsonb DEFAULT '[]'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT shared_permissions_pkey PRIMARY KEY (id),
  CONSTRAINT shared_permissions_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.users(user_id),
  CONSTRAINT shared_permissions_caregiver_id_fkey FOREIGN KEY (caregiver_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.snapshot_images (
  image_id uuid NOT NULL DEFAULT gen_random_uuid(),
  snapshot_id uuid NOT NULL,
  image_path text,
  cloud_url text,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT snapshot_images_pkey PRIMARY KEY (image_id),
  CONSTRAINT fk_snapshot_images_snapshot FOREIGN KEY (snapshot_id) REFERENCES public.snapshots(snapshot_id)
);
CREATE TABLE public.snapshots (
  snapshot_id uuid NOT NULL DEFAULT gen_random_uuid(),
  camera_id uuid NOT NULL,
  room_id uuid NOT NULL,
  user_id uuid,
  image_path character varying,
  cloud_url character varying,
  metadata jsonb,
  capture_type USER-DEFINED NOT NULL DEFAULT 'scheduled'::capture_type_enum,
  captured_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  processed_at timestamp with time zone,
  is_processed boolean NOT NULL DEFAULT false,
  CONSTRAINT snapshots_pkey PRIMARY KEY (snapshot_id),
  CONSTRAINT snapshots_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT snapshots_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id),
  CONSTRAINT snapshots_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id)
);
CREATE TABLE public.subscription_events (
  id bigint NOT NULL DEFAULT nextval('subscription_events_id_seq'::regclass),
  subscription_id uuid NOT NULL,
  event_type USER-DEFINED NOT NULL,
  event_data jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT subscription_events_pkey PRIMARY KEY (id),
  CONSTRAINT subscription_events_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES public.subscriptions(subscription_id)
);
CREATE TABLE public.subscriptions (
  subscription_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  plan_code text NOT NULL,
  status USER-DEFINED NOT NULL DEFAULT 'active'::subscription_status_enum,
  billing_period USER-DEFINED NOT NULL DEFAULT 'none'::billing_period_enum,
  started_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  current_period_start timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  current_period_end timestamp with time zone,
  trial_end_at timestamp with time zone,
  canceled_at timestamp with time zone,
  ended_at timestamp with time zone,
  auto_renew boolean NOT NULL DEFAULT false,
  extra_camera_quota integer NOT NULL DEFAULT 0,
  extra_caregiver_seats integer NOT NULL DEFAULT 0,
  extra_sites integer NOT NULL DEFAULT 0,
  extra_storage_gb integer NOT NULL DEFAULT 0,
  notes text,
  last_payment_at timestamp with time zone,
  CONSTRAINT subscriptions_pkey PRIMARY KEY (subscription_id),
  CONSTRAINT subscriptions_plan_code_fkey FOREIGN KEY (plan_code) REFERENCES public.plans(code),
  CONSTRAINT subscriptions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.system_settings (
  setting_id uuid NOT NULL DEFAULT gen_random_uuid(),
  setting_key character varying NOT NULL,
  setting_value text NOT NULL,
  description text,
  data_type USER-DEFINED NOT NULL DEFAULT 'string'::data_type_enum,
  category character varying DEFAULT 'general'::character varying,
  is_encrypted boolean DEFAULT false,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_by uuid NOT NULL,
  CONSTRAINT system_settings_pkey PRIMARY KEY (setting_id),
  CONSTRAINT system_settings_updated_by_fkey FOREIGN KEY (updated_by) REFERENCES public.users(user_id)
);
CREATE TABLE public.thread_memory (
  thread_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  conversation_history jsonb,
  context_cache jsonb,
  last_updated timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at timestamp with time zone,
  is_active boolean NOT NULL DEFAULT true,
  CONSTRAINT thread_memory_pkey PRIMARY KEY (thread_id),
  CONSTRAINT thread_memory_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.transactions (
  tx_id uuid NOT NULL DEFAULT gen_random_uuid(),
  agreement_id uuid NOT NULL,
  plan_code text NOT NULL,
  plan_snapshot jsonb NOT NULL,
  amount_subtotal bigint NOT NULL,
  amount_discount bigint NOT NULL DEFAULT 0,
  amount_tax bigint NOT NULL DEFAULT 0,
  amount_total bigint NOT NULL,
  currency character varying NOT NULL DEFAULT 'VND'::character varying,
  period_start timestamp with time zone NOT NULL,
  period_end timestamp with time zone NOT NULL,
  status USER-DEFINED NOT NULL,
  effective_action USER-DEFINED NOT NULL,
  provider USER-DEFINED NOT NULL,
  provider_payment_id character varying,
  idempotency_key character varying,
  related_tx_id uuid,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL,
  is_proration boolean NOT NULL DEFAULT false,
  payment_id uuid,
  plan_snapshot_new jsonb,
  plan_snapshot_old jsonb,
  proration_charge bigint NOT NULL DEFAULT 0,
  proration_credit bigint NOT NULL DEFAULT 0,
  CONSTRAINT transactions_pkey PRIMARY KEY (tx_id),
  CONSTRAINT transactions_agreement_id_fkey FOREIGN KEY (agreement_id) REFERENCES public.subscriptions(subscription_id),
  CONSTRAINT transactions_plan_code_fkey FOREIGN KEY (plan_code) REFERENCES public.plans(code),
  CONSTRAINT transactions_related_tx_id_fkey FOREIGN KEY (related_tx_id) REFERENCES public.transactions(tx_id),
  CONSTRAINT transactions_payment_id_fkey FOREIGN KEY (payment_id) REFERENCES public.payments(payment_id)
);
CREATE TABLE public.user_roles (
  user_id uuid NOT NULL,
  role USER-DEFINED NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT user_roles_pkey PRIMARY KEY (user_id, role),
  CONSTRAINT user_roles_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.user_room_assignments (
  assignment_id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  room_id uuid NOT NULL,
  bed_number character varying,
  assigned_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  unassigned_at timestamp with time zone,
  is_active boolean NOT NULL DEFAULT true,
  assigned_by uuid NOT NULL,
  assignment_notes text,
  CONSTRAINT user_room_assignments_pkey PRIMARY KEY (assignment_id),
  CONSTRAINT user_room_assignments_assigned_by_fkey FOREIGN KEY (assigned_by) REFERENCES public.users(user_id),
  CONSTRAINT user_room_assignments_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id),
  CONSTRAINT user_room_assignments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.users (
  user_id uuid NOT NULL DEFAULT gen_random_uuid(),
  username character varying NOT NULL,
  email character varying NOT NULL,
  password_hash character varying NOT NULL,
  full_name character varying NOT NULL,
  role USER-DEFINED NOT NULL DEFAULT 'customer'::role_enum,
  date_of_birth date,
  gender USER-DEFINED,
  phone_number character varying,
  emergency_contact character varying,
  medical_conditions text,
  mobility_limitations text,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  otp_code text,
  otp_expires_at timestamp with time zone,
  consent_at timestamp with time zone,
  notification_preferencesId uuid,
  CONSTRAINT users_pkey PRIMARY KEY (user_id),
  CONSTRAINT users_notification_preferencesId_fkey FOREIGN KEY (notification_preferencesId) REFERENCES public.notification_preferences(id)
);