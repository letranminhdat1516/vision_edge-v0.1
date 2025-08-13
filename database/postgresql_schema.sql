-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

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
  room_id uuid NOT NULL,
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
  CONSTRAINT cameras_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id)
);
CREATE TABLE public.caregiver_patient_assignments (
  assignment_id uuid NOT NULL DEFAULT gen_random_uuid(),
  caregiver_id uuid NOT NULL,
  patient_id uuid NOT NULL,
  assigned_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  unassigned_at timestamp with time zone,
  is_active boolean NOT NULL DEFAULT true,
  assigned_by uuid,
  assignment_notes text,
  CONSTRAINT caregiver_patient_assignments_pkey PRIMARY KEY (assignment_id)
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
  user_id uuid NOT NULL,
  name text NOT NULL,
  relation text NOT NULL,
  phone text NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT emergency_contacts_pkey PRIMARY KEY (id),
  CONSTRAINT emergency_contacts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.event_detections (
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
  status USER-DEFINED NOT NULL DEFAULT 'detected'::event_status_enum,
  context_data jsonb,
  detected_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  verified_at timestamp with time zone,
  verified_by uuid,
  acknowledged_at timestamp with time zone,
  acknowledged_by uuid,
  dismissed_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT event_detections_pkey PRIMARY KEY (event_id),
  CONSTRAINT event_detections_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id),
  CONSTRAINT event_detections_verified_by_fkey FOREIGN KEY (verified_by) REFERENCES public.users(user_id),
  CONSTRAINT event_detections_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id),
  CONSTRAINT event_detections_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id),
  CONSTRAINT event_detections_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.snapshots(snapshot_id)
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
  user_id uuid NOT NULL,
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
  CONSTRAINT patient_habits_pkey PRIMARY KEY (habit_id),
  CONSTRAINT patient_habits_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.patient_medical_records (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  conditions jsonb NOT NULL DEFAULT '[]'::jsonb,
  medications jsonb NOT NULL DEFAULT '[]'::jsonb,
  history jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT patient_medical_records_pkey PRIMARY KEY (id),
  CONSTRAINT patient_medical_records_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
);
CREATE TABLE public.patient_supplements (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name text,
  dob date,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT patient_supplements_pkey PRIMARY KEY (id),
  CONSTRAINT patient_supplements_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
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
CREATE TABLE public.snapshots (
  snapshot_id uuid NOT NULL DEFAULT gen_random_uuid(),
  camera_id uuid NOT NULL,
  room_id uuid NOT NULL,
  user_id uuid,
  image_path character varying NOT NULL,
  cloud_url character varying,
  metadata jsonb,
  capture_type USER-DEFINED NOT NULL DEFAULT 'scheduled'::capture_type_enum,
  captured_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  processed_at timestamp with time zone,
  is_processed boolean NOT NULL DEFAULT false,
  CONSTRAINT snapshots_pkey PRIMARY KEY (snapshot_id),
  CONSTRAINT snapshots_camera_id_fkey FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id),
  CONSTRAINT snapshots_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(room_id),
  CONSTRAINT snapshots_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
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
CREATE TABLE public.user_settings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  key character varying NOT NULL,
  value text NOT NULL,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_by uuid,
  CONSTRAINT user_settings_pkey PRIMARY KEY (id),
  CONSTRAINT user_settings_updated_by_fkey FOREIGN KEY (updated_by) REFERENCES public.users(user_id),
  CONSTRAINT user_settings_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(user_id)
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
  otp_expires_at date,
  CONSTRAINT users_pkey PRIMARY KEY (user_id)
);