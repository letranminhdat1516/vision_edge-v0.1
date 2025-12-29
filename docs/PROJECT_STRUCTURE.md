# Project Structure

This file lists the current repository layout (folders and key files). Generated from the working tree on 2025-12-29.

```text
vision_edge-v0.1/
├─ .dockerignore
├─ .env
├─ .git/
├─ .gitattributes
├─ .gitignore
├─ Dockerfile
├─ Dockerfile.pi
├─ docker-compose.yml
├─ deploy_pi.sh
├─ README.md
├─ requirements.txt
├─ yolov8n-pose.pt
├─ yolov8s.pt
├─ docs/
│  ├─ ALARM_ACTIVATION_FLOW.md
│  ├─ ALARM_API_GUIDE.md
│  ├─ ALARM_STOP_CONDITIONS.md
│  ├─ ALARM_STOP_METHODS.md
│  ├─ ALARM_TRIGGER_FLOW.md
│  ├─ DropIn_System_Architecture.md
│  ├─ EVENT_STATUS_LEVELS.md
│  ├─ FALL_THRESHOLDS_SUMMARY.md
│  ├─ LIFECYCLE_STATE_FLOW.md
│  ├─ NORMAL_OPTIMIZATION.md
│  ├─ Project_Documentation.md
│  ├─ SEIZURE_THRESHOLDS_SUMMARY.md
│  ├─ reliability_score_calculation.md
│  └─ test/
│     ├─ README.md
│     ├─ analyze_seizure_frames.py
│     ├─ debug_compare_fall.py
│     ├─ debug_fall_production.py
│     ├─ debug_keypoint_jitter.py
│     ├─ video_camera_service.py
│     ├─ yolov8m.pt
│     ├─ yolov8n.pt
│     ├─ yolov8n-pose.pt
│     ├─ yolov8s.pt
│     ├─ resource/
│     │  └─ .gitkeep
│     └─ test_results/
│        └─ video_analysis/
│           ├─ 1/
│           ├─ 2/
│           ├─ 3/
│           ├─ 4/
│           ├─ 5/
│           ├─ 6/
│           ├─ 7/
│           ├─ 8/
│           ├─ 9/
│           ├─ 10/
│           ├─ 11/
│           ├─ 12/
│           ├─ 13/
│           ├─ 14/
│           ├─ 15/
│           ├─ 16/
│           ├─ 17/
│           ├─ 18/
│           ├─ 19/
│           ├─ 20/
│           ├─ 21/
│           ├─ 22/
│           ├─ 23/
│           ├─ 25/
│           ├─ 26/
│           ├─ 27/
│           ├─ 28/
│           ├─ 29/
│           ├─ 30/
│           ├─ 31/
│           ├─ 32/
│           ├─ 33/
│           ├─ 34/
│           └─ 35/
├─ examples/
│  ├─ advanced_healthcare_monitor.py
│  ├─ check_database_records.py
│  ├─ check_foreign_keys.py
│  ├─ check_lifecycle_enum.py
│  ├─ check_normal_snapshots.py
│  ├─ check_schema.py
│  ├─ check_sensitivity.py
│  ├─ check_snapshots_table.py
│  ├─ check_trigger_function.py
│  ├─ check_triggers.py
│  ├─ debug_model_structure.py
│  ├─ debug_seizure.py
│  ├─ demo_vietnamese_captioning.py
│  ├─ external_vision_api.py
│  ├─ fcm_demo.py
│  ├─ full_vietnamese_model.py
│  ├─ healthcare_api_server.py
│  ├─ healthcare_realtime_api.py
│  ├─ healthcare_realtime_client.py
│  ├─ healthcare_realtime_demo.py
│  ├─ healthcare_realtime_test.py
│  ├─ healthcare_system_test.py
│  ├─ healthcare_websocket_production.py
│  ├─ local_vision_models.py
│  ├─ mobile_healthcare_app.py
│  ├─ multi_camera_healthcare_system.py
│  ├─ quick_alarm_test.py
│  ├─ quick_audio_test.py
│  ├─ quick_blip_test.py
│  ├─ quick_test_alarm_api.py
│  ├─ same_room_dual_detection_main.py
│  ├─ simple_http_server.py
│  ├─ supabase_realtime_demo.py
│  ├─ trigger_alarm_test.py
│  ├─ yolov8n-pose.pt
│  ├─ yolov8s.pt
│  ├─ test/
│  │  ├─ test_alarm_auto_stop.py
│  │  ├─ test_alarm_control_api.py
│  │  ├─ test_alarm_stop.py
│  │  ├─ test_alarm_system.py
│  │  ├─ test_all_audio_backends.py
│  │  ├─ test_audio_device.py
│  │  ├─ test_auto_alarm_30s.py
│  │  ├─ test_auto_called_3min.py
│  │  ├─ test_bending_caption.py
│  │  ├─ test_bluetooth_audio.py
│  │  ├─ test_bluetooth_final.py
│  │  ├─ test_bug_scenarios.py
│  │  ├─ test_dual_detection.py
│  │  ├─ test_event_response_format.py
│  │  ├─ test_fcm_notification.py
│  │  ├─ test_full_alarm_flow.py
│  │  ├─ test_healthcare_publisher.py
│  │  ├─ test_json_response.py
│  │  ├─ test_keypoint_improvements.py
│  │  ├─ test_manual_stop_resolved.py
│  │  ├─ test_mobile_realtime_system.py
│  │  ├─ test_model_loading.py
│  │  ├─ test_model_with_weights.py
│  │  ├─ test_multi_camera_connection.py
│  │  ├─ test_notification_direct.py
│  │  ├─ test_professional_pipeline.py
│  │  ├─ test_realtime_fcm.py
│  │  ├─ test_real_captioning.py
│  │  ├─ test_snapshot_creation.py
│  │  ├─ test_supabase_connection.py
│  │  ├─ test_supabase_connectivity.py
│  │  ├─ test_supabase_insert.py
│  │  ├─ test_supabase_js_api.py
│  │  ├─ test_supabase_realtime_integration.py
│  │  ├─ test_update_alarm_state.py
│  │  └─ test_vietnamese_caption.py
│  └─ test_results/
│     ├─ alerts/
│     ├─ keypoints/
│     ├─ logs/
│     ├─ reports/
│     └─ statistics/
├─ models/
│  ├─ pose_models/
│  │  └─ openpose_pose_coco.prototxt
│  └─ VSViG/
│     ├─ LICENSE
│     ├─ README.md
│     ├─ VSViG.py
│     ├─ VSViG-base.pth
│     ├─ dy_point_order.pt
│     ├─ extract_patches.py
│     ├─ pose.pth
│     ├─ train.py
│     └─ __pycache__/
├─ src/
│  ├─ __init__.py
│  ├─ alarm_fastapi_server.py
│  ├─ health_check.py
│  ├─ main.py
│  ├─ yolov8n-pose.pt
│  ├─ yolov8s.pt
│  ├─ camera/
│  │  ├─ __init__.py
│  │  ├─ config.py
│  │  ├─ simple_camera.py
│  │  └─ __pycache__/
│  ├─ config/
│  │  ├─ supabase_config.py
│  │  └─ __pycache__/
│  ├─ data/
│  │  └─ saved_frames/
│  │     ├─ alerts/
│  │     ├─ detections/
│  │     └─ keyframes/
│  ├─ examples/
│  │  └─ data/
│  │     └─ saved_frames/
│  │        └─ alerts/
│  ├─ fall_detection/
│  │  ├─ __init__.py
│  │  ├─ fall_prediction.py
│  │  ├─ simple_fall_detector.py
│  │  ├─ __pycache__/
│  │  ├─ ai_models/
│  │  │  ├─ lite-model_movenet_singlepose_thunder_3.tflite
│  │  │  ├─ posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite
│  │  │  ├─ posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
│  │  │  ├─ pose_labels.txt
│  │  │  └─ tflite-model-maker-falldetect-model.tflite
│  │  └─ pipeline/
│  │     ├─ __init__.py
│  │     ├─ fall_detect.py
│  │     ├─ inference.py
│  │     ├─ movenet_model.py
│  │     ├─ pose_base.py
│  │     └─ posenet_model.py
│  ├─ infrastructure/
│  │  ├─ services/
│  │  │  ├─ alarm_api.py
│  │  │  ├─ audio_alert_service.py
│  │  │  ├─ emergency_alarm_handler_psycopg.py
│  │  │  ├─ event_lifecycle_worker.py
│  │  │  ├─ snapshot_service.py
│  │  │  └─ __pycache__/
│  │  └─ storage/
│  │     ├─ minio_service.py
│  │     └─ __pycache__/
│  ├─ models/
│  │  ├─ generated/
│  │  │  ├─ _prisma_migrations.py
│  │  │  ├─ access_grants.py
│  │  │  ├─ activity_logs.py
│  │  │  ├─ cameras.py
│  │  │  ├─ caregiver_invitations.py
│  │  │  ├─ email_templates.py
│  │  │  ├─ emergency_contacts.py
│  │  │  ├─ event_detections.py
│  │  │  ├─ event_history.py
│  │  │  ├─ fcm_tokens.py
│  │  │  ├─ models.py
│  │  │  ├─ notifications.py
│  │  │  ├─ patient_habits.py
│  │  │  ├─ patient_medical_records.py
│  │  │  ├─ patient_sleep_checkins.py
│  │  │  ├─ patient_supplements.py
│  │  │  ├─ payments.py
│  │  │  ├─ permissions.py
│  │  │  ├─ plans.py
│  │  │  ├─ role_permissions.py
│  │  │  ├─ roles.py
│  │  │  ├─ shared_permissions.py
│  │  │  ├─ snapshot_images.py
│  │  │  ├─ snapshots.py
│  │  │  ├─ subscription_events.py
│  │  │  ├─ subscriptions.py
│  │  │  ├─ suggestions.py
│  │  │  ├─ system_config.py
│  │  │  ├─ system_settings.py
│  │  │  ├─ ticket.py
│  │  │  ├─ ticket_history.py
│  │  │  ├─ transactions.py
│  │  │  ├─ uploads.py
│  │  │  ├─ user_preferences.py
│  │  │  ├─ user_roles.py
│  │  │  ├─ user_settings.py
│  │  │  ├─ users.py
│  │  │  └─ __pycache__/
│  │  └─ pose_models/
│  ├─ seizure_detection/
│  │  ├─ __init__.py
│  │  ├─ enhanced_pose_estimator.py
│  │  ├─ mediapipe_pose.py
│  │  ├─ model_loader.py
│  │  ├─ pose_estimator.py
│  │  ├─ seizure_predictor.py
│  │  ├─ ultimate_pose_estimator.py
│  │  ├─ vsvig_detector.py
│  │  └─ yolov8_pose_estimator.py
│  ├─ service/
│  │  ├─ __pycache__/
│  │  ├─ advanced_healthcare_pipeline.py
│  │  ├─ ai_vision_description_service.py
│  │  ├─ camera_config_service.py
│  │  ├─ camera_network_coordinator.py
│  │  ├─ camera_service.py
│  │  ├─ clean_camera_service.py
│  │  ├─ database_camera_config.py
│  │  ├─ database_config_service.py
│  │  ├─ database_mock_adapter.py
│  │  ├─ dual_camera_surveillance_system.py
│  │  ├─ emergency_notification_dispatcher.py
│  │  ├─ enhanced_multi_camera_system.py
│  │  ├─ fall_detection_service.py
│  │  ├─ postgresql_healthcare_service.py
│  │  ├─ seizure_detection_service.py
│  │  ├─ simple_camera_db.py
│  │  ├─ supabase_realtime_service.py
│  │  └─ video_processing_service.py
│  ├─ sounds/
│  │  ├─ emergency_siren.mp3
│  │  └─ emergency_siren.wav
│  └─ video_processing/
│     ├─ __init__.py
│     └─ simple_processing.py
└─ test_results/
   └─ tuning/
```

Notes
- This tree focuses on directories and key files discovered during the scan. Some large result folders (e.g., numeric buckets under video analysis) are listed at folder level for brevity.
