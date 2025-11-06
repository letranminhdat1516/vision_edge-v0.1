
#!/usr/bin/env python3
"""
Vision Edge Healthcare System - Main Application
Supports both single and dual camera configurations
"""

import cv2
import time
import json
import sys
import uuid
import os
import logging
from pathlib import Path
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline

# Setup logging to see handler notifications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import intelligent action generation
try:
    from service.ai_vision_description_service import get_professional_caption_pipeline
    INTELLIGENT_ACTIONS_AVAILABLE = True
    print("🤖 Intelligent Action Generation: AVAILABLE")
except ImportError:
    INTELLIGENT_ACTIONS_AVAILABLE = False
    print("📝 Intelligent Action Generation: Using static messages")

def load_camera_config():
    """Load camera configuration from config.json"""
    config_path = Path("src/config/config.json")
    if not config_path.exists():
        config_path = Path("config/config.json")
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('database', {}).get('cameras', {})
    return {}

def validate_camera_credentials(camera_info):
    """Validate and potentially fix camera credentials"""
    rtsp_url = camera_info.get('rtsp_url', '')
    
    # Check for common credential issues
    if '401' in rtsp_url or 'Unauthorized' in rtsp_url:
        print(f"   ⚠️ Authentication issue detected in URL")
        return False
    
    # Parse RTSP URL for credentials
    if '@' not in rtsp_url:
        print(f"   ⚠️ No credentials found in RTSP URL")
        return False
    
    try:
        # Extract credentials and IP
        protocol_part, rest = rtsp_url.split('://', 1)
        creds_part, ip_part = rest.split('@', 1)
        username, password = creds_part.split(':', 1)
        ip = ip_part.split(':')[0]
        
        print(f"   🔐 Credentials: {username}:{'*' * len(password)}")
        print(f"   🌐 Camera IP: {ip}")
        
        # Suggest alternative credentials if current ones fail
        if username == "admin" and password in ["123456", "password"]:
            print(f"   💡 Trying common alternative credentials...")
            alternative_passwords = ["L2C37340", "admin", "12345", ""]
            
            for alt_password in alternative_passwords:
                if alt_password != password:
                    alt_url = rtsp_url.replace(f":{password}@", f":{alt_password}@")
                    print(f"   🧪 Alternative URL: {alt_url}")
                    
                    # Test quick connection
                    cap = cv2.VideoCapture(alt_url)
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            print(f"   ✅ Alternative credentials work! Updating URL...")
                            camera_info['rtsp_url'] = alt_url
                            return True
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error parsing URL: {e}")
        return False

def detect_camera_mode():
    """Detect if we should use single or dual camera mode with validation"""
    cameras_config = load_camera_config()
    
    if not cameras_config:
        print("🎥 No cameras configured - Using FALLBACK single camera mode")
        return 'single', None
    
    # Validate camera credentials first
    print("🔐 Validating camera credentials...")
    valid_cameras = {}
    
    for camera_key, camera_info in cameras_config.items():
        camera_name = camera_info.get('name', camera_key)
        print(f"   📹 Checking {camera_name}...")
        
        if validate_camera_credentials(camera_info):
            valid_cameras[camera_key] = camera_info
            print(f"   ✅ {camera_name} credentials valid")
        else:
            print(f"   ❌ {camera_name} has credential issues")
    
    if not valid_cameras:
        print("❌ No cameras have valid credentials - Using FALLBACK mode")
        return 'single', None
    
    if len(valid_cameras) >= 2:
        # Check if cameras are in same room
        rooms = set()
        for camera in valid_cameras.values():
            room_id = camera.get('room_id', 'unknown')
            rooms.add(room_id)
        
        if len(rooms) == 1:
            print(f"🎥🎥 Detected {len(valid_cameras)} cameras in same room - Using DUAL DETECTION mode")
            return 'dual', valid_cameras
        else:
            print(f"🎥 Detected {len(valid_cameras)} cameras in different rooms - Using SINGLE camera mode")
            return 'single', list(valid_cameras.values())[0]
    elif len(valid_cameras) == 1:
        print("🎥 Detected 1 valid camera - Using SINGLE camera mode")
        return 'single', list(valid_cameras.values())[0]
    else:
        print("🎥 No valid cameras found - Using FALLBACK single camera mode")
        return 'single', None

print("="*60)
print("🏥 Vision Edge Healthcare System v0.1")
print("🔍 Loading camera configuration from database...")
print("="*60)

if __name__ == "__main__":
    # Load user ID from environment
    import os
    user_id = os.getenv('DEFAULT_USER_ID')
    if not user_id:
        print("❌ DEFAULT_USER_ID not found in .env file")
        exit(1)
    
    # Load cameras from database and determine mode
    from service.clean_camera_service import camera_service
    
    print("📹 Loading camera configuration from database...")
    all_cameras = camera_service.get_cameras_for_user(user_id)
    
    if not all_cameras:
        print("❌ No cameras found for this user")
        print("💡 Please check database or user ID")
        exit(1)
    
    print(f"✅ Found {len(all_cameras)} cameras for user {user_id}")
    
    # Always use simple multi-camera approach - based on working single camera code
    if len(all_cameras) >= 2:
        # MULTI-CAMERA MODE - Using proven single camera logic
        locations = set([cam['location'] for cam in all_cameras])
        if len(locations) == 1:
            print(f"🎥🎥 MULTI-CAMERA MODE: {len(all_cameras)} cameras in same room ({list(locations)[0]})")
        else:
            print(f"🎥🎥 MULTI-CAMERA MODE: {len(all_cameras)} cameras in different rooms")
        
        # Process each camera individually using working single camera code
        print("🎥 Starting Multi-Camera Healthcare Monitoring System...")
        print("📱 Emergency notifications: ACTIVE")
        print("🏥 Real-time healthcare detection: ACTIVE")
        if INTELLIGENT_ACTIONS_AVAILABLE:
            print("🤖 Intelligent action generation: ACTIVE")
        else:
            print("📝 Static action messages: ACTIVE")
        print("="*60)
        print("🎮 CONTROLS:")
        print("   'q' = Quit system")  
        print("   's' = Show detailed statistics")
        print("")
        print("🤸 SEIZURE DETECTION TIPS (SIÊU NHẠY CẢM):")
        print("   🔥 Vẫy tay nhanh liên tục 3-5 giây")
        print("   🔥 Nhảy nhót, chuyển động bất thường")
        print("   🔥 Gật đầu mạnh nhiều lần")
        print("   🔥 Threshold siêu thấp: chỉ cần 0.01 confidence!")
        print("="*60)
        
        # Main processing loop for all cameras - Each camera with its own services
        cameras_data = []
        
        for i, cam in enumerate(all_cameras):
            print(f"🔧 Setting up processing for Camera {i+1}: {cam['name']}")
            
            # Parse resolution from database
            resolution_str = cam.get('resolution', '1920x1080')
            try:
                width, height = map(int, resolution_str.split('x'))
                resolution = (width, height)
            except:
                resolution = (1920, 1080)
            
            # Configure camera from database data (same as single mode)
            camera_config = {
                'url': cam['rtsp_url'],
                'buffer_size': 1,
                'fps': cam.get('fps', 30),
                'resolution': resolution,
                'auto_reconnect': True,
                'camera_id': cam['id'],
                'camera_name': cam['name']
            }
            
            processor_config = 120
            alerts_folder = "examples/data/saved_frames/alerts"
            
            # Initialize services for THIS camera
            from service.camera_service import CameraService
            from service.video_processing_service import VideoProcessingService
            from service.fall_detection_service import FallDetectionService
            from service.seizure_detection_service import SeizureDetectionService
            from seizure_detection.seizure_predictor import SeizurePredictor
            
            individual_camera = CameraService(camera_config)
            individual_camera.connect()
            individual_video_processor = VideoProcessingService(processor_config)
            individual_fall_detector = FallDetectionService()
            individual_seizure_detector = SeizureDetectionService()
            individual_seizure_predictor = SeizurePredictor(temporal_window=3, alert_threshold=0.01, warning_threshold=0.005)  # Siêu nhạy cảm!
            
            # Initialize individual Healthcare Pipeline
            print(f"🔧 Initializing Healthcare Pipeline with camera_id: {cam['id']}, user_id: {user_id}")
            individual_pipeline = AdvancedHealthcarePipeline(
                camera=individual_camera,
                video_processor=individual_video_processor,
                fall_detector=individual_fall_detector,
                seizure_detector=individual_seizure_detector,
                seizure_predictor=individual_seizure_predictor,
                alerts_folder=alerts_folder,
                camera_id=cam['id'],  # Use real camera ID from database
                user_id=user_id  # Use real user ID
            )
            
            # Store camera data with individual instances
            cameras_data.append({
                'camera': individual_camera,
                'pipeline': individual_pipeline,
                'name': cam['name'],
                'id': cam['id']
            })
            
            print(f"✅ Camera {i+1} ({cam['name']}) processing setup complete!")
        
        print(f"🎥 All {len(cameras_data)} cameras ready for processing!")
        
        # 🔊 Initialize Emergency Alarm Handler (PostgreSQL LISTEN/NOTIFY - psycopg3)
        print("\n🔊 Initializing Emergency Alarm System (REALTIME MODE)...")
        print("   - Bluetooth/USB speaker detection")
        print("   - Mobile app emergency trigger")
        print("   - PostgreSQL LISTEN/NOTIFY (Direct Connection - port 5432)")
        print("   - Automatic audio alert playback")
        from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
        from infrastructure.services.audio_alert_service import audio_alert_service
        
        # Connect PostgreSQL service to alarm handler (use first camera's pipeline)
        emergency_alarm_handler.set_postgresql_service(cameras_data[0]['pipeline'].event_publisher.postgresql_service)
        
        # Check audio device status
        audio_status = audio_alert_service.get_status()
        if audio_status['enabled']:
            print(f"   ✅ Audio system ready!")
            print(f"      Backend: {audio_status['audio_backend']}")
            print(f"      Devices detected: {audio_status['available_devices']}")
            if audio_status['devices']:
                print(f"      Device list:")
                for device in audio_status['devices'][:3]:  # Show max 3 devices
                    print(f"         - {device}")
        else:
            print(f"   ⚠️ Audio system disabled - check logs for details")
        
        # Start emergency alarm handler in background (LISTEN/NOTIFY)
        import asyncio
        import threading
        import logging
        
        # Enable logging for handler
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        def run_emergency_handler():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(emergency_alarm_handler.start_listening())
        
        alarm_thread = threading.Thread(target=run_emergency_handler, daemon=True)
        alarm_thread.start()
        
        # Wait for handler to connect
        print("   ⏳ Waiting for handler to connect...")
        time.sleep(3)
        
        print("   ✅ Emergency alarm handler started (PostgreSQL LISTEN/NOTIFY)!")
        print("   📡 Channel: 'system_alarm_channel'")
        print("   🔌 Connection: Direct (port 5432)")
        print("   📱 Ready for mobile app triggers (< 50ms response)")
        print("\n" + "=" * 80)
        print("💡 Handler logs will appear above when alarm is triggered")
        print("=" * 80 + "\n")
        
        while True:
            for cam_data in cameras_data:
                try:
                    frame = cam_data['camera'].get_frame()
                    if frame is None:
                        continue
                    
                    result = cam_data['pipeline'].process_frame(frame)
                    detection_result = result["detection_result"]
                    person_detections = result["person_detections"]
                    
                    # Debug logging removed for cleaner output
                    
                    # Generate intelligent action when alert detected
                    if detection_result.get('alert_level') in ['critical', 'high']:
                        emergency_type = detection_result.get('emergency_type', 'unknown')
                        confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
                        print(f"🚨 EMERGENCY ALERT in {cam_data['name']}: {emergency_type.upper()} detected (confidence: {confidence:.2f})")
                        
                        # 💾 SAVE EVENT TO DATABASE
                        try:
                            print(f"💾 Saving event to database for {cam_data['name']}...")
                            
                            # Determine event_type and status
                            event_type = "abnormal_behavior" if "seizure" in emergency_type else "fall"
                            status = "danger" if detection_result.get('alert_level') == 'critical' else "warning"
                            
                            # Get bounding boxes from detection result
                            bounding_boxes = detection_result.get('bounding_boxes', [])
                            if not bounding_boxes and person_detections:
                                # Fallback to person detections
                                bounding_boxes = [{
                                    'x': int(p.get('bbox', [0, 0, 0, 0])[0]),
                                    'y': int(p.get('bbox', [0, 0, 0, 0])[1]),
                                    'width': int(p.get('bbox', [0, 0, 0, 0])[2] - p.get('bbox', [0, 0, 0, 0])[0]),
                                    'height': int(p.get('bbox', [0, 0, 0, 0])[3] - p.get('bbox', [0, 0, 0, 0])[1]),
                                    'confidence': float(p.get('confidence', confidence)),
                                    'class': 'person'
                                } for p in person_detections[:3]]  # Max 3 persons
                            
                            # Publish event using pipeline's event publisher
                            if event_type == 'fall':
                                alert_result = cam_data['pipeline'].event_publisher.publish_fall_detection(
                                    confidence=confidence,
                                    bounding_boxes=bounding_boxes,
                                    context={
                                        'description': f"{emergency_type.upper()} detected in {cam_data['name']}",
                                        'alert_level': detection_result.get('alert_level'),
                                        'camera_name': cam_data['name']
                                    }
                                )
                            else:  # abnormal_behavior (seizure)
                                alert_result = cam_data['pipeline'].event_publisher.publish_seizure_detection(
                                    confidence=confidence,
                                    bounding_boxes=bounding_boxes,
                                    context={
                                        'description': f"{emergency_type.upper()} detected in {cam_data['name']}",
                                        'alert_level': detection_result.get('alert_level'),
                                        'camera_name': cam_data['name']
                                    }
                                )
                            
                            if alert_result and alert_result.get('success'):
                                print(f"   ✅ Event saved to database!")
                                print(f"   🆔 Event ID: {alert_result.get('event_id', 'N/A')}")
                                print(f"   📊 Status: {status}")
                                print(f"   📹 Camera: {cam_data['name']}")
                            else:
                                print(f"   ⚠️ Event save failed: {alert_result.get('error', 'Unknown error') if alert_result else 'No response'}")
                                
                        except Exception as db_error:
                            print(f"   ❌ Database save error: {db_error}")
                            import traceback
                            traceback.print_exc()
                    
                    # Debug warning level alerts too
                    elif detection_result.get('alert_level') == 'warning':
                        emergency_type = detection_result.get('emergency_type', 'unknown')
                        confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
                        print(f"⚠️ WARNING ALERT in {cam_data['name']}: {emergency_type.upper()} detected (confidence: {confidence:.2f})")
                    
                    # Display windows for each camera (using unique window names)
                    normal_window_name = f"Camera {cam_data['name']} - Normal View"
                    analysis_window_name = f"Camera {cam_data['name']} - Analysis View"
                    
                    cv2.imshow(normal_window_name, result["normal_window"])
                    
                    # Analysis view with statistics overlay
                    analysis_view = cam_data['pipeline'].visualize_dual_detection(frame, detection_result, person_detections)
                    analysis_view = cam_data['pipeline'].draw_statistics_overlay(analysis_view, cam_data['pipeline'].stats)
                    
                    cv2.imshow(analysis_window_name, analysis_view)
                    
                except Exception as e:
                    print(f"❌ Error processing {cam_data['name']}: {e}")
                    continue
            
            # Check keyboard input (same as single mode)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Shutting down Multi-Camera Healthcare Monitoring System...")
                break
            elif key == ord('s'):
                # Show statistics for all cameras
                for cam_data in cameras_data:
                    print(f"\n📊 Statistics for {cam_data['name']}:")
                    cam_data['pipeline'].print_final_statistics()
        
        print("📱 Notifications stopped")
        print("🏥 Multi-camera healthcare monitoring stopped")
        cv2.destroyAllWindows()
        exit(0)
        
    else:
        # SINGLE CAMERA MODE - only when 1 camera
        print(f"🎥 SINGLE CAMERA MODE: Only {len(all_cameras)} camera available")
        
        # Use first camera for single mode
        primary_camera = all_cameras[0]
        print(f"✅ Using camera: {primary_camera['name']} ({primary_camera['id']})")
        print(f"📍 Location: {primary_camera['location']}")
        print(f"🔗 RTSP URL: {primary_camera['rtsp_url']}")
        
        # Parse resolution from database (for single camera mode)
        resolution_str = primary_camera.get('resolution', '1920x1080')
        try:
            width, height = map(int, resolution_str.split('x'))
            resolution = (width, height)
        except:
            resolution = (1920, 1080)
        
        # Configure camera from database data
        camera_config = {
            'url': primary_camera['rtsp_url'],
            'buffer_size': 1,
            'fps': primary_camera.get('fps', 30),
            'resolution': resolution,
            'auto_reconnect': True,
            'camera_id': primary_camera['id'],
            'camera_name': primary_camera['name']
        }
        
        processor_config = 120
        alerts_folder = "examples/data/saved_frames/alerts"
        # Khởi tạo các service thật sự
        from service.camera_service import CameraService
        from service.video_processing_service import VideoProcessingService
        from service.fall_detection_service import FallDetectionService
        from service.seizure_detection_service import SeizureDetectionService

        camera = CameraService(camera_config)
        camera.connect()
        video_processor = VideoProcessingService(processor_config)
        fall_detector = FallDetectionService()
        seizure_detector = SeizureDetectionService()
        
        # Import và init seizure predictor
        from seizure_detection.seizure_predictor import SeizurePredictor
        seizure_predictor = SeizurePredictor(temporal_window=20, alert_threshold=0.55, warning_threshold=0.35)  # Giảm threshold để nhạy hơn
        
        # Khởi tạo AdvancedHealthcarePipeline
        print("🏥 Initializing Healthcare Pipeline...")
        print("   - Real-time fall detection")
        print("   - Real-time seizure detection") 
        print("   - Emergency notifications")
        print("   - Supabase realtime integration")
        print("   - Mobile app notifications")
        if INTELLIGENT_ACTIONS_AVAILABLE:
            print("   - 🤖 Intelligent action generation (BLIP + Translation)")
        else:
            print("   - 📝 Static action messages")
        
        pipeline = AdvancedHealthcarePipeline(
            camera=camera, 
            video_processor=video_processor, 
            fall_detector=fall_detector, 
            seizure_detector=seizure_detector, 
            seizure_predictor=seizure_predictor, 
            alerts_folder=alerts_folder
        )
        
        # 🔊 Initialize Emergency Alarm Handler (PostgreSQL LISTEN/NOTIFY - psycopg3)
        print("🔊 Initializing Emergency Alarm System (REALTIME MODE)...")
        print("   - Bluetooth/USB speaker detection")
        print("   - Mobile app emergency trigger")
        print("   - PostgreSQL LISTEN/NOTIFY (Direct Connection - port 5432)")
        print("   - Automatic audio alert playback")
        from infrastructure.services.emergency_alarm_handler_psycopg import emergency_alarm_handler
        from infrastructure.services.audio_alert_service import audio_alert_service
        
        # Connect PostgreSQL service to alarm handler
        emergency_alarm_handler.set_postgresql_service(pipeline.event_publisher.postgresql_service)
        
        # Check audio device status
        audio_status = audio_alert_service.get_status()
        if audio_status['enabled']:
            print(f"   ✅ Audio system ready!")
            print(f"      Backend: {audio_status['audio_backend']}")
            print(f"      Devices detected: {audio_status['available_devices']}")
            if audio_status['devices']:
                print(f"      Device list:")
                for device in audio_status['devices'][:3]:  # Show max 3 devices
                    print(f"         - {device}")
        else:
            print(f"   ⚠️ Audio system disabled - check logs for details")
        
        # Start emergency alarm handler in background (LISTEN/NOTIFY)
        import asyncio
        import threading
        import logging
        
        # Enable logging for handler
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        def run_emergency_handler():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(emergency_alarm_handler.start_listening())
        
        alarm_thread = threading.Thread(target=run_emergency_handler, daemon=True)
        alarm_thread.start()
        
        # Wait for handler to connect
        print("   ⏳ Waiting for handler to connect...")
        time.sleep(3)
        
        print("   ✅ Emergency alarm handler started (PostgreSQL LISTEN/NOTIFY)!")
        print("   📡 Channel: 'system_alarm_channel'")
        print("   🔌 Connection: Direct (port 5432)")
        print("   📱 Ready for mobile app triggers (< 50ms response)")
        print("\n" + "=" * 80)
        print("💡 Handler logs will appear above when alarm is triggered")
        print("=" * 80 + "\n")



        
        # Initialize intelligent action pipeline if available
        caption_pipeline = None
        if INTELLIGENT_ACTIONS_AVAILABLE:
            try:
                caption_pipeline = get_professional_caption_pipeline()
                print(f"   ✅ BLIP model loaded: {caption_pipeline.blip_loaded}")
                print(f"   ✅ Translation model loaded: {caption_pipeline.translator_loaded}")
            except Exception as e:
                print(f"   ⚠️ Intelligent action initialization failed: {e}")
                caption_pipeline = None
        
        print("✅ Healthcare Pipeline initialized!")
        print("📱 Mobile notifications handled by NestJS backend")
        print("="*60)

        print("🎥 Starting Healthcare Monitoring System...")
        print("📱 Emergency notifications: ACTIVE")
        print("🏥 Real-time healthcare detection: ACTIVE")
        if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline:
            print("🤖 Intelligent action generation: ACTIVE")
        else:
            print("📝 Static action messages: ACTIVE")
    print("🎮 CONTROLS:")
    print("   'q' = Quit system")
    print("   's' = Show statistics")
    print("   'i' = Intelligent action info")
    print("   'e' = Create random test event")
    print("   'a' = Show emergency alarm status")  # NEW
    print("   't' = Test emergency alarm manually")  # NEW
    print("="*60)

    # Real-time processing variables
    last_alert_image_path = None
    frame_count = 0

    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        
        frame_count += 1
        result = pipeline.process_frame(frame)
        detection_result = result["detection_result"]
        person_detections = result["person_detections"]
        
        # Generate intelligent action when alert detected
        if detection_result.get('alert_level') in ['critical', 'high']:
            emergency_type = detection_result.get('emergency_type', 'unknown')
            confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
            
            # Try to find the most recent alert image
            try:
                import glob
                import os
                from pathlib import Path
                
                alerts_path = Path(alerts_folder)
                if alerts_path.exists():
                    # Look for most recent alert image
                    image_files = list(alerts_path.glob("*.jpg"))
                    if image_files:
                        last_alert_image_path = max(image_files, key=lambda p: p.stat().st_ctime)
                        
            except Exception as e:
                print(f"⚠️ Could not find alert image: {e}")
                last_alert_image_path = None
            
            # Generate intelligent action description
            intelligent_action = "Standard alert message"
            if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline and last_alert_image_path:
                try:
                    # Generate intelligent action based on image content
                    status = "danger" if detection_result.get('alert_level') == 'critical' else "warning"
                    event_type = "abnormal_behavior" if "seizure" in emergency_type else "fall"
                    
                    # Get Vietnamese caption from image
                    vietnamese_caption, metadata = caption_pipeline.generate_professional_caption(str(last_alert_image_path))
                    
                    # Create enhanced action message
                    if status == "danger":
                        if event_type == "fall":
                            intelligent_action = f"🚨 KHẨN CẤP - TÉ NGÃ: {vietnamese_caption} - YÊU CẦU HỖ TRỢ NGAY! (Tin cậy: {confidence:.0%})"
                        else:
                            intelligent_action = f"🆘 KHẨN CẤP - CO GIẬT: {vietnamese_caption} - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"
                    else:
                        intelligent_action = f"⚠️ CẢNH BÁO: {vietnamese_caption} - Cần theo dõi (Tin cậy: {confidence:.0%})"
                        
                    print(f"🤖 INTELLIGENT ACTION: {intelligent_action}")
                    if metadata.get('english_caption'):
                        print(f"   🌍 English: {metadata['english_caption']}")
                        
                except Exception as e:
                    print(f"⚠️ Intelligent action generation failed: {e}")
                    intelligent_action = f"🚨 EMERGENCY: {emergency_type.upper()} detected (confidence: {confidence:.2f})"
            else:
                intelligent_action = f"🚨 EMERGENCY: {emergency_type.upper()} detected (confidence: {confidence:.2f})"
            
            # Log emergency alert
            print(f"🚨 EMERGENCY ALERT: {emergency_type.upper()} detected (confidence: {confidence:.2f})")
            print(f"   📱 Notification sent to backend")
            print(f"   📡 Event published to Supabase realtime")
            print(f"   💬 Action: {intelligent_action}")
            
            # 💾 SAVE EVENT TO DATABASE
            try:
                print(f"💾 Saving event to database...")
                
                # Determine event_type and status
                event_type = "abnormal_behavior" if "seizure" in emergency_type else "fall"
                status = "danger" if detection_result.get('alert_level') == 'critical' else "warning"
                
                # Get bounding boxes from detection result
                bounding_boxes = detection_result.get('bounding_boxes', [])
                if not bounding_boxes and person_detections:
                    # Fallback to person detections
                    bounding_boxes = [{
                        'x': int(p.get('bbox', [0, 0, 0, 0])[0]),
                        'y': int(p.get('bbox', [0, 0, 0, 0])[1]),
                        'width': int(p.get('bbox', [0, 0, 0, 0])[2] - p.get('bbox', [0, 0, 0, 0])[0]),
                        'height': int(p.get('bbox', [0, 0, 0, 0])[3] - p.get('bbox', [0, 0, 0, 0])[1]),
                        'confidence': float(p.get('confidence', confidence)),
                        'class': 'person'
                    } for p in person_detections[:3]]  # Max 3 persons
                
                # Publish event using pipeline's event publisher
                if event_type == 'fall':
                    alert_result = pipeline.event_publisher.publish_fall_detection(
                        confidence=confidence,
                        bounding_boxes=bounding_boxes,
                        context={
                            'description': intelligent_action,
                            'alert_level': detection_result.get('alert_level'),
                            'frame_count': frame_count
                        }
                    )
                else:  # abnormal_behavior (seizure)
                    alert_result = pipeline.event_publisher.publish_seizure_detection(
                        confidence=confidence,
                        bounding_boxes=bounding_boxes,
                        context={
                            'description': intelligent_action,
                            'alert_level': detection_result.get('alert_level'),
                            'frame_count': frame_count
                        }
                    )
                
                if alert_result and alert_result.get('success'):
                    print(f"   ✅ Event saved to database!")
                    print(f"   🆔 Event ID: {alert_result.get('event_id', 'N/A')}")
                    print(f"   📊 Status: {status}")
                    print(f"   💾 Database: PostgreSQL")
                else:
                    print(f"   ⚠️ Event save failed: {alert_result.get('error', 'Unknown error') if alert_result else 'No response'}")
                    
            except Exception as db_error:
                print(f"   ❌ Database save error: {db_error}")
                import traceback
                traceback.print_exc()
        
        # Hiển thị Normal View
        cv2.imshow("Healthcare Monitor - Normal View", result["normal_window"])
        
        # Hiển thị Analysis View với statistics overlay
        analysis_view = pipeline.visualize_dual_detection(frame, detection_result, person_detections)
        analysis_view = pipeline.draw_statistics_overlay(analysis_view, pipeline.stats)
        
        # Add intelligent action status to analysis view
        if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline:
            status_text = f"🤖 Intelligent Actions: {'BLIP' if caption_pipeline.blip_loaded else 'Rule-based'} + {'AI Translation' if caption_pipeline.translator_loaded else 'Rule-based Translation'}"
        else:
            status_text = "📝 Static Actions Only"
        
        cv2.putText(analysis_view, status_text, (10, analysis_view.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Healthcare Monitor - Analysis View", analysis_view)
        
        # Check keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 Shutting down Healthcare Monitoring System...")
            emergency_alarm_handler.stop()  # Stop alarm handler
            break
        elif key == ord('s'):
            # Show detailed statistics
            pipeline.print_final_statistics()
            if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline:
                print(f"\n🤖 INTELLIGENT ACTION STATUS:")
                print(f"   BLIP Model: {'✅ Loaded' if caption_pipeline.blip_loaded else '❌ Not loaded'}")
                print(f"   Translation: {'✅ AI Model' if caption_pipeline.translator_loaded else '📝 Rule-based'}")
        elif key == ord('a'):
            # Show emergency alarm status
            print("\n🔊 EMERGENCY ALARM SYSTEM STATUS:")
            print(f"   Handler Running: {'✅ Yes' if emergency_alarm_handler.is_running else '❌ No'}")
            print(f"   Channel: {emergency_alarm_handler.channel_name}")
            print(f"   Processed Events: {len(emergency_alarm_handler.processed_events)}")
            
            # Show audio status
            audio_status = audio_alert_service.get_status()
            print(f"   Audio Enabled: {'✅ Yes' if audio_status['enabled'] else '❌ No'}")
            print(f"   Audio Backend: {audio_status.get('audio_backend', 'unknown')}")
            print(f"   Audio Devices: {audio_status.get('available_devices', 0)}")
        elif key == ord('t'):
            # Test emergency alarm manually
            print("\n🔊 TESTING EMERGENCY ALARM...")
            print("   This will play a 5-second test alarm")
            try:
                test_user_id = os.getenv('DEFAULT_USER_ID', 'test_user')
                test_result = asyncio.run(audio_alert_service.play_emergency_alarm(
                    user_id=test_user_id,
                    triggered_by='manual_test',
                    duration=5  # Short 5-second test
                ))
                if test_result['success']:
                    print(f"   ✅ Test alarm played successfully!")
                    print(f"      Volume: {test_result.get('volume', 1.0) * 100:.0f}%")
                    print(f"      Duration: {test_result.get('duration', 5)}s")
                else:
                    print(f"   ❌ Test alarm failed: {test_result['message']}")
            except Exception as e:
                print(f"   ❌ Test alarm error: {e}")
        elif key == ord('i'):
            # Show intelligent action info
            if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline:
                print(f"\n🤖 INTELLIGENT ACTION INFO:")
                print(f"   BLIP Model: {'✅ Active' if caption_pipeline.blip_loaded else '❌ Inactive'}")
                print(f"   Translation: {'✅ AI Model' if caption_pipeline.translator_loaded else '📝 Rule-based fallback'}")
                print(f"   Last Alert Image: {last_alert_image_path.name if last_alert_image_path else 'None'}")
                print(f"   Frame Count: {frame_count}")
            else:
                print(f"\n📝 Static action messages only - Install 'transformers torch pillow' for intelligent actions")
        elif key == ord('e'):
            # Create random event and save directly to database
            print("\n🎲 Creating random test event...")
            try:
                import random
                import uuid
                import json
                from datetime import datetime, timezone
                
                # Random event types and data
                event_types = ['fall', 'abnormal_behavior']
                event_type = random.choice(event_types)
                confidence = random.uniform(0.3, 0.95)
                
                # Random Vietnamese descriptions for testing
                test_descriptions = [
                    "Một người đàn ông trong glasses đang đứng trong phòng",
                    "Một phụ nữ đang ngồi trên ghế",
                    "Hai người đang nói chuyện trong phòng khách",
                    "Một người già đang đi bộ",
                    "Một em bé đang chơi trên sàn nhà",
                    "Một người đàn ông trong áo đen đang cầm điện thoại",
                    "Một phụ nữ đang đọc sách trên giường",
                    "Một người đàn ông đang xem TV"
                ]
                
                random_description = random.choice(test_descriptions)
                
                # Generate intelligent action for console
                if event_type == 'abnormal_behavior':
                    if confidence >= 0.50:
                        intelligent_action = f"🆘 KHẨN CẤP - CO GIẬT: {random_description} - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"
                        status = 'danger'
                    elif confidence >= 0.30:
                        intelligent_action = f"⚠️ CẢNH BÁO BẤT THƯỜNG: {random_description} - Cần theo dõi chặt chẽ (Tin cậy: {confidence:.0%})"
                        status = 'warning'
                    else:
                        intelligent_action = f"📊 QUAN SÁT: {random_description} - Tiếp tục theo dõi (Tin cậy: {confidence:.0%})"
                        status = 'normal'
                elif event_type == 'fall':
                    if confidence >= 0.60:
                        intelligent_action = f"🚨 KHẨN CẤP - TÉ NGÃ: {random_description} - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: {confidence:.0%})"
                        status = 'danger'
                    elif confidence >= 0.40:
                        intelligent_action = f"⚠️ CẢNH BÁO TÉ NGÃ: {random_description} - Cần theo dõi (Tin cậy: {confidence:.0%})"
                        status = 'warning'
                    else:
                        intelligent_action = f"📊 THEO DÕI: {random_description} - Quan sát (Tin cậy: {confidence:.0%})"
                        status = 'normal'
                
                print(f"🎯 Test Event Details:")
                print(f"   Type: {event_type.upper()}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Description: {random_description}")
                print(f"🤖 INTELLIGENT ACTION: {intelligent_action}")
                
                # Save directly to database
                try:
                    # Get database service from pipeline
                    db_service = pipeline.event_publisher.postgresql_service
                    
                    # Generate new event ID
                    event_id = str(uuid.uuid4())
                    
                    # Get database connection
                    conn = db_service.get_connection()
                    if conn:
                        cursor = conn.cursor()
                        
                        # Get required IDs for database constraints
                        user_id = os.getenv('DEFAULT_USER_ID')
                        camera_id = db_service._get_user_camera_id(user_id)
                        if not camera_id:
                            camera_id = db_service._get_any_camera_id()
                        
                        # Create snapshot_id
                        snapshot_id = db_service._create_minimal_snapshot(camera_id, user_id)
                        if not snapshot_id:
                            snapshot_id = str(uuid.uuid4())
                            print("⚠️ Using dummy snapshot_id")
                        
                        # Insert directly into event_detections table
                        insert_query = """
                            INSERT INTO event_detections (
                                event_id, user_id, camera_id, snapshot_id, event_type, 
                                event_description, confidence_score, status, detected_at, 
                                created_at, detection_data
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        current_time = datetime.now(timezone.utc)
                        detection_data = {
                            'test_event': True,
                            'manual_trigger': True,
                            'original_description': random_description,
                            'bounding_boxes': [{
                                'x': random.randint(100, 400),
                                'y': random.randint(100, 300),
                                'width': random.randint(50, 200),
                                'height': random.randint(50, 200),
                                'confidence': confidence,
                                'class': 'person'
                            }]
                        }
                        
                        # Validate intelligent_action before inserting
                        if not intelligent_action or intelligent_action.strip() == '' or intelligent_action.lower() == 'null':
                            print(f"❌ Skipping event insert - empty intelligent_action for {event_type}")
                            continue
                        
                        cursor.execute(insert_query, (
                            event_id,
                            user_id,
                            camera_id,
                            snapshot_id,
                            event_type,
                            intelligent_action,  # Use full intelligent action as description
                            confidence,
                            status,
                            current_time,
                            current_time,
                            json.dumps(detection_data)  # Use json.dumps instead
                        ))
                        
                        conn.commit()
                        db_service.return_connection(conn)
                        
                        print(f"✅ Event saved successfully to database!")
                        print(f"   🆔 Event ID: {event_id}")
                        print(f"   📊 Status: {status}")
                        print(f"   💾 Database: PostgreSQL")
                        print(f"   ⏰ Time: {current_time.strftime('%H:%M:%S')}")
                        
                    else:
                        print(f"❌ Failed to get database connection!")
                        
                except Exception as db_error:
                    print(f"❌ Database error: {db_error}")
                    # Try alternative method
                    print("🔄 Trying alternative saving method...")
                    
                    # Fallback: use the existing event publisher
                    if event_type == 'fall':
                        alert_result = pipeline.event_publisher.publish_fall_detection(
                            confidence=confidence,
                            bounding_boxes=[{
                                'x': random.randint(100, 400),
                                'y': random.randint(100, 300),
                                'width': random.randint(50, 200),
                                'height': random.randint(50, 200),
                                'confidence': confidence,
                                'class': 'person'
                            }],
                            context={
                                'description': random_description,
                                'manual_trigger': True,
                                'test_event': True
                            }
                        )
                    else:
                        alert_result = pipeline.event_publisher.publish_seizure_detection(
                            confidence=confidence,
                            bounding_boxes=[{
                                'x': random.randint(100, 400),
                                'y': random.randint(100, 300),
                                'width': random.randint(50, 200),
                                'height': random.randint(50, 200),
                                'confidence': confidence,
                                'class': 'person'
                            }],
                            context={
                                'description': random_description,
                                'manual_trigger': True,
                                'test_event': True
                            }
                        )
                    
                    if alert_result and isinstance(alert_result, dict):
                        event_id = alert_result.get('event_id', 'unknown')
                        print(f"✅ Event saved via fallback method!")
                        print(f"   🆔 Event ID: {event_id}")
                        print(f"   � Result: {alert_result}")
                
            except Exception as e:
                print(f"❌ Error creating random event: {e}")
                import traceback
                print(f"   🔍 Traceback: {traceback.format_exc()}")
        # ...các xử lý khác như lưu ảnh, cập nhật thống kê...

    print("📱 Notifications stopped")
    print("🏥 Healthcare monitoring stopped") 
    cv2.destroyAllWindows()