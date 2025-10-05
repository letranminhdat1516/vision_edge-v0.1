
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
from pathlib import Path
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline

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
print("🔍 Analyzing camera configuration...")

# Detect camera mode
camera_mode, camera_data = detect_camera_mode()
print("="*60)

if __name__ == "__main__":
    # Choose camera system based on detection
    if camera_mode == 'dual' and camera_data:
        print("🎥🎥 Initializing DUAL CAMERA SYSTEM...")
        
        try:
            from service.dual_camera_surveillance_system import SameRoomDualDetection
            from service.video_processing_service import VideoProcessingService
            from service.fall_detection_service import FallDetectionService
            from service.seizure_detection_service import SeizureDetectionService
            
            # Prepare camera configs for dual detection
            camera_configs = []
            for i, (camera_id, camera_info) in enumerate(camera_data.items()):
                camera_configs.append({
                    'camera_id': camera_id,
                    'name': camera_info.get('name', f'Camera {i+1}'),
                    'position': camera_info.get('position', 'unknown'),
                    'area': camera_info.get('location', 'Unknown Room'),
                    'rtsp_url': camera_info.get('rtsp_url')
                })
            
            # Initialize services
            video_processor = VideoProcessingService(120)
            fall_detector = FallDetectionService()
            seizure_detector = SeizureDetectionService()
            
            # Initialize database service for dual detection (simplified)
            database_service = None
            print("   💾 Using mock database for dual detection")
            
            # Create dual detection system
            dual_detector = SameRoomDualDetection(
                camera_configs=camera_configs,
                video_processor=video_processor,
                fall_detector=fall_detector,
                seizure_detector=seizure_detector
            )
            
            print(f"✅ Dual Detection System initialized with {len(camera_configs)} cameras")
            for config in camera_configs:
                print(f"   📹 {config['name']} ({config['position']})")
            
            print("🎥🎥 Starting Dual Camera Healthcare Monitoring...")
            print("🚫 Blind spot elimination: ACTIVE")
            print("📱 Emergency notifications: ACTIVE")
            print("🏥 Real-time healthcare detection: ACTIVE")
            print("Press Ctrl+C to stop")
            print("="*60)
            
            # Start dual detection
            if dual_detector.start():
                print("✅ Enhanced Dual Detection System running...")
                
                try:
                    stats_counter = 0
                    while True:
                        # Enhanced emergency detection check
                        emergency_status = dual_detector.detect_emergency_events()
                        
                        if emergency_status['emergency']:
                            print(f"🚨 EMERGENCY DETECTED! Confidence: {emergency_status['confidence']:.2f}")
                            print(f"   📊 Events: {len(emergency_status['events'])}")
                            print(f"   🤝 Consensus: {emergency_status['consensus']:.2f}")
                            print(f"   📐 Coverage: {emergency_status['coverage']:.1%}")
                            print(f"   👥 Persons: {emergency_status['persons_detected']}")
                            print(f"   📹 Sources: {emergency_status['detection_sources']}")
                            
                            # Generate Vietnamese caption for emergency
                            if hasattr(dual_detector, 'caption_pipeline') and dual_detector.caption_pipeline and 'fused_result' in emergency_status:
                                try:
                                    fused_result = emergency_status['fused_result']
                                    if hasattr(fused_result, 'latest_frame') and fused_result.latest_frame is not None:
                                        # Save emergency frame temporarily for captioning
                                        temp_frame_path = f"data/saved_frames/temp_emergency_{int(time.time())}.jpg"
                                        cv2.imwrite(temp_frame_path, fused_result.latest_frame)
                                        
                                        # Generate Vietnamese caption
                                        vietnamese_caption, metadata = dual_detector.caption_pipeline.generate_professional_caption(temp_frame_path)
                                        print(f"   🇻🇳 Mô tả: {vietnamese_caption}")
                                        if metadata.get('english_caption'):
                                            print(f"   🌍 English: {metadata['english_caption']}")
                                        
                                        # Clean up temp file
                                        try:
                                            import os
                                            os.remove(temp_frame_path)
                                        except:
                                            pass
                                            
                                except Exception as e:
                                    print(f"   ⚠️ Caption generation failed: {e}")
                            
                            for event in emergency_status['events']:
                                print(f"      🔥 {event['type']}: {event['confidence']:.2f}")
                            
                            # Save emergency to database
                            try:
                                # Create emergency event data
                                event_data = {
                                    'event_id': str(uuid.uuid4()),
                                    'event_type': emergency_status['events'][0]['type'] if emergency_status['events'] else 'unknown',
                                    'confidence': emergency_status['confidence'],
                                    'consensus': emergency_status['consensus'],
                                    'coverage': emergency_status['coverage'],
                                    'persons_detected': emergency_status['persons_detected'],
                                    'detection_sources': emergency_status['detection_sources'],
                                    'timestamp': time.time(),
                                    'vietnamese_caption': vietnamese_caption if 'vietnamese_caption' in locals() else None,
                                    'dual_camera': True
                                }
                                
                                # Save to mock database (or real database if available)
                                from service.database_mock_adapter import mock_supabase_service
                                table = mock_supabase_service.table('emergency_events')
                                result = table.insert(event_data)
                                
                                print(f"   💾 Emergency saved to database: {event_data['event_id'][:8]}...")
                                
                            except Exception as save_error:
                                print(f"   ⚠️ Failed to save emergency: {save_error}")
                            
                            # Also save keypoints if available
                            if 'fused_result' in emergency_status and hasattr(emergency_status['fused_result'], 'combined_persons'):
                                try:
                                    persons_with_keypoints = []
                                    for person in emergency_status['fused_result'].combined_persons:
                                        if 'keypoints' in person:
                                            persons_with_keypoints.append({
                                                'person_id': str(uuid.uuid4()),
                                                'bbox': person.get('bbox', []),
                                                'confidence': person.get('confidence', 0),
                                                'keypoints': person['keypoints'],
                                                'timestamp': time.time()
                                            })
                                    
                                    if persons_with_keypoints:
                                        keypoints_table = mock_supabase_service.table('person_keypoints')
                                        for person_data in persons_with_keypoints:
                                            keypoints_table.insert(person_data)
                                        
                                        print(f"   🦴 Keypoints saved: {len(persons_with_keypoints)} persons")
                                        
                                except Exception as keypoint_error:
                                    print(f"   ⚠️ Failed to save keypoints: {keypoint_error}")
                        
                        # Show enhanced statistics every 30 seconds
                        stats_counter += 1
                        if stats_counter % 30 == 0:
                            stats = dual_detector.get_statistics()
                            real_time_status = dual_detector.get_real_time_status()
                            
                            print("="*60)
                            print("📊 DUAL CAMERA SYSTEM STATUS:")
                            print(f"   🎥 Cameras Active: {stats['camera_count']}")
                            print(f"   📈 Total Detections: {stats['total_camera_detections']}")
                            print(f"   🔄 Fusion Efficiency: {stats['fusion_efficiency']:.1f}%")
                            print(f"   🤝 Consensus Rate: {stats['consensus_rate']:.1f}%")
                            print(f"   🚨 Emergency Events: {stats['emergency_events']}")
                            print(f"   ⚡ Dual Camera Boosts: {stats['dual_camera_boost_applied']}")
                            print(f"   🔍 Coverage Improvements: {stats['coverage_improvements']}")
                            print(f"   🟢 System Status: {real_time_status['system_status']}")
                            print("="*60)
                        
                        # Check for recent detections with enhanced feedback
                        if dual_detector.has_recent_detections():
                            real_time_status = dual_detector.get_real_time_status()
                            if real_time_status['dual_camera_operational']:
                                print("🎥🎥 Dual camera detection active - Enhanced coverage")
                            else:
                                print("🎥 Single camera detection active")
                        
                        time.sleep(1)
                        
                except KeyboardInterrupt:
                    print("\n🛑 Stopping dual detection system...")
                finally:
                    dual_detector.stop()
                    print("✅ Dual detection system stopped")
                    exit(0)  # Exit after dual camera system
            else:
                print("❌ Failed to start dual detection, analyzing fallback options...")
                
                # Enhanced fallback with credential testing
                print("🔄 Testing individual camera connections for fallback...")
                working_camera = None
                
                if camera_data:
                    for camera_key, camera_info in camera_data.items():
                        camera_name = camera_info.get('name', camera_key)
                        rtsp_url = camera_info.get('rtsp_url', '')
                        
                        print(f"   🧪 Testing {camera_name}...")
                        
                        # Quick connection test
                        try:
                            cap = cv2.VideoCapture(rtsp_url)
                            if cap and cap.isOpened():
                                ret, frame = cap.read()
                                cap.release()
                                if ret and frame is not None:
                                    print(f"   ✅ {camera_name} is working!")
                                    working_camera = camera_info
                                    break
                                else:
                                    print(f"   ❌ {camera_name} connected but no frames")
                            else:
                                print(f"   ❌ {camera_name} connection failed")
                        except Exception as e:
                            print(f"   ❌ {camera_name} error: {e}")
                
                if working_camera:
                    print(f"🎥 Found working camera, switching to single camera mode...")
                    camera_mode = 'single'
                    camera_data = working_camera
                else:
                    print("❌ No working cameras found, using fallback configuration...")
                    camera_mode = 'single'
                    camera_data = None
                
        except Exception as e:
            print(f"❌ Dual detection error: {e}")
            print("🔄 Falling back to single camera system...")
            camera_mode = 'single'
            if camera_data and isinstance(camera_data, dict):
                camera_data = list(camera_data.values())[0]
            else:
                camera_data = None
    
    # Single camera system (original logic) with enhanced fallback
    if camera_mode == 'single':
        print("🎥 Starting SINGLE CAMERA SYSTEM...")
        
        # Setup camera configuration with fallback options
        if camera_data and isinstance(camera_data, dict):
            camera_url = camera_data.get('rtsp_url', '')
            camera_name = camera_data.get('name', 'Camera')
            print(f"   📹 Using configured camera: {camera_name}")
            print(f"   🔗 URL: {camera_url}")
        else:
            print("   📹 No working camera found, trying fallback configurations...")
            
            # Fallback URLs to try
            fallback_urls = [
                'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
                'rtsp://admin:123456@192.168.8.122:554/cam/realmonitor?channel=1&subtype=0',
                'rtsp://admin:L2400907@192.168.8.86:554/cam/realmonitor?channel=1&subtype=1',
                'rtsp://admin:admin@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
                0  # USB camera fallback
            ]
            
            camera_url = None
            for i, url in enumerate(fallback_urls, 1):
                print(f"   🧪 Trying fallback option {i}: {url}")
                
                try:
                    cap = cv2.VideoCapture(url)
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            print(f"   ✅ Fallback option {i} works!")
                            camera_url = url
                            break
                        else:
                            print(f"   ❌ Option {i}: Connected but no frames")
                    else:
                        print(f"   ❌ Option {i}: Connection failed")
                except Exception as e:
                    print(f"   ❌ Option {i}: Error - {e}")
            
            if not camera_url:
                print("❌ All fallback options failed. System cannot start.")
                print("💡 Please check:")
                print("   - Camera power and network connection")
                print("   - Correct IP addresses and credentials")
                print("   - RTSP port accessibility")
                print("   - Try connecting with VLC media player first")
                exit(1)
            
        camera_config = {
            'url': camera_url,
            'buffer_size': 1,
            'fps': 15,
            'resolution': (640, 480),
            'auto_reconnect': True
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
    seizure_predictor = SeizurePredictor(temporal_window=25, alert_threshold=0.7, warning_threshold=0.5)
    
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
    print("Press 'q' to quit, 's' to show statistics, 'i' to toggle intelligent actions, 'e' to create random test event")
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
                    event_type = "seizure" if "seizure" in emergency_type else "fall"
                    
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
            break
        elif key == ord('s'):
            # Show detailed statistics
            pipeline.print_final_statistics()
            if INTELLIGENT_ACTIONS_AVAILABLE and caption_pipeline:
                print(f"\n🤖 INTELLIGENT ACTION STATUS:")
                print(f"   BLIP Model: {'✅ Loaded' if caption_pipeline.blip_loaded else '❌ Not loaded'}")
                print(f"   Translation: {'✅ AI Model' if caption_pipeline.translator_loaded else '📝 Rule-based'}")
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
                        
                        # Insert directly into event_detections table
                        insert_query = """
                            INSERT INTO event_detections (
                                event_id, event_type, event_description, confidence_score, 
                                status, detected_at, created_at, detection_data
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                        
                        cursor.execute(insert_query, (
                            event_id,
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