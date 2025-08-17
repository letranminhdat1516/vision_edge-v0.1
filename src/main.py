

import cv2
from service.monitor_service import MonitorService
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
from service.healthcare_event_publisher import healthcare_publisher

# Import FCM service để đảm bảo nó được khởi tạo
try:
    from service.fcm_notification_service import fcm_service
    print("🔥 FCM Service initialized for emergency notifications")
    print(f"   Firebase Project: {fcm_service.project_id}")
    print(f"   FCM Enabled: {fcm_service.enable_notifications}")
    if fcm_service.device_tokens:
        print(f"   Device Tokens: {len(fcm_service.device_tokens)} loaded from .env")
    if fcm_service.caregiver_tokens:
        print(f"   Caregiver Tokens: {len(fcm_service.caregiver_tokens)} loaded from .env")
    if fcm_service.emergency_tokens:
        print(f"   Emergency Tokens: {len(fcm_service.emergency_tokens)} loaded from .env")
except ImportError as e:
    print(f"⚠️ FCM Service not available: {e}")
    fcm_service = None

print("="*60)

if __name__ == "__main__":
    camera_config = {
        'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
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
    
    # Khởi tạo AdvancedHealthcarePipeline với FCM support
    print("🏥 Initializing Healthcare Pipeline with Real FCM Integration...")
    print("   - Real-time fall detection")
    print("   - Real-time seizure detection") 
    print("   - Emergency FCM notifications")
    print("   - Supabase realtime integration")
    print("   - Mobile app notifications")
    
    pipeline = AdvancedHealthcarePipeline(
        camera=camera, 
        video_processor=video_processor, 
        fall_detector=fall_detector, 
        seizure_detector=seizure_detector, 
        seizure_predictor=seizure_predictor, 
        alerts_folder=alerts_folder,
        user_fcm_tokens=None  # FCM tokens sẽ được load từ .env automatically
    )
    
    print("✅ Healthcare Pipeline initialized with FCM emergency notifications!")
    print("📱 Mobile devices will receive real-time emergency alerts")
    print("="*60)


    print("🎥 Starting Healthcare Monitoring System...")
    print("📱 Emergency FCM notifications: ACTIVE")
    print("🏥 Real-time healthcare detection: ACTIVE")
    print("Press 'q' to quit, 's' to show statistics")
    print("="*60)

    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        result = pipeline.process_frame(frame)
        detection_result = result["detection_result"]
        person_detections = result["person_detections"]
        
        # Log critical alerts to console
        if detection_result.get('alert_level') in ['critical', 'high']:
            emergency_type = detection_result.get('emergency_type', 'unknown')
            confidence = detection_result.get('fall_confidence', 0) if 'fall' in emergency_type else detection_result.get('seizure_confidence', 0)
            print(f"🚨 EMERGENCY ALERT: {emergency_type.upper()} detected (confidence: {confidence:.2f})")
            print(f"   📱 FCM notification sent to mobile devices")
            print(f"   📡 Event published to Supabase realtime")
        
        # Hiển thị Normal View
        cv2.imshow("Healthcare Monitor - Normal View", result["normal_window"])
        
        # Hiển thị Analysis View với statistics overlay
        analysis_view = pipeline.visualize_dual_detection(frame, detection_result, person_detections)
        analysis_view = pipeline.draw_statistics_overlay(analysis_view, pipeline.stats)
        cv2.imshow("Healthcare Monitor - Analysis View", analysis_view)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\n🛑 Shutting down Healthcare Monitoring System...")
            break
        elif key == ord('s'):
            # Show detailed statistics
            pipeline.print_final_statistics()
        # ...các xử lý khác như lưu ảnh, cập nhật thống kê...

    print("📱 FCM notifications stopped")
    print("🏥 Healthcare monitoring stopped") 
    cv2.destroyAllWindows()

    # Xóa duplicate main block cuối file
