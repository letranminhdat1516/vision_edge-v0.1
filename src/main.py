

import cv2
from service.monitor_service import MonitorService
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
from service.healthcare_event_publisher import healthcare_publisher

# Import intelligent action generation
try:
    from service.image_caption_service import get_professional_caption_pipeline
    INTELLIGENT_ACTIONS_AVAILABLE = True
    print("🤖 Intelligent Action Generation: AVAILABLE")
except ImportError:
    INTELLIGENT_ACTIONS_AVAILABLE = False
    print("📝 Intelligent Action Generation: Using static messages")

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
    print("Press 'q' to quit, 's' to show statistics, 'i' to toggle intelligent actions")
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
        
        key = cv2.waitKey(1)
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
        # ...các xử lý khác như lưu ảnh, cập nhật thống kê...

    print("📱 Notifications stopped")
    print("🏥 Healthcare monitoring stopped") 
    cv2.destroyAllWindows()
