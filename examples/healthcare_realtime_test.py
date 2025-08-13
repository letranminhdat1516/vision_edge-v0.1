"""
Healthcare System với Supabase Realtime Integration
Test thật với database và mobile notifications
"""

import asyncio
import logging
import json
from datetime import datetime
from src.service.supabase_realtime_service import SupabaseRealtimeService
from src.service.healthcare_event_publisher_v2 import HealthcareEventPublisher
from src.camera.rtsp_camera import RTSPCamera
from src.video_processing.video_processor import VideoProcessor
from src.fall_detection.yolo_fall_detector import YOLOFallDetector
from src.seizure_detection.vsvig_detector import VSViGDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthcareRealtimeSystem:
    """Healthcare system với Supabase realtime integration"""
    
    def __init__(self):
        logger.info("🏥 Initializing Healthcare Realtime System...")
        
        # Initialize services
        self.supabase_service = SupabaseRealtimeService()
        self.event_publisher = HealthcareEventPublisher()
        
        # Initialize AI detectors
        self.fall_detector = YOLOFallDetector()
        self.seizure_detector = VSViGDetector()
        
        # Initialize camera and video processor
        self.camera = RTSPCamera()
        self.video_processor = VideoProcessor(
            fall_detector=self.fall_detector,
            seizure_detector=self.seizure_detector
        )
        
        self.is_running = False
        
    async def start_realtime_monitoring(self):
        """Start healthcare monitoring với Supabase realtime"""
        logger.info("🚀 Starting Healthcare Realtime Monitoring...")
        
        try:
            # Connect to camera
            if not self.camera.connect():
                logger.error("❌ Failed to connect to camera")
                return
                
            logger.info("📹 Camera connected successfully")
            self.is_running = True
            
            frame_count = 0
            
            while self.is_running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process every 5th frame to reduce load
                if frame_count % 5 == 0:
                    # Process frame for healthcare events
                    results = await self._process_healthcare_frame(frame)
                    
                    # Send events to Supabase for realtime notifications
                    for event in results:
                        await self._send_realtime_event(event)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping healthcare monitoring...")
        except Exception as e:
            logger.error(f"❌ Error in healthcare monitoring: {e}")
        finally:
            self._cleanup()
    
    async def _process_healthcare_frame(self, frame):
        """Process frame for healthcare events"""
        events = []
        
        try:
            # Fall detection
            fall_results = self.fall_detector.detect(frame)
            if fall_results and len(fall_results) > 0:
                for detection in fall_results:
                    if detection.get('confidence', 0) > 0.3:  # Threshold for logging
                        event = {
                            'event_type': 'fall',
                            'confidence_score': float(detection.get('confidence', 0)),
                            'detected_at': datetime.now().isoformat(),
                            'event_id': f'fall_{int(datetime.now().timestamp())}_{detection.get("track_id", 0)}',
                            'camera_id': 'rtsp_cam_01',
                            'location': 'Healthcare Room',
                            'description': f'Fall detected with {detection.get("confidence", 0):.2f} confidence',
                            'bounding_box': detection.get('bbox', {}),
                            'metadata': {
                                'detection_method': 'YOLO',
                                'track_id': detection.get('track_id', 0)
                            }
                        }
                        events.append(event)
            
            # Seizure detection
            seizure_results = self.seizure_detector.detect(frame)
            if seizure_results and len(seizure_results) > 0:
                for detection in seizure_results:
                    if detection.get('confidence', 0) > 0.3:  # Threshold for logging
                        event = {
                            'event_type': 'abnormal_behavior',
                            'confidence_score': float(detection.get('confidence', 0)),
                            'detected_at': datetime.now().isoformat(),
                            'event_id': f'seizure_{int(datetime.now().timestamp())}_{detection.get("track_id", 0)}',
                            'camera_id': 'rtsp_cam_01',
                            'location': 'Healthcare Room',
                            'description': f'Abnormal behavior detected with {detection.get("confidence", 0):.2f} confidence',
                            'bounding_box': detection.get('bbox', {}),
                            'metadata': {
                                'detection_method': 'VSViG',
                                'track_id': detection.get('track_id', 0)
                            }
                        }
                        events.append(event)
                        
        except Exception as e:
            logger.error(f"❌ Error processing healthcare frame: {e}")
        
        return events
    
    async def _send_realtime_event(self, event_data):
        """Send event to Supabase for mobile realtime notifications"""
        try:
            # Only send high confidence events to avoid spam
            confidence = event_data.get('confidence_score', 0)
            event_type = event_data.get('event_type')
            
            should_send = False
            
            if event_type == 'fall' and confidence >= 0.5:
                should_send = True
            elif event_type == 'abnormal_behavior' and confidence >= 0.4:
                should_send = True
            
            if should_send:
                # Insert to Supabase for realtime mobile notifications
                success = await self.supabase_service.insert_healthcare_event(event_data)
                
                if success:
                    logger.info(f"📤 Sent realtime notification: {event_type} ({confidence:.2f})")
                    
                    # Also publish to local system
                    self.event_publisher.publish_healthcare_event(event_data)
                else:
                    logger.warning(f"⚠️ Failed to send realtime notification for {event_type}")
            
        except Exception as e:
            logger.error(f"❌ Error sending realtime event: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up healthcare system...")
        self.is_running = False
        
        if self.camera:
            self.camera.disconnect()

async def test_realtime_event():
    """Test sending a realtime event manually"""
    logger.info("🧪 Testing Supabase Realtime Event...")
    
    supabase_service = SupabaseRealtimeService()
    
    # Test event
    test_event = {
        'event_type': 'fall',
        'confidence_score': 0.87,
        'detected_at': datetime.now().isoformat(),
        'event_id': f'test_fall_{int(datetime.now().timestamp())}',
        'camera_id': 'test_cam',
        'location': 'Test Room',
        'description': 'Manual test fall detection event',
        'bounding_box': {'x': 100, 'y': 150, 'width': 200, 'height': 300},
        'metadata': {'test': True, 'manual': True}
    }
    
    success = await supabase_service.insert_healthcare_event(test_event)
    
    if success:
        print("✅ Test event sent successfully!")
        print("🔔 Check your mobile demo HTML at: http://localhost:8000/mobile_realtime_demo.html")
        print("📱 You should see a realtime notification appear!")
        
        # Show mobile payload format
        mobile_payload = supabase_service.create_mobile_notification_payload(test_event)
        print("\n📋 Mobile notification payload:")
        print(json.dumps(mobile_payload, indent=2, ensure_ascii=False))
    else:
        print("❌ Failed to send test event")
        print("💡 Make sure you have configured .env file with Supabase credentials")

async def main():
    """Main function"""
    print("🏥 Healthcare Realtime System với Supabase")
    print("=" * 50)
    print("1. Test realtime event")
    print("2. Start full healthcare monitoring")
    print("=" * 50)
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        await test_realtime_event()
    elif choice == "2":
        system = HealthcareRealtimeSystem()
        await system.start_realtime_monitoring()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
