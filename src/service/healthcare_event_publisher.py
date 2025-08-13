"""
Healthcare Event Publisher Service
Integrates healthcare detection pipeline with Supabase realtime system
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging

# Try to import Supabase service, fallback to mock
try:
    from service.postgresql_healthcare_service import postgresql_service as realtime_service
    from service.mobile_realtime_notification_service import send_mobile_notification
    MOCK_MODE = not realtime_service.is_connected
    if MOCK_MODE:
        logger = logging.getLogger(__name__)
        logger.warning("Supabase connection failed, using mock mode")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import services: {e}, using mock mode")
    MOCK_MODE = True
    
    # Mock mobile notification function
    def send_mobile_notification(event_response): 
        print(f"📱 Mock mobile notification: {event_response}")

if MOCK_MODE:
    from service.mock_supabase_service import mock_supabase_service as realtime_service

logger = logging.getLogger(__name__)

class HealthcareEventPublisher:
    """Service for publishing healthcare events to Supabase realtime system"""
    
    def __init__(self, default_user_id: Optional[str] = None, default_camera_id: Optional[str] = None, default_room_id: Optional[str] = None):
        self.default_user_id = default_user_id or str(uuid.uuid4())
        self.default_camera_id = default_camera_id or str(uuid.uuid4())
        self.default_room_id = default_room_id or str(uuid.uuid4())
        
        # Use PostgreSQL service directly
        self.postgresql_service = realtime_service
        self.default_camera_id = default_camera_id or str(uuid.uuid4())
        self.default_room_id = default_room_id or str(uuid.uuid4())
        
        # Start event listeners
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """Setup realtime event listeners"""
        try:
            # Listen for new event detections
            realtime_service.subscribe_to_events(
                'event_detections', 
                'INSERT', 
                self._handle_event_detection
            )
            
            # Listen for new alerts
            realtime_service.subscribe_to_events(
                'alerts',
                'INSERT',
                self._handle_alert
            )
            
            logger.info("Healthcare event listeners setup successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup event listeners: {e}")
    
    def _handle_event_detection(self, event_data: Dict[str, Any]):
        """Handle new event detection from realtime"""
        try:
            detection = event_data.get('new_data', {})
            event_type = detection.get('event_type')
            confidence = detection.get('confidence_score', 0.0)
            
            logger.info(f"🔔 Realtime Event: {event_type} detected with confidence {confidence:.2f}")
            
            # You can add custom handling here
            # For example: send notifications, update UI, etc.
            
        except Exception as e:
            logger.error(f"Error handling event detection: {e}")
    
    def _handle_alert(self, event_data: Dict[str, Any]):
        """Handle new alert from realtime"""
        try:
            alert = event_data.get('new_data', {})
            alert_type = alert.get('alert_type')
            severity = alert.get('severity')
            message = alert.get('alert_message')
            
            logger.info(f"🚨 Realtime Alert: {alert_type} [{severity}] - {message}")
            
            # You can add custom alert handling here
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def _create_event_response(self, event_id: Optional[str], status: str, event_type: str, 
                              confidence: float, camera_id: str, snapshot_timestamp: datetime) -> Dict[str, Any]:
        """Create standardized event response format for mobile realtime"""
        # Generate snapshot URL based on event_id
        image_url = f"https://healthcare-system.com/snapshots/{event_id or 'default'}.jpg"
        
        # Generate action message based on status and event type
        action = self._generate_action_message(status, event_type, confidence)
        
        return {
            "imageUrl": image_url,
            "status": status,  # normal|warning|danger
            "action": action,
            "time": snapshot_timestamp.isoformat()  # Time from snapshot creation
        }
    
    def _generate_action_message(self, status: str, event_type: str, confidence: float) -> str:
        """Generate action message based on status and event type"""
        if status == "normal":
            return "Không có gì bất thường"
        
        elif status == "warning":
            if event_type == "fall":
                return f"Phát hiện té ({confidence:.0%} confidence) - Cần theo dõi"
            elif event_type in ["abnormal_behavior", "seizure"]:
                return f"Phát hiện co giật ({confidence:.0%} confidence) - Cần theo dõi"
            else:
                return f"Phát hiện hoạt động bất thường ({confidence:.0%} confidence)"
        
        elif status == "danger":
            if event_type == "fall":
                return "⚠️ BÁO ĐỘNG NGUY HIỂM: Phát hiện té - Yêu cầu hỗ trợ gấp!"
            elif event_type in ["abnormal_behavior", "seizure"]:
                return "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!"
            else:
                return "🚨 BÁO ĐỘNG NGUY HIỂM: Yêu cầu hỗ trợ gấp!"
        
        return "Đang theo dõi..."

    def publish_fall_detection(self, confidence: float, bounding_boxes: List[Dict], 
                              context: Optional[Dict] = None, camera_id: Optional[str] = None, 
                              room_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Publish a fall detection event using PostgreSQL service"""
        try:
            # Extract IDs from context if provided, with fallback to defaults
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '3c0b0000-0000-4000-8000-000000000001'
            final_room_id = room_id or (context.get('room_id') if context else None) or '2d0a0000-0000-4000-8000-000000000001'
            final_user_id = user_id or (context.get('user_id') if context else None) or '34e92ef3-1300-40d0-a0e0-72989cf30121'
            
            current_time = datetime.now()
            
            # Fix confidence thresholds based on requirements
            if confidence >= 0.6:  # Changed to 60% for fall warning
                status = "warning"
            else:
                status = "normal"
            
            if confidence >= 0.8:  # 80% for fall danger
                status = "danger"
            
            # Create event data dict for PostgreSQL service
            event_data = {
                'event_type': 'fall',
                'description': f'Fall detected with {confidence:.1%} confidence',
                'detection_data': {
                    'algorithm': 'yolo_fall_detection',
                    'model_version': 'v1.0',
                    'detection_timestamp': current_time.isoformat()
                },
                'confidence': confidence,
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish to database and get event_id
            event_id = self.postgresql_service.publish_event_detection(event_data)
            
            # Create response format
            response = self._create_event_response(
                event_id=event_id,
                status=status,
                event_type="fall",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time
            )
            
            # Send realtime notification to mobile
            send_mobile_notification(response)
            
            return response
        except Exception as e:
            print(f"Error publishing fall detection: {e}")
            return {
                "imageUrl": "",
                "status": "normal", 
                "action": "Error processing fall detection",
                "time": datetime.now().isoformat()
            }

    def publish_seizure_detection(self, confidence: float, bounding_boxes: List[Dict],
                                 context: Optional[Dict] = None, camera_id: Optional[str] = None,
                                 room_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Publish a seizure detection event using PostgreSQL service"""
        try:
            # Extract IDs from context if provided, with fallback to defaults
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '3c0b0000-0000-4000-8000-000000000002'
            final_room_id = room_id or (context.get('room_id') if context else None) or '2d0a0000-0000-4000-8000-000000000002'
            final_user_id = user_id or (context.get('user_id') if context else None) or '361a335c-4f4d-4ed4-9e5c-ab7715d081b4'
            
            current_time = datetime.now()
            
            # Fix confidence thresholds based on requirements  
            if confidence >= 0.5:  # Changed to 50% for seizure warning
                status = "warning"
            else:
                status = "normal"
            
            if confidence >= 0.7:  # 70% for seizure danger
                status = "danger"
                
            # Create event data dict for PostgreSQL service
            event_data = {
                'event_type': 'abnormal_behavior',
                'description': f'Seizure activity detected with {confidence:.1%} confidence',
                'detection_data': {
                    'algorithm': 'seizure_detection',
                    'behavior_type': 'seizure',
                    'model_version': 'v1.0',
                    'detection_timestamp': current_time.isoformat()
                },
                'confidence': confidence,
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish to database and get event_id
            try:
                event_id = self.postgresql_service.publish_event_detection(event_data)
                if isinstance(event_id, dict):
                    event_id = event_id.get('event_id', 'unknown')
            except:
                event_id = 'fallback_id'
            
            # Create response format
            response = self._create_event_response(
                event_id=str(event_id) if event_id else None,
                status=status,
                event_type="seizure",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time
            )
                
            return response
        except Exception as e:
            print(f"Error publishing seizure detection: {e}")
            return {
                "imageUrl": "",
                "status": "normal",
                "action": "Error processing seizure detection", 
                "time": datetime.now().isoformat()
            }
    
    def publish_system_status(self, status: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Publish system status update
        
        Args:
            status: System status ('online', 'offline', 'error', 'maintenance')
            metrics: Additional metrics data
        """
        try:
            status_data = {
                'status': status,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'camera_id': self.default_camera_id,
                'metrics': metrics or {}
            }
            
            # You can implement broadcast functionality here if needed
            logger.info(f"System status: {status}")
            
        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
    
    def get_recent_events(self, limit: int = 10) -> list:
        """Get recent healthcare events"""
        return realtime_service.get_recent_events(limit)
    
    def close(self):
        """Close the event publisher"""
        realtime_service.close()

# Global publisher instance
healthcare_publisher = HealthcareEventPublisher()
