"""
Healthcare Event Publisher Service - Updated with Mobile Notifications
Integrates healthcare detection pipeline with database and mobile realtime notifications
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging

# Try to import services
try:
    from service.postgresql_healthcare_service import postgresql_service
    from service.mobile_realtime_notification_service import send_mobile_notification
    SERVICE_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Services not available: {e}")
    SERVICE_AVAILABLE = False
    
    # Mock functions
    def send_mobile_notification(event_response):
        print(f"📱 Mock mobile notification: {event_response}")
    
    class MockService:
        def publish_event_detection(self, data):
            return {
                'event_id': str(uuid.uuid4()),
                'snapshot_id': str(uuid.uuid4()),
                'created_at': datetime.now()
            }
    
    postgresql_service = MockService()

logger = logging.getLogger(__name__)

class HealthcareEventPublisher:
    """Healthcare event publisher with mobile realtime notifications and priority-based alerts"""
    
    def __init__(self):
        self.postgresql_service = postgresql_service
        # Import AlertStateManager here to avoid circular imports
        try:
            from service.alert_state_manager import get_alert_state_manager
            self.alert_manager = get_alert_state_manager(postgresql_service)
        except ImportError:
            logger.warning("AlertStateManager not available")
            self.alert_manager = None
        
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
        """Publish a fall detection event with priority-based alert management"""
        try:
            # Extract IDs from context if provided, with fallback to defaults
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '3c0b0000-0000-4000-8000-000000000001'
            final_room_id = room_id or (context.get('room_id') if context else None) or '2d0a0000-0000-4000-8000-000000000001'
            final_user_id = user_id or (context.get('user_id') if context else None) or '34e92ef3-1300-40d0-a0e0-72989cf30121'
            
            current_time = datetime.now()
            
            # Determine mobile status based on confidence thresholds
            status = "normal"
            if confidence >= 0.6:  # 60-79% = warning
                status = "warning"
            if confidence >= 0.8:  # >= 80% = danger
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
                'confidence_score': confidence,  # For alert manager
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish to database and get event_id
            db_result = self.postgresql_service.publish_event_detection(event_data)
            event_id = str(db_result.get('event_id', 'fallback_id')) if isinstance(db_result, dict) else str(db_result)
            
            # Add event_id to event_data for alert management
            event_data['event_id'] = event_id
            
            # Priority-based alert management
            alert_created = False
            if self.alert_manager and confidence >= 0.4:  # Only create alerts for meaningful detections
                # Check if should create alert based on priority
                if self.alert_manager.should_create_alert(event_data):
                    alert_id = self.alert_manager.create_alert(event_data)
                    if alert_id:
                        alert_created = True
                        logger.info(f"Created priority-based alert {alert_id} for fall detection (confidence: {confidence:.1%})")
                else:
                    logger.info(f"Skipped alert creation - lower priority than existing alerts")
            
            # Create response format
            response = self._create_event_response(
                event_id=event_id,
                status=status,
                event_type="fall",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time
            )
            
            # Add alert info to response
            response['alert_created'] = alert_created
            
            # Send realtime notification to mobile only if alert was created or confidence is high
            if alert_created or confidence >= 0.7:
                send_mobile_notification(response)
                logger.info(f"Sent mobile notification for fall detection (confidence: {confidence:.1%})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error publishing fall detection: {e}")
            return {
                "imageUrl": "",
                "status": "normal", 
                "action": "Error processing fall detection",
                "time": datetime.now().isoformat(),
                "alert_created": False
            }

    def publish_seizure_detection(self, confidence: float, bounding_boxes: List[Dict],
                                 context: Optional[Dict] = None, camera_id: Optional[str] = None,
                                 room_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Publish a seizure detection event with priority-based alert management"""
        try:
            # Extract IDs from context if provided, with fallback to defaults
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '3c0b0000-0000-4000-8000-000000000002'
            final_room_id = room_id or (context.get('room_id') if context else None) or '2d0a0000-0000-4000-8000-000000000002'
            final_user_id = user_id or (context.get('user_id') if context else None) or '361a335c-4f4d-4ed4-9e5c-ab7715d081b4'
            
            current_time = datetime.now()
            
            # Determine mobile status based on confidence thresholds  
            status = "normal"
            if confidence >= 0.5:  # 50-69% = warning
                status = "warning"
            if confidence >= 0.7:  # >= 70% = danger
                status = "danger"
                
            # Create event data dict for PostgreSQL service
            event_data = {
                'event_type': 'seizure',  # Changed to 'seizure' for better alert categorization
                'description': f'Seizure activity detected with {confidence:.1%} confidence',
                'detection_data': {
                    'algorithm': 'vsvig_seizure_detection',
                    'behavior_type': 'seizure',
                    'model_version': 'vsvig_v1.0',
                    'detection_timestamp': current_time.isoformat()
                },
                'confidence': confidence,
                'confidence_score': confidence,  # For alert manager
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish to database and get event_id
            db_result = self.postgresql_service.publish_event_detection(event_data)
            event_id = str(db_result.get('event_id', 'fallback_id')) if isinstance(db_result, dict) else str(db_result)
            
            # Add event_id to event_data for alert management
            event_data['event_id'] = event_id
            
            # Priority-based alert management (seizure has higher priority than fall)
            alert_created = False
            if self.alert_manager and confidence >= 0.3:  # Lower threshold for seizure as it's more critical
                # Check if should create alert based on priority
                if self.alert_manager.should_create_alert(event_data):
                    alert_id = self.alert_manager.create_alert(event_data)
                    if alert_id:
                        alert_created = True
                        logger.info(f"Created priority-based alert {alert_id} for seizure detection (confidence: {confidence:.1%})")
                else:
                    logger.info(f"Skipped seizure alert creation - lower priority than existing alerts")
            
            # Create response format
            response = self._create_event_response(
                event_id=event_id,
                status=status,
                event_type="seizure",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time
            )
            
            # Add alert info to response
            response['alert_created'] = alert_created
            
            # Send realtime notification to mobile only if alert was created or confidence is high
            if alert_created or confidence >= 0.6:
                send_mobile_notification(response)
                logger.info(f"Sent mobile notification for seizure detection (confidence: {confidence:.1%})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error publishing seizure detection: {e}")
            return {
                "imageUrl": "",
                "status": "normal",
                "action": "Error processing seizure detection", 
                "time": datetime.now().isoformat(),
                "alert_created": False
            }
    
    def publish_system_status(self, status: str, metrics: Optional[Dict[str, Any]] = None):
        """Publish system status update"""
        try:
            status_data = {
                'status': status,
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"System status: {status}")
            return status_data
            
        except Exception as e:
            logger.error(f"Error publishing system status: {e}")

# Global instance
healthcare_publisher = HealthcareEventPublisher()

if __name__ == "__main__":
    # Test the updated publisher
    print("🧪 Testing Updated Healthcare Event Publisher with Mobile Notifications")
    
    # Test fall detection
    fall_response = healthcare_publisher.publish_fall_detection(
        confidence=0.85,
        bounding_boxes=[{'x': 100, 'y': 200, 'width': 150, 'height': 200}],
        context={'test': True}
    )
    print(f"Fall Detection Response: {fall_response}")
    
    # Test seizure detection
    seizure_response = healthcare_publisher.publish_seizure_detection(
        confidence=0.75,
        bounding_boxes=[{'x': 120, 'y': 180, 'width': 180, 'height': 280}],
        context={'test': True}
    )
    print(f"Seizure Detection Response: {seizure_response}")
