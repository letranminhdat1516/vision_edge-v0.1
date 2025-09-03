"""
Healthcare Event Publisher Service with Priority-Based Alert System
Integrates healthcare detection pipeline with Supabase realtime system
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging
from enum import Enum

# Import image caption service for intelligent action generation
try:
    from service.image_caption_service import get_professional_caption_pipeline
    IMAGE_CAPTION_AVAILABLE = True
except ImportError:
    IMAGE_CAPTION_AVAILABLE = False
    logging.warning("Image caption service not available - using static action messages")

# Priority-based alert system imports
class AlertPriority(Enum):
    RESOLVED = 0
    ACKNOWLEDGED_LOW = 1
    ACKNOWLEDGED_MEDIUM = 2
    ACTIVE_LOW = 3
    ACTIVE_MEDIUM = 4
    ACTIVE_HIGH = 5
    ACTIVE_CRITICAL = 6

# Try to import Supabase service, fallback to mock
try:
    from service.postgresql_healthcare_service import postgresql_service as realtime_service
    from service.mobile_realtime_notification_service import MobileRealtimeNotificationService
    
    # Initialize mobile notification service
    mobile_notification_service = MobileRealtimeNotificationService()
    mobile_notification_service.start_service()
    
    def send_mobile_notification(event_response): 
        """Send mobile notification through service"""
        mobile_notification_service.send_healthcare_notification(event_response)
    
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
    """Service for publishing healthcare events with priority-based alert system"""
    
    # Confidence thresholds for severity mapping
    SEVERITY_THRESHOLDS = {
        'fall': {'high': 0.60, 'medium': 0.40, 'low': 0.20},  # Remove 'critical'
        'seizure': {'high': 0.50, 'medium': 0.30, 'low': 0.10}  # Remove 'critical'
    }
    
    # Notification thresholds (send notification even if no alert created)
    NOTIFICATION_THRESHOLDS = {
        'fall': 0.70,
        'seizure': 0.60
    }
    
    def __init__(self, default_user_id: Optional[str] = None, default_camera_id: Optional[str] = None, default_room_id: Optional[str] = None):
        self.default_user_id = default_user_id or str(uuid.uuid4())
        self.default_camera_id = default_camera_id or str(uuid.uuid4())
        self.default_room_id = default_room_id or str(uuid.uuid4())
        
        # Use PostgreSQL service directly
        self.postgresql_service = realtime_service
        
        # Start event listeners
        self._setup_event_listeners()
    
    def _map_confidence_to_severity(self, confidence: float, event_type: str) -> str:
        """Map confidence score to database severity"""
        thresholds = self.SEVERITY_THRESHOLDS.get(event_type, self.SEVERITY_THRESHOLDS['fall'])
        
        if confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _map_status_for_mobile(self, severity: str) -> str:
        """Map database severity to mobile status format"""
        severity_to_mobile = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'normal'
        }
        return severity_to_mobile.get(severity, 'normal')
    
    def _calculate_priority_level(self, severity: str, alert_status: str) -> int:
        """Calculate priority level for alert comparison"""
        base_priority = {
            'high': 4,
            'medium': 3,
            'low': 2
        }.get(severity, 1)
        
        # Reduce priority for acknowledged/resolved alerts
        if alert_status == 'acknowledged':
            return max(1, base_priority - 2)
        elif alert_status == 'resolved':
            return 0
        
        return base_priority
    
    def _get_highest_priority_alert(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current highest priority active alert for user"""
        try:
            conn = self.postgresql_service.get_connection()
            if not conn:
                return None
                
            with conn.cursor() as cursor:
                # Get active alerts ordered by priority
                cursor.execute("""
                    SELECT a.*, 
                           CASE a.severity
                               WHEN 'high' THEN 4  
                               WHEN 'medium' THEN 3
                               WHEN 'low' THEN 2
                               ELSE 1
                           END as priority_level
                    FROM alerts a
                    WHERE a.user_id = %s AND a.status = 'active'
                    ORDER BY priority_level DESC, a.created_at DESC
                    LIMIT 1
                """, (user_id,))
                
                result = cursor.fetchone()
                self.postgresql_service.return_connection(conn)
                
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Error getting highest priority alert: {e}")
            if conn:
                self.postgresql_service.return_connection(conn)
            return None
    
    def _should_create_alert(self, confidence: float, event_type: str, user_id: str) -> tuple[bool, str]:
        """Determine if alert should be created based on priority comparison"""
        # Calculate new event priority
        severity = self._map_confidence_to_severity(confidence, event_type)
        new_priority = self._calculate_priority_level(severity, 'active')
        
        # Get highest existing priority
        highest_alert = self._get_highest_priority_alert(user_id)
        if highest_alert:
            current_max_priority = highest_alert.get('priority_level', 0)
            
            # Only create alert if new priority is higher or equal
            should_create = new_priority >= current_max_priority
            reason = f"Priority {new_priority} vs current max {current_max_priority}"
        else:
            # No existing alerts, create if not low priority
            should_create = new_priority > 2  # Skip low priority if no existing alerts
            reason = f"No existing alerts, priority {new_priority}"
        
        logger.info(f"Alert decision: {should_create} - {reason}")
        return should_create, severity
    
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
    
    def _get_recent_alert_image_path(self, event_type: str, confidence: float) -> Optional[str]:
        """Try to find the most recent alert image for intelligent action generation"""
        try:
            import os
            import glob
            from pathlib import Path
            
            # Common alert image folders
            possible_folders = [
                Path(__file__).parent.parent / "examples" / "data" / "saved_frames" / "alerts",
                Path(__file__).parent / "examples" / "data" / "saved_frames" / "alerts", 
                Path("examples") / "data" / "saved_frames" / "alerts",
                Path("src") / "examples" / "data" / "saved_frames" / "alerts",
                Path("data") / "saved_frames" / "alerts"
            ]
            
            for alerts_folder in possible_folders:
                if alerts_folder.exists():
                    # Look for images with matching event type and similar confidence
                    pattern = f"*{event_type}*conf_{confidence:.2f}*.jpg"
                    matching_files = list(alerts_folder.glob(pattern))
                    
                    if not matching_files:
                        # Try broader pattern
                        pattern = f"*{event_type}*.jpg"
                        matching_files = list(alerts_folder.glob(pattern))
                    
                    if matching_files:
                        # Return most recent file
                        latest_file = max(matching_files, key=lambda p: p.stat().st_ctime)
                        logger.info(f"📸 Found recent alert image: {latest_file.name}")
                        return str(latest_file)
                        
            logger.debug("📸 No recent alert image found for intelligent action generation")
            return None
            
        except Exception as e:
            logger.debug(f"Failed to find alert image: {e}")
            return None

    def _create_event_response(self, event_id: Optional[str], status: str, event_type: str, 
                              confidence: float, camera_id: str, snapshot_timestamp: datetime,
                              image_path: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized event response format for mobile realtime"""
        # Generate snapshot URL based on event_id
        image_url = f"https://healthcare-system.com/snapshots/{event_id or 'default'}.jpg"
        
        # Try to find recent alert image if not provided
        if not image_path and status in ["warning", "danger"]:
            image_path = self._get_recent_alert_image_path(event_type, confidence)
        
        # Generate action message based on status, event type, and optionally image content
        action = self._generate_action_message(status, event_type, confidence, image_path)
        
        return {
            "imageUrl": image_url,
            "status": status,  # normal|warning|danger
            "action": action,
            "time": snapshot_timestamp.isoformat()  # Time from snapshot creation
        }
    
    def _generate_action_message(self, status: str, event_type: str, confidence: float, 
                                image_path: Optional[str] = None) -> str:
        """Generate action message based on status, event type, and optionally image content"""
        
        # Try to generate intelligent action from image content first
        if image_path and IMAGE_CAPTION_AVAILABLE:
            try:
                intelligent_action = self._generate_intelligent_action(image_path, status, event_type, confidence)
                if intelligent_action:
                    return intelligent_action
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate intelligent action: {e}")
        
        # Fallback to static action messages
        return self._generate_static_action_message(status, event_type, confidence)
    
    def _generate_intelligent_action(self, image_path: str, status: str, event_type: str, confidence: float) -> Optional[str]:
        """Generate intelligent action message using BLIP + Translation pipeline"""
        try:
            if not IMAGE_CAPTION_AVAILABLE:
                return None
                
            # Get image caption pipeline
            caption_pipeline = get_professional_caption_pipeline()
            
            # Generate Vietnamese caption from image
            vietnamese_caption, metadata = caption_pipeline.generate_professional_caption(image_path)
            
            if not metadata.get("success", False):
                logger.warning("Image captioning failed, using static action")
                return None
            
            # Enhance caption with emergency context based on status and confidence
            enhanced_action = self._enhance_caption_with_emergency_context(
                vietnamese_caption, status, event_type, confidence
            )
            
            logger.info(f"🤖 Generated intelligent action: {enhanced_action}")
            logger.info(f"📸 Based on image content: {metadata.get('english_caption', 'N/A')}")
            
            return enhanced_action
            
        except Exception as e:
            logger.error(f"❌ Intelligent action generation failed: {e}")
            return None
    
    def _enhance_caption_with_emergency_context(self, base_caption: str, status: str, 
                                              event_type: str, confidence: float) -> str:
        """Enhance image caption with emergency context and urgency"""
        
        # Remove existing alert markers from caption to avoid duplication
        cleaned_caption = base_caption.replace("⚠️ Cảnh báo:", "").replace("🚨 Cảnh báo:", "").strip()
        
        # Generate emergency context based on status
        if status == "danger":
            if event_type == "fall":
                emergency_prefix = "🚨 KHẨN CẤP - TÉ NGÃ:"
                urgency_suffix = f" - YÊU CẦU HỖ TRỢ NGAY LẬP TỨC! (Tin cậy: {confidence:.0%})"
            elif event_type in ["abnormal_behavior", "seizure"]:
                emergency_prefix = "🆘 KHẨN CẤP - CO GIẬT:"
                urgency_suffix = f" - CẦN ĐIỀU TRỊ Y TẾ NGAY! (Tin cậy: {confidence:.0%})"
            else:
                emergency_prefix = "🚨 TÌNH HUỐNG KHẨN CẤP:"
                urgency_suffix = f" - Cần hỗ trợ gấp! (Tin cậy: {confidence:.0%})"
                
        elif status == "warning":
            if event_type == "fall":
                emergency_prefix = "⚠️ CẢNH BÁO TÉ NGÃ:"
                urgency_suffix = f" - Cần theo dõi và kiểm tra (Tin cậy: {confidence:.0%})"
            elif event_type in ["abnormal_behavior", "seizure"]:
                emergency_prefix = "⚠️ CẢNH BÁO BẤT THƯỜNG:"
                urgency_suffix = f" - Cần quan sát chặt chẽ (Tin cậy: {confidence:.0%})"
            else:
                emergency_prefix = "⚠️ CẢNH BÁO:"
                urgency_suffix = f" - Cần theo dõi (Tin cậy: {confidence:.0%})"
                
        else:  # normal
            return f"✅ BÌNH THƯỜNG: {cleaned_caption}"
        
        # Combine all parts
        enhanced_action = f"{emergency_prefix} {cleaned_caption}{urgency_suffix}"
        
        return enhanced_action
    
    def _generate_static_action_message(self, status: str, event_type: str, confidence: float) -> str:
        """Generate static action message (original implementation)"""
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
        """Publish fall detection with priority-based alert system"""
        try:
            # Extract IDs from context if provided, with fallback to real database IDs
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '22222222-2222-2222-2222-222222222201'
            final_room_id = room_id or (context.get('room_id') if context else None) or '11111111-1111-1111-1111-111111111101'
            final_user_id = user_id or (context.get('user_id') if context else None) or '00000000-0000-0000-0000-0000000000aa'
            
            current_time = datetime.now()
            
            # Determine if alert should be created and get severity
            should_create_alert, severity = self._should_create_alert(confidence, 'fall', final_user_id)
            
            # Always create event detection (for audit trail)
            event_data = {
                'event_type': 'fall',
                'description': f'Fall detected with {confidence:.1%} confidence',
                'detection_data': {
                    'algorithm': 'yolo_fall_detection',
                    'model_version': 'v1.0',
                    'detection_timestamp': current_time.isoformat(),
                    'severity': severity
                },
                'confidence': confidence,
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish event to database
            if hasattr(self.postgresql_service, 'publish_event_detection'):
                event_result = self.postgresql_service.publish_event_detection(event_data)
                event_id = event_result.get('event_id') if isinstance(event_result, dict) else str(event_result)
            else:
                event_id = str(uuid.uuid4())  # Fallback for mock mode
            
            # Create mobile response format
            mobile_status = self._map_status_for_mobile(severity)
            
            # Try to find alert image for intelligent action generation
            alert_image_path = self._get_recent_alert_image_path('fall', confidence)
            
            response = self._create_event_response(
                event_id=event_id,
                status=mobile_status,
                event_type="fall",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time,
                image_path=alert_image_path
            )
            
            # Add priority system metadata
            response['alert_created'] = should_create_alert
            response['severity'] = severity
            response['priority_level'] = self._calculate_priority_level(severity, 'active')
            
            # Create alert only if priority check passed
            if should_create_alert and hasattr(self.postgresql_service, 'publish_alert'):
                alert_data = {
                    'event_id': event_id,
                    'user_id': final_user_id,
                    'alert_type': 'emergency',  # Use valid enum value
                    'severity': severity,
                    'message': self._generate_action_message(mobile_status, 'fall', confidence, alert_image_path),
                    'alert_data': {
                        'confidence': float(confidence),  # Ensure JSON serializable
                        'bounding_boxes': bounding_boxes,
                        'detection_type': context.get('detection_type', 'direct') if context else 'direct'
                    }
                }
                self.postgresql_service.publish_alert(alert_data)
            
            # Send mobile notification based on conditions
            should_notify = (
                should_create_alert or  # Alert was created
                confidence >= self.NOTIFICATION_THRESHOLDS['fall']  # High confidence
            )
            
            if should_notify:
                send_mobile_notification(response)
                logger.info(f"📱 Fall notification sent: {mobile_status} - confidence {confidence:.2f}")
            else:
                logger.info(f"📵 Fall notification skipped: priority filter")
            
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
        """Publish seizure detection with priority-based alert system"""
        try:
            # Extract IDs from context if provided, with fallback to real database IDs
            final_camera_id = camera_id or (context.get('camera_id') if context else None) or '22222222-2222-2222-2222-222222222202'
            final_room_id = room_id or (context.get('room_id') if context else None) or '11111111-1111-1111-1111-111111111102'
            final_user_id = user_id or (context.get('user_id') if context else None) or '00000000-0000-0000-0000-0000000000aa'
            
            current_time = datetime.now()
            
            # Determine if alert should be created and get severity
            should_create_alert, severity = self._should_create_alert(confidence, 'seizure', final_user_id)
                
            # Always create event detection (for audit trail)
            event_data = {
                'event_type': 'abnormal_behavior',
                'description': f'Seizure activity detected with {confidence:.1%} confidence',
                'detection_data': {
                    'algorithm': 'seizure_detection',
                    'behavior_type': 'seizure',
                    'model_version': 'v1.0',
                    'detection_timestamp': current_time.isoformat(),
                    'severity': severity
                },
                'confidence': confidence,
                'bounding_boxes': bounding_boxes,
                'context': context or {},
                'camera_id': final_camera_id,
                'room_id': final_room_id,
                'user_id': final_user_id
            }
            
            # Publish event to database
            if hasattr(self.postgresql_service, 'publish_event_detection'):
                event_result = self.postgresql_service.publish_event_detection(event_data)
                event_id = event_result.get('event_id') if isinstance(event_result, dict) else str(event_result)
            else:
                event_id = str(uuid.uuid4())  # Fallback for mock mode
            
            # Create mobile response format
            mobile_status = self._map_status_for_mobile(severity)
            
            # Try to find alert image for intelligent action generation
            alert_image_path = self._get_recent_alert_image_path('seizure', confidence)
            
            response = self._create_event_response(
                event_id=event_id,
                status=mobile_status,
                event_type="seizure",
                confidence=confidence,
                camera_id=final_camera_id,
                snapshot_timestamp=current_time,
                image_path=alert_image_path
            )
            
            # Add priority system metadata
            response['alert_created'] = should_create_alert
            response['severity'] = severity
            response['priority_level'] = self._calculate_priority_level(severity, 'active')
            
            # Create alert only if priority check passed
            if should_create_alert and hasattr(self.postgresql_service, 'publish_alert'):
                alert_data = {
                    'event_id': event_id,
                    'user_id': final_user_id,
                    'alert_type': 'warning',  # Use valid enum value
                    'severity': severity,
                    'message': self._generate_action_message(mobile_status, 'seizure', confidence, alert_image_path),
                    'alert_data': {
                        'confidence': float(confidence),  # Ensure JSON serializable
                        'bounding_boxes': bounding_boxes,
                        'detection_type': context.get('detection_type', 'confirmation') if context else 'confirmation'
                    }
                }
                self.postgresql_service.publish_alert(alert_data)
            
            # Send mobile notification based on conditions
            should_notify = (
                should_create_alert or  # Alert was created
                confidence >= self.NOTIFICATION_THRESHOLDS['seizure']  # High confidence
            )
            
            if should_notify:
                send_mobile_notification(response)
                logger.info(f"📱 Seizure notification sent: {mobile_status} - confidence {confidence:.2f}")
            else:
                logger.info(f"📵 Seizure notification skipped: priority filter")
                
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
    
    def get_recent_events(self, limit: int = 10) -> list:
        """Get recent healthcare events"""
        try:
            if hasattr(realtime_service, 'get_recent_events'):
                return realtime_service.get_recent_events(limit)
            else:
                return []  # Fallback for mock mode
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
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

    def close(self):
        """Close the event publisher"""
        try:
            if hasattr(realtime_service, 'close'):
                realtime_service.close()
        except Exception as e:
            logger.error(f"Error closing event publisher: {e}")

# Global publisher instance
healthcare_publisher = HealthcareEventPublisher()
