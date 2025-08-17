"""
Firebase Cloud Messaging (FCM) Notification Service
Tích hợp FCM cho hệ thống Healthcare Emergency Alerts
"""

import firebase_admin
from firebase_admin import credentials, messaging
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class FCMNotificationService:
    """
    Firebase Cloud Messaging service cho healthcare emergency alerts
    Hỗ trợ gửi notification real-time cho các sự kiện nguy hiểm
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize FCM service
        
        Args:
            credentials_path: Path to Firebase credentials JSON file
        """
        self.app = None
        self.initialized = False
        
        # Get settings from environment variables
        self.project_id = os.getenv("FIREBASE_PROJECT_ID")
        self.default_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "src/config/firebase-adminsdk.json")
        self.default_topic = os.getenv("FCM_DEFAULT_TOPIC", "emergency_alerts")
        self.emergency_sound = os.getenv("FCM_EMERGENCY_SOUND", "emergency.wav")
        self.warning_sound = os.getenv("FCM_WARNING_SOUND", "warning.wav")
        self.enable_notifications = os.getenv("FCM_ENABLE_NOTIFICATIONS", "true").lower() == "true"
        self.log_level = os.getenv("FCM_LOG_LEVEL", "INFO")
        
        # Load FCM tokens from environment
        self.device_tokens = self._load_tokens_from_env("FCM_DEVICE_TOKENS")
        self.caregiver_tokens = self._load_tokens_from_env("FCM_CAREGIVER_TOKENS") 
        self.emergency_tokens = self._load_tokens_from_env("FCM_EMERGENCY_TOKENS")
        
        # All available tokens
        self.all_tokens = list(set(self.device_tokens + self.caregiver_tokens + self.emergency_tokens))
        
        # Setup logging
        self._setup_logging()
        
        # Possible paths để tìm credentials (ưu tiên .env config)
        possible_paths = [
            credentials_path,
            self.default_credentials_path,
            "firebase-adminsdk.json",
            "config/firebase-adminsdk.json", 
            "src/config/firebase-adminsdk.json",
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    self._initialize_firebase(path)
                    logger.info(f"✅ FCM initialized with credentials: {path}")
                    break
                except Exception as e:
                    logger.error(f"❌ Failed to initialize FCM with {path}: {e}")
                    continue
        
        if not self.initialized:
            logger.warning("⚠️ FCM not initialized - emergency notifications will be logged only")
    
    def _load_tokens_from_env(self, env_var: str) -> List[str]:
        """Load FCM tokens from environment variable"""
        tokens_str = os.getenv(env_var, "")
        if not tokens_str or tokens_str.strip() == "":
            return []
        
        # Split by comma and clean up
        tokens = [token.strip() for token in tokens_str.split(",") if token.strip()]
        
        # Filter out placeholder tokens
        real_tokens = [token for token in tokens if not token.startswith(("REAL_TOKEN", "CAREGIVER_TOKEN", "EMERGENCY_TOKEN", "EXAMPLE", "demo_", "mock_"))]
        
        if real_tokens:
            logger.info(f"✅ Loaded {len(real_tokens)} real FCM tokens from {env_var}")
        else:
            logger.warning(f"⚠️ No real FCM tokens found in {env_var} (found {len(tokens)} placeholder tokens)")
            
        return real_tokens
    
    def _setup_logging(self):
        """Setup logging for FCM service"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def get_all_available_tokens(self) -> List[str]:
        """Get all available FCM tokens"""
        return self.all_tokens.copy()
    
    def get_emergency_tokens(self) -> List[str]:
        """Get emergency-specific tokens"""
        return self.emergency_tokens.copy() if self.emergency_tokens else self.all_tokens.copy()
    
    def get_caregiver_tokens(self) -> List[str]:
        """Get caregiver-specific tokens"""
        return self.caregiver_tokens.copy() if self.caregiver_tokens else self.all_tokens.copy()
    
    def _initialize_firebase(self, credentials_path: str):
        """Initialize Firebase Admin SDK"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            self.app = firebase_admin.initialize_app(cred)
        else:
            self.app = firebase_admin.get_app()
        
        self.initialized = True
    
    async def send_emergency_alert(self, 
                                 event_type: str, 
                                 confidence: float,
                                 user_tokens: Optional[List[str]] = None,
                                 additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Gửi emergency alert cho các sự kiện nguy hiểm
        
        Args:
            event_type: Loại sự kiện ('fall', 'seizure')
            confidence: Độ tin cậy của detection (0.0-1.0)
            user_tokens: List FCM tokens (if None, use tokens from .env)
            additional_data: Dữ liệu bổ sung
            
        Returns:
            dict: Response từ FCM service
        """
        # Use tokens from .env if not provided
        if user_tokens is None:
            if event_type.lower() in ['fall', 'seizure']:
                user_tokens = self.get_emergency_tokens()
            else:
                user_tokens = self.get_all_available_tokens()
        
        # Check if notifications are enabled
        if not self.enable_notifications:
            return self._log_disabled_notification(event_type, confidence, user_tokens)
        
        if not self.initialized:
            return self._log_mock_notification(event_type, confidence, user_tokens)
        
        if not user_tokens:
            logger.warning("⚠️ No FCM tokens available for notification")
            return {
                "success": False,
                "error": "No FCM tokens available",
                "message_id": None
            }
        
        try:
            # Tạo message content dựa trên event type
            title, body, priority = self._create_message_content(event_type, confidence)
            
            # Prepare message data - ensure all values are strings
            message_data = {
                "event_type": str(event_type),
                "confidence": f"{confidence:.2f}",
                "timestamp": datetime.now().isoformat(),
                "priority": str(priority),
                "alert_id": f"{event_type}_{int(datetime.now().timestamp())}"
            }
            
            # Add additional data nếu có - convert to strings
            if additional_data:
                for key, value in additional_data.items():
                    message_data[key] = str(value)
            
            # Create FCM message
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=message_data,
                android=messaging.AndroidConfig(
                    priority=priority,
                    notification=messaging.AndroidNotification(
                        channel_id="emergency_alerts",
                        sound=self.emergency_sound if priority == "high" else self.warning_sound,
                        color="#FF0000" if event_type == "fall" else "#FF8C00",
                        tag=f"{event_type}_alert"
                    )
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(title=title, body=body),
                            sound=self.emergency_sound if priority == "high" else self.warning_sound,
                            badge=1
                        )
                    )
                ),
                tokens=user_tokens
            )
            
            # Send message
            response = messaging.send_each_for_multicast(message)
            
            # Log results
            success_count = response.success_count
            failure_count = response.failure_count
            
            logger.info(f"🚨 FCM Emergency Alert sent: {event_type}")
            logger.info(f"   ✅ Success: {success_count}/{len(user_tokens)}")
            logger.info(f"   ❌ Failed: {failure_count}/{len(user_tokens)}")
            
            if response.responses:
                for idx, resp in enumerate(response.responses):
                    if not resp.success:
                        logger.error(f"   Token {idx} failed: {resp.exception}")
            
            return {
                "success": True,
                "message_id": f"{event_type}_{int(datetime.now().timestamp())}",
                "success_count": success_count,
                "failure_count": failure_count,
                "total_tokens": len(user_tokens),
                "responses": response.responses
            }
            
        except Exception as e:
            logger.error(f"❌ FCM send failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message_id": None
            }
    
    def _create_message_content(self, event_type: str, confidence: float) -> tuple[str, str, str]:
        """Tạo nội dung message dựa trên event type"""
        
        if event_type.lower() == "fall":
            if confidence >= 0.8:
                title = "🚨 FALL DETECTED - CRITICAL"
                body = f"High confidence fall detected ({confidence:.1%}). Immediate attention required!"
                priority = "high"
            elif confidence >= 0.6:
                title = "⚠️ FALL DETECTED"
                body = f"Possible fall detected ({confidence:.1%}). Please check patient status."
                priority = "normal"
            else:
                title = "📱 Fall Warning"
                body = f"Low confidence fall alert ({confidence:.1%}). Monitor patient."
                priority = "normal"
                
        elif event_type.lower() == "seizure":
            if confidence >= 0.7:
                title = "🚨 SEIZURE DETECTED - CRITICAL"
                body = f"High confidence seizure detected ({confidence:.1%}). Emergency response needed!"
                priority = "high"
            elif confidence >= 0.5:
                title = "⚠️ SEIZURE DETECTED"
                body = f"Possible seizure detected ({confidence:.1%}). Check patient immediately."
                priority = "normal"
            else:
                title = "📱 Seizure Warning"
                body = f"Unusual movement detected ({confidence:.1%}). Monitor patient."
                priority = "normal"
                
        else:
            title = f"🏥 Health Alert - {event_type.title()}"
            body = f"Health event detected ({confidence:.1%}). Please check patient."
            priority = "normal"
        
        return title, body, priority
    
    def _log_disabled_notification(self, event_type: str, confidence: float, user_tokens: List[str]) -> Dict[str, Any]:
        """Log disabled notification"""
        title, body, priority = self._create_message_content(event_type, confidence)
        
        logger.info("🔕 FCM NOTIFICATIONS DISABLED (FCM_ENABLE_NOTIFICATIONS=false):")
        logger.info(f"   Title: {title}")
        logger.info(f"   Body: {body}")
        logger.info(f"   Priority: {priority}")
        logger.info(f"   Would send to: {len(user_tokens)} devices")
        
        return {
            "success": True,
            "disabled": True,
            "message_id": f"disabled_{event_type}_{int(datetime.now().timestamp())}",
            "success_count": len(user_tokens),
            "failure_count": 0,
            "total_tokens": len(user_tokens)
        }
    
    def _log_mock_notification(self, event_type: str, confidence: float, user_tokens: List[str]) -> Dict[str, Any]:
        """Log mock notification khi FCM không available"""
        title, body, priority = self._create_message_content(event_type, confidence)
        
        logger.warning("📱 MOCK FCM NOTIFICATION (FCM not initialized):")
        logger.warning(f"   Title: {title}")
        logger.warning(f"   Body: {body}")
        logger.warning(f"   Priority: {priority}")
        logger.warning(f"   Tokens: {len(user_tokens)} recipients")
        
        return {
            "success": True,
            "mock": True,
            "message_id": f"mock_{event_type}_{int(datetime.now().timestamp())}",
            "success_count": len(user_tokens),
            "failure_count": 0,
            "total_tokens": len(user_tokens)
        }
    
    async def send_topic_notification(self, 
                                    topic: str,
                                    title: str, 
                                    body: str,
                                    data: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Gửi notification tới topic (cho nhiều users cùng lúc)
        
        Args:
            topic: Topic name (e.g., "emergency_alerts", "room_123")
            title: Notification title
            body: Notification body
            data: Additional data
            
        Returns:
            dict: FCM response
        """
        if not self.initialized:
            logger.warning(f"📱 MOCK TOPIC NOTIFICATION: {topic} - {title}")
            return {"success": True, "mock": True, "topic": topic}
        
        try:
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                topic=topic
            )
            
            response = messaging.send(message)
            logger.info(f"📢 Topic notification sent to '{topic}': {response}")
            
            return {
                "success": True,
                "message_id": response,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"❌ Topic notification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    
    def subscribe_to_topic(self, tokens: List[str], topic: str) -> Dict[str, Any]:
        """Subscribe device tokens to topic"""
        if not self.initialized:
            logger.warning(f"📱 MOCK TOPIC SUBSCRIPTION: {len(tokens)} tokens to '{topic}'")
            return {"success": True, "mock": True}
        
        try:
            response = messaging.subscribe_to_topic(tokens, topic)
            logger.info(f"📢 Subscribed {len(tokens)} tokens to topic '{topic}'")
            return {
                "success": True,
                "success_count": response.success_count,
                "failure_count": response.failure_count
            }
            
        except Exception as e:
            logger.error(f"❌ Topic subscription failed: {e}")
            return {"success": False, "error": str(e)}

# Global FCM service instance
fcm_service = FCMNotificationService()

# Convenience functions
async def send_fall_alert(confidence: float, user_tokens: List[str], additional_data: Dict = None):
    """Gửi fall detection alert"""
    return await fcm_service.send_emergency_alert("fall", confidence, user_tokens, additional_data)

async def send_seizure_alert(confidence: float, user_tokens: List[str], additional_data: Dict = None):
    """Gửi seizure detection alert"""
    return await fcm_service.send_emergency_alert("seizure", confidence, user_tokens, additional_data)
