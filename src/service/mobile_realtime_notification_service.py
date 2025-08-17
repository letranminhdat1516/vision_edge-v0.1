"""
Mobile Realtime Notification Service  
Handles sending healthcare events to mobile devices in realtime
Enhanced with FCM integration for real mobile notifications
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import asyncio

# Import FCM service for real mobile notifications
try:
    from service.fcm_notification_service import fcm_service
    FCM_AVAILABLE = True
    print("📱 FCM Service loaded for mobile notifications")
except ImportError as e:
    FCM_AVAILABLE = False
    print(f"⚠️ FCM Service not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileRealtimeNotificationService:
    """
    Service to handle real-time mobile notifications for healthcare events
    Sends structured JSON format for mobile consumption
    """
    
    def __init__(self):
        self.active_connections = []  # Simulated mobile connections
        self.notification_history = []
        self.is_running = False
        
    def start_service(self):
        """Start the mobile notification service"""
        self.is_running = True
        logger.info("🔔 Mobile Realtime Notification Service started")
        
    def stop_service(self):
        """Stop the mobile notification service"""
        self.is_running = False
        logger.info("🔇 Mobile Realtime Notification Service stopped")
        
    def send_healthcare_notification(self, event_response: Dict[str, Any]):
        """
        Send healthcare event notification to mobile devices with FCM integration
        
        Args:
            event_response: Healthcare event response with format:
            {
                "imageUrl": "https://healthcare-system.com/snapshots/{event_id}.jpg",
                "status": "normal|warning|danger", 
                "action": "Action description",
                "time": "2025-08-14T10:30:00.123456"
            }
        """
        if not self.is_running:
            logger.warning("Service not running, cannot send notification")
            return
            
        # Add notification metadata
        notification = {
            "id": f"notif_{int(time.time())}_{len(self.notification_history)}",
            "type": "healthcare_event",
            "priority": self._get_priority_from_status(event_response.get("status", "normal")),
            "data": event_response,
            "sent_at": datetime.now().isoformat()
        }
        
        # Store notification history
        self.notification_history.append(notification)
        
        # Send to mobile devices with FCM
        self._send_to_mobile_devices(notification)
        
        # Send real FCM notification for critical events
        if FCM_AVAILABLE and event_response.get("status") in ["warning", "danger"]:
            self._send_fcm_notification(event_response)
        
        # Log the notification
        logger.info(f"📱 Mobile notification sent: {notification['data']['status']} - {notification['data']['action']}")
        
    def _send_fcm_notification(self, event_response: Dict[str, Any]):
        """Send real FCM notification to mobile devices"""
        try:
            status = event_response.get("status", "normal")
            action = event_response.get("action", "Healthcare Event")
            
            # Map status to event type for FCM
            event_type_map = {
                "danger": "emergency",
                "warning": "warning",
                "normal": "info"
            }
            
            # Determine FCM event type and confidence
            if "fall" in action.lower():
                fcm_event_type = "fall"
                confidence = 0.85 if status == "danger" else 0.65
            elif "seizure" in action.lower():
                fcm_event_type = "seizure" 
                confidence = 0.90 if status == "danger" else 0.70
            else:
                fcm_event_type = "emergency"
                confidence = 0.75
            
            # Prepare additional data for FCM
            additional_data = {
                'alert_level': status,
                'emergency_type': fcm_event_type,
                'location': 'Healthcare Room',
                'image_url': event_response.get("imageUrl", ""),
                'detection_time': event_response.get("time", datetime.now().isoformat()),
                'mobile_notification_id': f"mobile_{int(time.time())}"
            }
            
            # Send FCM notification asynchronously
            def send_fcm_async():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    response = loop.run_until_complete(
                        fcm_service.send_emergency_alert(
                            event_type=fcm_event_type,
                            confidence=confidence,
                            user_tokens=None,  # Use tokens from .env
                            additional_data=additional_data
                        )
                    )
                    
                    loop.close()
                    
                    if response.get('success'):
                        success_count = response.get('success_count', 0)
                        total_tokens = response.get('total_tokens', 0)
                        logger.info(f"🔥 FCM Emergency Alert sent: {fcm_event_type} to {success_count}/{total_tokens} devices")
                    else:
                        logger.error(f"❌ FCM Alert failed: {response.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ FCM Async Error: {e}")
            
            # Run FCM in background thread
            fcm_thread = threading.Thread(target=send_fcm_async)
            fcm_thread.daemon = True
            fcm_thread.start()
            
        except Exception as e:
            logger.error(f"❌ FCM Notification Error: {e}")
        
    def _get_priority_from_status(self, status: str) -> str:
        """Get notification priority based on status"""
        priority_map = {
            "normal": "low",
            "warning": "medium", 
            "danger": "high"
        }
        return priority_map.get(status, "low")
        
    def _send_to_mobile_devices(self, notification: Dict[str, Any]):
        """
        Send notification to connected mobile devices
        In production, this would use WebSocket, FCM, or similar
        """
        status = notification["data"]["status"]
        action = notification["data"]["action"]
        
        # Simulate mobile device notification display
        print("\n" + "="*60)
        print("📱 MOBILE NOTIFICATION")
        print("="*60)
        
        if status == "danger":
            print("🚨 EMERGENCY ALERT 🚨")
            print(f"Status: {status.upper()}")
            print(f"Message: {action}")
            print(f"Image: {notification['data']['imageUrl']}")
            print(f"Time: {notification['data']['time']}")
            print("⚠️ IMMEDIATE ATTENTION REQUIRED")
            
        elif status == "warning":
            print("⚠️ WARNING ALERT")
            print(f"Status: {status.upper()}")
            print(f"Message: {action}")
            print(f"Image: {notification['data']['imageUrl']}")
            print(f"Time: {notification['data']['time']}")
            print("💡 Please monitor patient")
            
        else:  # normal
            print("ℹ️ Information Update")
            print(f"Status: {status.upper()}")
            print(f"Message: {action}")
            print(f"Time: {notification['data']['time']}")
            
        print("="*60)
        print()
        
        # Simulate sending to multiple devices
        for device_id in ["device_001", "device_002", "device_003"]:
            logger.info(f"Notification sent to {device_id}")
            
    def get_notification_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        return self.notification_history[-limit:] if limit > 0 else self.notification_history
        
    def get_active_connections_count(self) -> int:
        """Get number of active mobile connections"""
        return len(self.active_connections)

# Global instance
mobile_notification_service = MobileRealtimeNotificationService()

def start_mobile_notifications():
    """Start the mobile notification service"""
    mobile_notification_service.start_service()
    
def send_mobile_notification(event_response: Dict[str, Any]):
    """Send notification to mobile devices"""
    mobile_notification_service.send_healthcare_notification(event_response)
    
def stop_mobile_notifications():
    """Stop the mobile notification service"""
    mobile_notification_service.stop_service()

if __name__ == "__main__":
    # Test the service
    print("🧪 Testing Mobile Realtime Notification Service")
    
    # Start service
    start_mobile_notifications()
    
    # Test notifications with different statuses
    test_events = [
        {
            "imageUrl": "https://healthcare-system.com/snapshots/test_001.jpg",
            "status": "normal",
            "action": "Không có gì bất thường",
            "time": datetime.now().isoformat()
        },
        {
            "imageUrl": "https://healthcare-system.com/snapshots/test_002.jpg", 
            "status": "warning",
            "action": "Phát hiện té (65% confidence) - Cần theo dõi",
            "time": datetime.now().isoformat()
        },
        {
            "imageUrl": "https://healthcare-system.com/snapshots/test_003.jpg",
            "status": "danger", 
            "action": "🚨 BÁO ĐỘNG NGUY HIỂM: Phát hiện co giật - Yêu cầu hỗ trợ gấp!",
            "time": datetime.now().isoformat()
        }
    ]
    
    # Send test notifications
    for i, event in enumerate(test_events):
        print(f"\n--- Sending Test Notification {i+1} ---")
        send_mobile_notification(event)
        time.sleep(2)  # Delay between notifications
        
    # Show notification history
    print("\n📋 NOTIFICATION HISTORY:")
    history = mobile_notification_service.get_notification_history()
    for notif in history:
        print(f"  {notif['sent_at']} - {notif['priority']} - {notif['data']['status']}")
        
    # Stop service
    stop_mobile_notifications()
    print("\n✅ Mobile notification service test completed")
