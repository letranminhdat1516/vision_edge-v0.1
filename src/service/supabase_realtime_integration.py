"""
Supabase Realtime Integration for Healthcare Events
This integrates the healthcare event publisher with Supabase realtime channels
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    logger.warning("Supabase library not installed. Run: pip install supabase")
    SUPABASE_AVAILABLE = False
    Client = None

class SupabaseRealtimeIntegration:
    """
    Integrates healthcare events with Supabase realtime channels for mobile consumption
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase realtime integration
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon key
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Optional[Client] = None
        self.is_connected = False
        
        if SUPABASE_AVAILABLE:
            self._initialize_client()
        else:
            logger.error("Supabase not available. Please install: pip install supabase")
    
    def _initialize_client(self):
        """Initialize Supabase client"""
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            self.is_connected = True
            logger.info("✅ Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.is_connected = False
    
    def publish_realtime_event(self, event_data: Dict[str, Any], channel_name: str = "healthcare_events"):
        """
        Publish healthcare event to Supabase realtime channel
        
        Args:
            event_data: Event data to publish
            channel_name: Supabase realtime channel name
        """
        if not self.is_connected:
            logger.warning("Supabase not connected, cannot publish realtime event")
            return False
            
        try:
            # Insert event into database (this will trigger realtime notifications)
            response = self.client.table('event_detections').insert(event_data).execute()
            
            # Also send to realtime channel directly (optional)
            # self.client.realtime.send(channel_name, 'healthcare_event', event_data)
            
            logger.info(f"📡 Realtime event published to channel '{channel_name}': {event_data.get('event_type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish realtime event: {e}")
            return False
    
    def create_mobile_notification_payload(self, event_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create mobile-optimized notification payload
        
        Args:
            event_response: Healthcare event response
            
        Returns:
            Mobile notification payload
        """
        return {
            "notification_id": f"healthcare_{int(datetime.now().timestamp())}",
            "type": "healthcare_event",
            "priority": self._get_priority_from_status(event_response.get("status", "normal")),
            "data": event_response,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (datetime.now().timestamp() + 3600) * 1000,  # 1 hour in milliseconds
        }
    
    def _get_priority_from_status(self, status: str) -> str:
        """Get notification priority from status"""
        priority_map = {
            "normal": "low",
            "warning": "medium",
            "danger": "high"
        }
        return priority_map.get(status, "low")
    
    def setup_mobile_push_notifications(self, device_tokens: list = None):
        """
        Setup push notifications for mobile devices
        (This would integrate with FCM/APNS in production)
        
        Args:
            device_tokens: List of mobile device tokens
        """
        logger.info("📱 Mobile push notifications setup (mock implementation)")
        # In production, you would:
        # 1. Store device tokens in database
        # 2. Setup FCM/APNS integration
        # 3. Send push notifications to registered devices
        pass
    
    def test_realtime_connection(self):
        """Test the realtime connection"""
        if not self.is_connected:
            logger.error("❌ Supabase not connected")
            return False
            
        try:
            # Test database connection
            response = self.client.table('event_detections').select("*").limit(1).execute()
            logger.info("✅ Supabase realtime connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False

class HealthcareRealtimePublisher:
    """
    Enhanced healthcare event publisher with Supabase realtime integration
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize healthcare realtime publisher
        
        Args:
            supabase_url: Supabase project URL (from environment if not provided)
            supabase_key: Supabase anon key (from environment if not provided)
        """
        # Get Supabase credentials from environment or parameters
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("⚠️ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
            self.supabase_integration = None
        else:
            self.supabase_integration = SupabaseRealtimeIntegration(self.supabase_url, self.supabase_key)
    
    def publish_healthcare_event_with_realtime(self, event_response: Dict[str, Any]):
        """
        Publish healthcare event to database and send realtime notification
        
        Args:
            event_response: Healthcare event response in mobile format:
            {
                "imageUrl": "https://...",
                "status": "normal|warning|danger",
                "action": "Action description",
                "time": "ISO timestamp"
            }
        """
        if not self.supabase_integration or not self.supabase_integration.is_connected:
            logger.warning("Supabase integration not available, using mock mode")
            self._mock_realtime_notification(event_response)
            return
        
        # Create database event record
        db_event = {
            "event_id": f"evt_{int(datetime.now().timestamp())}",
            "event_type": self._infer_event_type_from_action(event_response.get("action", "")),
            "event_description": event_response.get("action", ""),
            "confidence_score": self._extract_confidence_from_action(event_response.get("action", "")),
            "status": "detected",
            "detected_at": event_response.get("time", datetime.now().isoformat()),
            "created_at": datetime.now().isoformat(),
            # Add other required fields based on your schema
        }
        
        # Publish to Supabase (this triggers realtime notifications to mobile)
        success = self.supabase_integration.publish_realtime_event(db_event)
        
        if success:
            # Create mobile notification payload
            mobile_payload = self.supabase_integration.create_mobile_notification_payload(event_response)
            logger.info(f"📱 Mobile notification payload created: {mobile_payload}")
        else:
            logger.error("Failed to publish healthcare event to Supabase")
    
    def _infer_event_type_from_action(self, action: str) -> str:
        """Infer event type from action description"""
        if "té" in action.lower() or "fall" in action.lower():
            return "fall"
        elif "co giật" in action.lower() or "seizure" in action.lower():
            return "abnormal_behavior"
        else:
            return "unknown"
    
    def _extract_confidence_from_action(self, action: str) -> float:
        """Extract confidence score from action description"""
        import re
        match = re.search(r'(\d+)%', action)
        if match:
            return float(match.group(1)) / 100.0
        return 0.0
    
    def _mock_realtime_notification(self, event_response: Dict[str, Any]):
        """Mock realtime notification for demo purposes"""
        print("\n" + "="*60)
        print("📱 MOCK MOBILE REALTIME NOTIFICATION")
        print("="*60)
        
        status = event_response.get("status", "normal")
        if status == "danger":
            print("🚨 EMERGENCY PUSH NOTIFICATION")
        elif status == "warning":  
            print("⚠️ WARNING PUSH NOTIFICATION")
        else:
            print("ℹ️ INFO PUSH NOTIFICATION")
            
        print(f"Status: {status.upper()}")
        print(f"Message: {event_response.get('action', '')}")
        print(f"Image: {event_response.get('imageUrl', 'N/A')}")
        print(f"Time: {event_response.get('time', '')}")
        print("📱 Sent to all registered mobile devices")
        print("="*60)

# Global instance
healthcare_realtime_publisher = HealthcareRealtimePublisher()

def publish_to_mobile(event_response: Dict[str, Any]):
    """
    Convenience function to publish healthcare event to mobile devices
    
    Args:
        event_response: Healthcare event response
    """
    healthcare_realtime_publisher.publish_healthcare_event_with_realtime(event_response)

if __name__ == "__main__":
    # Test the realtime integration
    print("🧪 Testing Healthcare Supabase Realtime Integration")
    
    # Test events
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
    
    # Test connection
    if healthcare_realtime_publisher.supabase_integration:
        healthcare_realtime_publisher.supabase_integration.test_realtime_connection()
    
    # Publish test events
    for i, event in enumerate(test_events):
        print(f"\n--- Publishing Test Event {i+1} ---")
        publish_to_mobile(event)
        import time
        time.sleep(2)
