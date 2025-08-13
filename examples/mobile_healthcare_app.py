"""
Healthcare Realtime Client - Mobile App Simulation
Real-time client to listen for healthcare events from Supabase
Simulates Flutter/mobile app receiving notifications  
"""

import time
import json
import logging
import sys
import os
from datetime import datetime
import signal

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.postgresql_healthcare_service import postgresql_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MobileHealthcareApp:
    """Healthcare mobile app simulation"""
    
    def __init__(self):
        self.running = True
        self.last_event_id = None
        
    def handle_fall_detection(self, event_data):
        """Handle fall detection event"""
        confidence = float(event_data.get('confidence_score', 0))
        camera_id = event_data.get('camera_id')
        detected_at = str(event_data.get('detected_at', ''))[:19]
        
        print("🚨" * 20)
        print("    FALL DETECTED - EMERGENCY ALERT    ")  
        print("🚨" * 20)
        print(f"📍 Location: Camera {camera_id}")
        print(f"🎯 Confidence: {confidence:.1%}")
        print(f"⏰ Time: {detected_at}")
        print(f"🚨 CRITICAL ALERT ACTIVATED")
        print()
        
        self.send_critical_notification(
            title="🚨 FALL DETECTED",
            message=f"Emergency! Person fell with {confidence:.0%} confidence",
            event_data=event_data
        )
        
    def handle_seizure_detection(self, event_data):
        """Handle seizure detection event"""
        confidence = float(event_data.get('confidence_score', 0))
        camera_id = event_data.get('camera_id')
        detected_at = str(event_data.get('detected_at', ''))[:19]
        
        print("⚡" * 20) 
        print("  SEIZURE DETECTED - MEDICAL ALERT  ")
        print("⚡" * 20)
        print(f"📍 Location: Camera {camera_id}")
        print(f"🎯 Confidence: {confidence:.1%}")
        print(f"⏰ Time: {detected_at}")
        print(f"⚡ MEDICAL ATTENTION REQUIRED")
        print()
        
        self.send_critical_notification(
            title="⚡ SEIZURE ACTIVITY",
            message=f"Medical emergency! Seizure detected with {confidence:.0%} confidence",
            event_data=event_data
        )
        
    def send_critical_notification(self, title: str, message: str, event_data: dict):
        """Send critical mobile notification"""
        print("📱 MOBILE NOTIFICATION SENT:")
        print(f"   Title: {title}")
        print(f"   Message: {message}")
        print()
        
        print("🔔 MULTI-CHANNEL ALERT ACTIVATED:")
        print("   📱 Push Notification ✅")
        print("   📧 Email Alert ✅")
        print("   📞 SMS Alert ✅")
        print("   🚨 Emergency Contacts ✅")
        print()
        
        print("📊 Event Details:")
        print(f"   ID: {event_data.get('event_id', 'N/A')}")
        print(f"   Type: {event_data.get('event_type', 'N/A')}")
        print(f"   Camera: {event_data.get('camera_id', 'N/A')}")
        print(f"   User: {event_data.get('user_id', 'N/A')}")
        print()
        print("=" * 80)
        print()
        
    def listen_for_events(self):
        """Listen for new healthcare events"""
        print("📱 Healthcare Mobile App Started")
        print("🏥 Vision Edge Healthcare Monitoring")
        print("🔊 Listening for real-time events...")
        print("=" * 80)
        
        last_check_time = datetime.now()
        
        while self.running:
            try:
                # Get recent events
                recent_events = postgresql_service.get_recent_events(limit=5)
                
                # Process new events
                for event_data in recent_events:
                    event_id = event_data.get('event_id')
                    event_time = event_data.get('detected_at')
                    
                    # Only process events newer than our last check
                    if self.last_event_id != event_id and event_time and event_time > last_check_time:
                        event_type = event_data.get('event_type')
                        
                        if event_type == 'fall':
                            self.handle_fall_detection(event_data)
                        elif event_type == 'abnormal_behavior':
                            # Check if it's seizure
                            description = event_data.get('event_description', '').lower()
                            if 'seizure' in description:
                                self.handle_seizure_detection(event_data)
                            else:
                                print(f"🔍 Abnormal behavior detected: {event_data.get('event_description')}")
                                print()
                        
                        self.last_event_id = event_id
                
                last_check_time = datetime.now()
                
                # Sleep before next poll
                time.sleep(3)  # Check every 3 seconds
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error in event listener: {e}")
                time.sleep(5)
                
        print("\n📱 Healthcare Mobile App Stopped")
        
    def start(self):
        """Start the mobile app simulation"""
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down mobile app...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        self.listen_for_events()

def main():
    """Main function"""
    app = MobileHealthcareApp()
    app.start()

if __name__ == "__main__":
    main()
