"""
Healthcare Realtime Client 
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

class HealthcareRealtimeClient:
    """Healthcare realtime client simulating mobile app"""
    
    def __init__(self):
        self.running = True
        self.event_count = 0
        self.alert_count = 0
        
    def handle_event_detection(self, event_data):
        """Handle new event detection from realtime"""
        try:
            self.event_count += 1
            event = event_data.get('new_data', {})
            
            print("=" * 60)
            print(f"🔔 NEW HEALTHCARE EVENT #{self.event_count}")
            print("=" * 60)
            print(f"Event ID: {event.get('event_id', 'N/A')}")
            print(f"Type: {event.get('event_type', 'N/A')}")
            print(f"Description: {event.get('event_description', 'N/A')}")
            print(f"Confidence: {event.get('confidence_score', 0):.2%}")
            print(f"Status: {event.get('status', 'N/A')}")
            print(f"Detected At: {event.get('detected_at', 'N/A')}")
            
            # Display context data
            context = event.get('context_data', {})
            if context:
                print("\n📊 Context Data:")
                for key, value in context.items():
                    print(f"  • {key}: {value}")
            
            # Display AI analysis
            ai_analysis = event.get('ai_analysis_result', {})
            if ai_analysis:
                print("\n🤖 AI Analysis:")
                for key, value in ai_analysis.items():
                    print(f"  • {key}: {value}")
            
            # Display bounding boxes
            bounding_boxes = event.get('bounding_boxes', [])
            if bounding_boxes:
                print(f"\n📍 Bounding Boxes ({len(bounding_boxes)}):")
                for i, box in enumerate(bounding_boxes):
                    print(f"  • Box {i+1}: x={box.get('x', 0)}, y={box.get('y', 0)}, "
                          f"w={box.get('width', 0)}, h={box.get('height', 0)}")
            
            print("\n" + "=" * 60)
            
            # Handle specific event types
            if event.get('event_type') == 'fall_detection':
                self._handle_fall_detection(event)
            elif event.get('event_type') == 'seizure_detection':
                self._handle_seizure_detection(event)
                
        except Exception as e:
            logger.error(f"Error handling event detection: {e}")
    
    def handle_alert(self, event_data):
        """Handle new alert from realtime"""
        try:
            self.alert_count += 1
            alert = event_data.get('new_data', {})
            
            print("🚨" * 20)
            print(f"🚨 NEW ALERT #{self.alert_count}")
            print("🚨" * 20)
            print(f"Alert ID: {alert.get('alert_id', 'N/A')}")
            print(f"Type: {alert.get('alert_type', 'N/A')}")
            print(f"Severity: {alert.get('severity', 'N/A').upper()}")
            print(f"Message: {alert.get('alert_message', 'N/A')}")
            print(f"Status: {alert.get('status', 'N/A')}")
            print(f"Created At: {alert.get('created_at', 'N/A')}")
            
            # Display alert data
            alert_data = alert.get('alert_data', {})
            if alert_data:
                print("\n📋 Alert Data:")
                for key, value in alert_data.items():
                    if key == 'bounding_boxes' and isinstance(value, list):
                        print(f"  • {key}: {len(value)} boxes")
                    else:
                        print(f"  • {key}: {value}")
            
            print("🚨" * 20)
            
            # Handle based on severity
            severity = alert.get('severity', '').lower()
            if severity in ['high', 'critical']:
                self._handle_emergency_alert(alert)
            elif severity == 'medium':
                self._handle_warning_alert(alert)
                
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def handle_snapshot(self, event_data):
        """Handle new snapshot from realtime"""
        try:
            snapshot = event_data.get('new_data', {})
            
            print("📸" * 15)
            print("📸 NEW SNAPSHOT")
            print("📸" * 15)
            print(f"Snapshot ID: {snapshot.get('snapshot_id', 'N/A')}")
            print(f"Image Path: {snapshot.get('image_path', 'N/A')}")
            print(f"Capture Type: {snapshot.get('capture_type', 'N/A')}")
            print(f"Captured At: {snapshot.get('captured_at', 'N/A')}")
            
            # Display metadata
            metadata = snapshot.get('metadata', {})
            if metadata:
                print("\n📋 Metadata:")
                for key, value in metadata.items():
                    print(f"  • {key}: {value}")
            
            print("📸" * 15)
                
        except Exception as e:
            logger.error(f"Error handling snapshot: {e}")
    
    def _handle_fall_detection(self, event):
        """Handle fall detection specific logic"""
        confidence = event.get('confidence_score', 0)
        
        if confidence >= 0.8:
            print("🆘 HIGH CONFIDENCE FALL - EMERGENCY RESPONSE REQUIRED!")
            # Here you could trigger:
            # - Emergency notification to caregivers
            # - Automatic 911 call
            # - Alert security system
            # - Send SMS/email to family members
        elif confidence >= 0.6:
            print("⚠️ MODERATE FALL RISK - CHECK PATIENT STATUS")
            # Here you could trigger:
            # - Notification to on-duty nurse
            # - Log in monitoring system
            # - Send alert to mobile app
        
        # Example integration actions
        print("💡 Suggested Actions:")
        print("  • Verify patient condition via camera")
        print("  • Dispatch medical personnel if needed") 
        print("  • Log incident in patient record")
        print("  • Contact emergency contacts if severe")
    
    def _handle_seizure_detection(self, event):
        """Handle seizure detection specific logic"""
        confidence = event.get('confidence_score', 0)
        
        if confidence >= 0.7:
            print("🆘 SEIZURE ACTIVITY DETECTED - MEDICAL ATTENTION REQUIRED!")
            # Here you could trigger:
            # - Immediate medical response
            # - Prepare seizure protocol
            # - Contact neurologist/doctor
            # - Clear area of hazards remotely
        else:
            print("⚠️ POSSIBLE SEIZURE ACTIVITY - MONITOR CLOSELY")
            # Here you could trigger:
            # - Increase monitoring frequency
            # - Alert medical staff
            # - Prepare for potential escalation
        
        # Example integration actions
        print("💡 Suggested Actions:")
        print("  • Activate seizure response protocol")
        print("  • Ensure airway is clear")
        print("  • Time the seizure duration")
        print("  • Contact medical team immediately")
    
    def _handle_emergency_alert(self, alert):
        """Handle high/critical severity alerts"""
        print("🆘 EMERGENCY ALERT - IMMEDIATE ACTION REQUIRED!")
        
        # Example emergency actions
        print("💡 Emergency Protocol Activated:")
        print("  • Medical team dispatched")
        print("  • Emergency contacts notified")
        print("  • Incident logged with timestamp")
        print("  • Camera feed prioritized for monitoring")
    
    def _handle_warning_alert(self, alert):
        """Handle medium severity alerts"""
        print("⚠️ WARNING ALERT - ATTENTION NEEDED")
        
        # Example warning actions
        print("💡 Warning Protocol:")
        print("  • Staff notified via mobile app")
        print("  • Patient flagged for increased monitoring")
        print("  • Incident logged in daily report")
    
    def start_listening(self):
        """Start listening for realtime events"""
        try:
            print("🚀 Starting Healthcare Realtime Client...")
            print("📡 Connecting to Supabase realtime...")
            
            # Subscribe to event detections
            realtime_service.subscribe_to_events(
                'event_detections',
                'INSERT', 
                self.handle_event_detection
            )
            
            # Subscribe to alerts
            realtime_service.subscribe_to_events(
                'alerts',
                'INSERT',
                self.handle_alert
            )
            
            # Subscribe to snapshots
            realtime_service.subscribe_to_events(
                'snapshots',
                'INSERT', 
                self.handle_snapshot
            )
            
            print("✅ Realtime client started successfully!")
            print("🔊 Listening for healthcare events...")
            print("⏹️  Press Ctrl+C to stop")
            print("-" * 60)
            
            # Keep the client running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping realtime client...")
                
        except Exception as e:
            logger.error(f"Error starting realtime client: {e}")
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop listening and cleanup"""
        try:
            realtime_service.close()
            print("✅ Realtime client stopped")
        except Exception as e:
            logger.error(f"Error stopping realtime client: {e}")
    
    def get_stats(self):
        """Get client statistics"""
        return {
            'events_received': self.event_count,
            'alerts_received': self.alert_count,
            'uptime': datetime.now().isoformat()
        }

def main():
    """Main function to run the realtime client"""
    client = HealthcareRealtimeClient()
    
    try:
        client.start_listening()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        client.stop_listening()

if __name__ == "__main__":
    main()
