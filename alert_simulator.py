#!/usr/bin/env python3
"""
Real-time Healthcare Alert Simulator
Automatically sends alerts every few seconds for testing WebSocket dashboard
"""

import asyncio
import websockets
import json
import random
from datetime import datetime
import time

class HealthcareAlertSimulator:
    def __init__(self, websocket_url="ws://localhost:8080"):
        self.websocket_url = websocket_url
        self.websocket = None
        self.user_ids = ["patient123"]
        self.locations = ["Room 101"]
        self.statuses = ["normal", "warning", "danger"]
        self.actions = ["monitoring", "alert_warning", "alert_fall", "alert_seizure"]
        
    async def connect_websocket(self):
        """Connect to WebSocket server with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add ping settings to prevent timeout
                self.websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                print(f"✅ Connected to WebSocket: {self.websocket_url}")
                return True
            except Exception as e:
                print(f"❌ Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    print("❌ Failed to connect after all retries")
                    return False
    
    def generate_alert_data(self, alert_type=None):
        """Generate random alert data"""
        timestamp = datetime.now()
        session_id = f"session_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
        
        # Determine alert type if not specified
        if alert_type is None:
            alert_type = random.choice(["normal", "warning", "fall", "seizure"])
        
        # Map alert types to status and action
        if alert_type == "normal":
            status = "normal"
            action = "monitoring"
        elif alert_type == "warning":
            status = "warning"
            action = "alert_warning"
        elif alert_type == "fall":
            status = "danger"
            action = "alert_fall"
        elif alert_type == "seizure":
            status = "danger"
            action = "alert_seizure"
        
        # Generate image URL
        image_filename = f"alert_{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}_{action}_conf_0.{random.randint(750, 950)}.jpg"
        
        return {
            "userId": random.choice(self.user_ids),
            "sessionId": session_id,
            "imageUrl": f"http://localhost:8003/api/demo/alerts/{image_filename}",
            "status": status,
            "action": action,
            "location": random.choice(self.locations),
            "time": timestamp.isoformat()
        }
    
    async def send_alert(self, alert_data):
        """Send alert data via WebSocket with auto-reconnect"""
        if not self.websocket:
            print("🔄 WebSocket disconnected, attempting to reconnect...")
            if not await self.connect_websocket():
                return False
        
        try:
            message = json.dumps(alert_data, indent=2)
            if self.websocket:
                await self.websocket.send(message)
                
                # Print sent data (simplified)
                timestamp = datetime.now().strftime('%H:%M:%S')
                status_emoji = "🟢" if alert_data['status'] == 'normal' else "🟡" if alert_data['status'] == 'warning' else "🔴"
                print(f"{status_emoji} [{timestamp}] {alert_data['userId']} - {alert_data['action']} @ {alert_data['location']}")
                
                # Highlight emergencies
                if alert_data['status'] in ['warning', 'danger']:
                    print(f"🚨 EMERGENCY: {alert_data['action']} - {alert_data['location']}")
                
                return True
        except websockets.exceptions.ConnectionClosed:
            print("🔄 Connection lost, will retry next time...")
            self.websocket = None
            return False
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
            self.websocket = None
            return False
        
        return False
    
    async def run_simulation(self, interval=3, total_alerts=None):
        """Run continuous alert simulation"""
        print("🏥 Healthcare Alert Simulator")
        print(f"📡 WebSocket URL: {self.websocket_url}")
        print(f"⏱️  Alert interval: {interval} seconds")
        print(f"🎯 Total alerts: {'Unlimited' if total_alerts is None else total_alerts}")
        print("=" * 50)
        
        # Connect to WebSocket
        if not await self.connect_websocket():
            print("❌ Failed to connect. Exiting...")
            return
        
        alert_count = 0
        successful_sends = 0
        failed_sends = 0
        
        try:
            while True:
                # Generate different types of alerts
                alert_types = ["normal", "normal", "warning", "fall", "seizure"]  # More normal alerts
                alert_type = random.choice(alert_types)
                
                # Generate and send alert
                alert_data = self.generate_alert_data(alert_type)
                success = await self.send_alert(alert_data)
                
                alert_count += 1
                if success:
                    successful_sends += 1
                else:
                    failed_sends += 1
                
                # Show statistics every 10 alerts
                if alert_count % 10 == 0:
                    print(f"📊 Total: {alert_count} | ✅ Success: {successful_sends} | ❌ Failed: {failed_sends}")
                
                # Check if we've reached the limit
                if total_alerts and alert_count >= total_alerts:
                    print(f"\n✅ Sent {alert_count} alerts. Simulation complete!")
                    break
                
                # Wait before next alert
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n👋 Simulation stopped by user. Total alerts sent: {alert_count}")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
                print("🔌 WebSocket connection closed")

async def main():
    """Main function"""
    print("🏥 Real-time Healthcare Alert Simulator")
    print("=" * 50)
    print("This will automatically send alerts to test the dashboard")
    print("Press Ctrl+C to stop")
    print()
    
    # Configuration
    websocket_url = "ws://localhost:8080"
    interval = 2  # Send alert every 2 seconds
    
    simulator = HealthcareAlertSimulator(websocket_url)
    await simulator.run_simulation(interval=interval)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
