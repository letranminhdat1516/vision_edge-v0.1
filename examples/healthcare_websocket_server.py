#!/usr/bin/env python3
"""
Healthcare WebSocket Server for Real-time Alerts
Sends alerts to mobile clients in real-time
"""

import asyncio
import websockets
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Set
from pathlib import Path
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HealthcareWebSocket")

class HealthcareWebSocketServer:
    """
    WebSocket server for real-time healthcare alerts
    Supports multiple mobile clients simultaneously
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.session_id = str(uuid.uuid4())
        self.user_sessions = {}  # Track user sessions
        self.alert_history = []  # Store recent alerts
        self.running = False
        
    async def register_client(self, websocket, path=None):
        """Register new mobile client"""
        self.clients.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"📱 Mobile client connected: {client_id}")
        
        # Send welcome message with session info
        welcome_msg = {
            "type": "connection",
            "sessionId": self.session_id,
            "clientId": client_id,
            "time": datetime.now().isoformat(),
            "message": "Connected to Healthcare Monitor"
        }
        
        try:
            await websocket.send(json.dumps(welcome_msg))
            
            # Send recent alerts to new client
            for alert in self.alert_history[-5:]:  # Last 5 alerts
                await websocket.send(json.dumps(alert))
            
            # Keep connection alive - listen for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"📩 Received from {client_id}: {data}")
                    # Handle client messages here if needed
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from {client_id}: {message}")
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📱 Client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"🚨 Error handling client {client_id}: {e}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            logger.info(f"📱 Mobile client disconnected: {client_id}")
    
    async def broadcast_alert(self, alert_data: Dict):
        """Broadcast alert to all connected mobile clients"""
        if not self.clients:
            logger.warning("🚫 No mobile clients connected - alert not sent")
            return
        
        # Ensure standard format: sessionId, imageUrl, status, action, time
        standard_alert = {
            "sessionId": alert_data.get("sessionId", self.session_id),
            "imageUrl": alert_data.get("imageUrl", ""),
            "status": alert_data.get("status", "unknown"),
            "action": alert_data.get("action", "unknown"),
            "time": alert_data.get("time", int(time.time()))
        }
        
        # Store in history
        self.alert_history.append(standard_alert)
        if len(self.alert_history) > 50:  # Keep only recent 50 alerts
            self.alert_history.pop(0)
        
        message = json.dumps(standard_alert, ensure_ascii=False)
        logger.info(f"📤 Broadcasting alert: {standard_alert['action']} (status: {standard_alert['status']})")
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        
        logger.info(f"✅ Alert sent to {len(self.clients)} mobile clients")
    
    def send_alert_sync(self, alert_data: Dict):
        """Synchronous method to send alert (for use in main thread)"""
        if self.running:
            # Run in the event loop
            asyncio.create_task(self.broadcast_alert(alert_data))
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"🚀 Healthcare WebSocket Server starting on {self.host}:{self.port}")
        
        server = await websockets.serve(
            self.register_client,
            self.host,
            self.port,
            ping_interval=30,  # Keep connections alive
            ping_timeout=10,
            close_timeout=5
        )
        
        logger.info(f"✅ Healthcare WebSocket Server running on ws://{self.host}:{self.port}")
        logger.info("📱 Mobile clients can now connect for real-time alerts")
        
        try:
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.running = False
    
    def run_server_in_thread(self):
        """Run server in separate thread"""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.start_server())
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
            finally:
                loop.close()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
    
    def get_stats(self) -> Dict:
        """Get server statistics"""
        return {
            "connected_clients": len(self.clients),
            "session_id": self.session_id,
            "alerts_sent": len(self.alert_history),
            "running": self.running,
            "server_address": f"ws://{self.host}:{self.port}"
        }

# Standalone server for testing
if __name__ == "__main__":
    server = HealthcareWebSocketServer()
    
    # Test alert data - SIMPLIFIED FORMAT
    test_alerts = [
        {
            "sessionId": "test_session_001",
            "imageUrl": "http://192.168.8.122:8080/alerts/test_fall.jpg",
            "status": "critical",
            "action": "fall_detected",
            "time": int(time.time())
        },
        {
            "sessionId": "test_session_001",
            "imageUrl": "http://192.168.8.122:8080/alerts/test_seizure.jpg",
            "status": "critical",
            "action": "seizure_detected",
            "time": int(time.time())
        }
    ]
    
    async def send_test_alerts():
        """Send test alerts periodically"""
        await asyncio.sleep(5)  # Wait for server to start
        
        for i, alert in enumerate(test_alerts):
            await server.broadcast_alert(alert)
            await asyncio.sleep(10)  # Wait 10 seconds between alerts
            
            # Send warning alert
            warning_alert = {
                "sessionId": alert["sessionId"],
                "imageUrl": alert["imageUrl"],
                "status": "warning",
                "action": alert["action"].replace("detected", "warning"),
                "time": int(time.time())
            }
            
            await asyncio.sleep(5)
            await server.broadcast_alert(warning_alert)
    
    async def main():
        # Start server and test alerts concurrently
        await asyncio.gather(
            server.start_server(),
            send_test_alerts()
        )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
