#!/usr/bin/env python3
"""
Healthcare WebSocket Server for Real-time Alerts (Production Version)
Sends alerts to mobile clients in real-time - NO TEST ALERTS
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
            logger.info(f"✅ Welcome message sent to {client_id}")
            
            # Send recent alerts to new client (only last 3 to avoid overload)
            if self.alert_history:
                for alert in self.alert_history[-3:]:  # Last 3 alerts only
                    await websocket.send(json.dumps(alert))
                logger.info(f"📋 Sent {min(3, len(self.alert_history))} recent alerts to {client_id}")
            
            # Keep connection alive - listen for messages or ping
            async for message in websocket:
                try:
                    # Handle any incoming messages from client
                    data = json.loads(message)
                    logger.info(f"📩 Received from {client_id}: {data.get('type', 'unknown')}")
                    
                    # Echo back a response if needed
                    if data.get('type') == 'ping':
                        await websocket.send(json.dumps({"type": "pong", "time": datetime.now().isoformat()}))
                        
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from {client_id}: {message}")
                except Exception as e:
                    logger.error(f"🚨 Error processing message from {client_id}: {e}")
                    break
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📱 Client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"🚨 Error handling client {client_id}: {e}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            logger.info(f"📱 Mobile client disconnected: {client_id} (Total: {len(self.clients)} clients)")
    
    async def broadcast_alert(self, alert_data: Dict):
        """Broadcast alert to all connected mobile clients"""
        if not self.clients:
            logger.warning("🚫 No mobile clients connected - alert not sent")
            return
        
        # Ensure standard format: imageUrl, status, action, time (4 fields only)
        standard_alert = {
            "imageUrl": alert_data.get("imageUrl", ""),
            "status": alert_data.get("status", "unknown"),
            "action": alert_data.get("action", "unknown"),
            "time": alert_data.get("time", int(time.time()))
        }
        
        # Store in history (limit to prevent memory issues)
        self.alert_history.append(standard_alert)
        if len(self.alert_history) > 20:  # Keep only recent 20 alerts
            self.alert_history.pop(0)
        
        message = json.dumps(standard_alert, ensure_ascii=False)
        logger.info(f"📤 Broadcasting alert: {standard_alert['action']} (status: {standard_alert['status']}) to {len(self.clients)} clients")
        
        # Send to all connected clients
        disconnected_clients = set()
        successful_sends = 0
        
        for client in self.clients.copy():
            try:
                await client.send(message)
                successful_sends += 1
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client disconnected during send")
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        
        logger.info(f"✅ Alert sent successfully to {successful_sends} mobile clients")
    
    def send_alert_sync(self, alert_data: Dict):
        """Synchronous method to send alert (for use in main thread)"""
        if self.running and hasattr(self, '_loop') and self._loop.is_running():
            # Schedule the coroutine in the event loop
            asyncio.run_coroutine_threadsafe(self.broadcast_alert(alert_data), self._loop)
        else:
            logger.warning("WebSocket server not running or event loop not available")
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        self._loop = asyncio.get_event_loop()
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
        logger.info("🔄 Server is ready to receive alerts from healthcare monitor")
        
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
            self._loop = loop
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

# Production server - no test alerts
if __name__ == "__main__":
    server = HealthcareWebSocketServer()
    
    try:
        # Just start server without test alerts
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped")
