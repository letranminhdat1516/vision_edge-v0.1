#!/usr/bin/env python3    def __init__(self, host="localhost", port=8080):"""
Simple WebSocket Server for Healthcare Monitor
Receives JSON data from advanced_healthcare_monitor.py and logs/broadcasts it
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set, Any
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WebSocketServer')

class HealthcareWebSocketServer:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.running = False
        
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        print(f"📱 Client connected from {websocket.remote_address}")
        
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        print(f"📱 Client disconnected. Remaining: {len(self.clients)}")
        
    async def broadcast_to_clients(self, message: str):
        """Broadcast message to all connected clients except sender"""
        if self.clients:
            # Send to all clients (web dashboard, mobile apps, etc.)
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            for client in disconnected:
                await self.unregister_client(client)
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse incoming JSON
                    data = json.loads(message)
                    
                    # Log the received data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n🔔 [{timestamp}] Received JSON from healthcare monitor:")
                    print("=" * 50)
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    print("=" * 50)
                    
                    # Check if it's an alert
                    if data.get('status') and data.get('status') != 'normal':
                        print(f"🚨 ALERT DETECTED:")
                        print(f"   User: {data.get('userId', 'Unknown')}")
                        print(f"   Status: {data.get('status', 'Unknown')}")
                        print(f"   Action: {data.get('action', 'Unknown')}")
                        print(f"   Location: {data.get('location', 'Unknown')}")
                        if data.get('imageUrl'):
                            print(f"   Image: {data.get('imageUrl')}")
                    
                    # Broadcast to other clients (dashboard, mobile apps)
                    await self.broadcast_to_clients(message)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    print(f"❌ Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        
        print(f"🌐 Starting WebSocket Server...")
        print(f"📡 Host: {self.host}")
        print(f"🔌 Port: {self.port}")
        print(f"🔗 URL: ws://{self.host}:{self.port}/ws/patient123")
        print(f"📊 Ready to receive healthcare monitor data!")
        print("=" * 60)
        
        # Start server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await server.wait_closed()
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        logger.info("WebSocket server stopping...")
        print("\n👋 WebSocket server stopping...")

# Global server instance
server = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global server
    if server:
        server.stop_server()
    sys.exit(0)

async def main():
    """Main function"""
    global server
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = HealthcareWebSocketServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Server error: {e}")
    finally:
        if server:
            server.stop_server()

if __name__ == "__main__":
    print("🏥 Healthcare WebSocket Server")
    print("Listening for data from advanced_healthcare_monitor.py")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
