#!/usr/bin/env python3
"""
Simple WebSocket Server for Healthcare Monitor
Receives JSON data and broadcasts to connected clients
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWebSocketServer:
    def __init__(self, host="localhost", port=8086):
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"📱 Client connected. Total: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"📱 Client disconnected. Remaining: {len(self.clients)}")
        
    async def broadcast(self, message: str):
        """Broadcast message to all clients"""
        if self.clients:
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    print(f"❌ Error sending to client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            for client in disconnected:
                await self.unregister_client(client)
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON
                    data = json.loads(message)
                    
                    # Log received data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n🔔 [{timestamp}] Received JSON:")
                    print("=" * 40)
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    print("=" * 40)
                    
                    # Check if it's an alert
                    if data.get('status') and data.get('status') != 'normal':
                        print(f"🚨 ALERT DETECTED:")
                        print(f"   User: {data.get('userId', 'Unknown')}")
                        print(f"   Status: {data.get('status', 'Unknown')}")
                        print(f"   Action: {data.get('action', 'Unknown')}")
                    
                    # Broadcast to other clients
                    await self.broadcast(message)
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        print(f"🌐 Starting WebSocket Server...")
        print(f"📡 Host: {self.host}")
        print(f"🔌 Port: {self.port}")
        print(f"🔗 URL: ws://{self.host}:{self.port}")
        print(f"📊 Ready to receive healthcare data!")
        print("=" * 50)
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

async def main():
    """Main function"""
    server = SimpleWebSocketServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    print("🏥 Healthcare WebSocket Server - Port 8086")
    print("Listening for data from advanced_healthcare_monitor.py")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
