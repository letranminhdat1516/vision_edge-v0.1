"""
VISION EDGE HEALTHCARE - TEST CLIENT
Test WebSocket connections and API endpoints
"""

import asyncio
import json
import uuid
from datetime import datetime
import httpx
import websockets
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000"
TEST_USER_ID = "550e8400-e29b-41d4-a716-446655440000"

async def test_api_endpoints():
    """Test REST API endpoints"""
    print("🧪 TESTING API ENDPOINTS")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health check
            response = await client.get(f"{API_BASE_URL}/api/health")
            print(f"✅ Health Check: {response.status_code}")
            print(f"   Response: {response.json()}")
            
            # Test create detection event
            detection_event = {
                "user_id": TEST_USER_ID,
                "session_id": f"session_{uuid.uuid4()}",
                "image_url": "/test/image.jpg",
                "status": "Normal",
                "action": "walking",
                "location": {"x": 100, "y": 200, "room": "living_room"},
                "confidence_score": 0.95
            }
            
            response = await client.post(
                f"{API_BASE_URL}/api/detection-events",
                json=detection_event
            )
            print(f"✅ Create Detection Event: {response.status_code}")
            print(f"   Response: {response.json()}")
            
            # Test get detection events
            response = await client.get(
                f"{API_BASE_URL}/api/users/{TEST_USER_ID}/detection-events?limit=5"
            )
            print(f"✅ Get Detection Events: {response.status_code}")
            print(f"   Count: {len(response.json().get('events', []))}")
            
            # Test get AI summary
            response = await client.get(
                f"{API_BASE_URL}/api/users/{TEST_USER_ID}/ai-summary"
            )
            print(f"✅ Get AI Summary: {response.status_code}")
            print(f"   Summary: {response.json().get('ai_summary', 'N/A')[:50]}...")
            
        except Exception as e:
            print(f"❌ API Test Error: {e}")

async def test_websocket_connection():
    """Test WebSocket real-time connection"""
    print("\n🔌 TESTING WEBSOCKET CONNECTION")
    print("=" * 50)
    
    try:
        uri = f"{WEBSOCKET_URL}/ws/{TEST_USER_ID}"
        
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to WebSocket: {uri}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            response = await websocket.recv()
            print(f"✅ Ping Response: {json.loads(response)}")
            
            # Request recent events
            await websocket.send(json.dumps({
                "type": "get_recent_events",
                "limit": 5
            }))
            
            # Listen for messages
            print("👂 Listening for messages...")
            
            timeout_count = 0
            while timeout_count < 3:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    print(f"📨 Received: {data['type']}")
                    
                    if data['type'] == 'initial_data':
                        events_count = len(data['data'].get('recent_events', []))
                        print(f"   Initial data: {events_count} events")
                    
                    timeout_count = 0  # Reset timeout count
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"⏰ Timeout {timeout_count}/3")
            
            print("✅ WebSocket test completed")
            
    except Exception as e:
        print(f"❌ WebSocket Test Error: {e}")

async def simulate_ai_analyst_data():
    """Simulate AI Analyst module sending detection events"""
    print("\n🤖 SIMULATING AI ANALYST MODULE")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            session_id = f"ai_session_{uuid.uuid4()}"
            
            # Simulate multiple detection events
            statuses = ["Normal", "Warning", "Normal", "Danger", "Normal"]
            actions = ["walking", "sitting", "standing", "fallen", "walking"]
            
            for i, (status, action) in enumerate(zip(statuses, actions)):
                event = {
                    "user_id": TEST_USER_ID,
                    "session_id": session_id,
                    "image_url": f"/ai/detection_{i+1}.jpg",
                    "status": status,
                    "action": action,
                    "location": {
                        "x": 100 + i*10,
                        "y": 200 + i*5,
                        "room": "living_room"
                    },
                    "confidence_score": 0.85 + (i * 0.02),
                    "ai_metadata": {
                        "model_version": "yolo_v8",
                        "processing_time_ms": 150 + i*10
                    }
                }
                
                response = await client.post(
                    f"{API_BASE_URL}/api/detection-events",
                    json=event
                )
                
                if response.status_code == 200:
                    print(f"✅ Event {i+1}: {status} - {action}")
                else:
                    print(f"❌ Event {i+1}: Failed")
                
                await asyncio.sleep(0.5)  # Simulate real-time interval
            
            print(f"🎉 Simulated {len(statuses)} AI detection events")
            
        except Exception as e:
            print(f"❌ AI Simulation Error: {e}")

async def test_real_time_streaming():
    """Test real-time streaming with concurrent WebSocket and API calls"""
    print("\n🔄 TESTING REAL-TIME STREAMING")
    print("=" * 50)
    
    # Start WebSocket listener
    async def websocket_listener():
        try:
            uri = f"{WEBSOCKET_URL}/ws/{TEST_USER_ID}"
            async with websockets.connect(uri) as websocket:
                print("👂 WebSocket listener started")
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(message)
                        
                        if data['type'] == 'detection_event':
                            event_data = data['data']
                            print(f"🔴 Real-time event: {event_data['status']} - {event_data['action']}")
                        elif data['type'] == 'alert':
                            alert_data = data['data']
                            print(f"🚨 Real-time alert: {alert_data['severity']} - {alert_data['message'][:30]}...")
                            
                    except asyncio.TimeoutError:
                        print("⏰ WebSocket timeout - stopping listener")
                        break
                        
        except Exception as e:
            print(f"❌ WebSocket Listener Error: {e}")
    
    # Start API event generator
    async def api_event_generator():
        await asyncio.sleep(1)  # Wait for WebSocket to connect
        
        async with httpx.AsyncClient() as client:
            try:
                for i in range(3):
                    # Create detection event
                    event = {
                        "user_id": TEST_USER_ID,
                        "session_id": f"realtime_session_{uuid.uuid4()}",
                        "image_url": f"/realtime/event_{i+1}.jpg",
                        "status": ["Normal", "Warning", "Danger"][i],
                        "action": ["walking", "sitting", "fallen"][i],
                        "location": {"x": 100+i*20, "y": 200+i*10, "room": "test_room"}
                    }
                    
                    await client.post(f"{API_BASE_URL}/api/detection-events", json=event)
                    print(f"📤 Sent event {i+1}")
                    
                    # Create alert if dangerous
                    if event["status"] == "Danger":
                        alert = {
                            "user_id": TEST_USER_ID,
                            "alert_type": "fall_detection",
                            "severity": "high",
                            "message": "Potential fall detected in test_room"
                        }
                        
                        await client.post(f"{API_BASE_URL}/api/alerts", json=alert)
                        print(f"🚨 Sent alert for event {i+1}")
                    
                    await asyncio.sleep(2)  # Wait between events
                    
            except Exception as e:
                print(f"❌ API Generator Error: {e}")
    
    # Run both concurrently
    await asyncio.gather(
        websocket_listener(),
        api_event_generator()
    )

async def main():
    """Main test function"""
    print("🏥 VISION EDGE HEALTHCARE - SYSTEM TEST")
    print("=" * 60)
    print(f"📍 API URL: {API_BASE_URL}")
    print(f"🔌 WebSocket URL: {WEBSOCKET_URL}")
    print(f"👤 Test User ID: {TEST_USER_ID}")
    print("=" * 60)
    
    # Run tests sequentially
    await test_api_endpoints()
    await test_websocket_connection()
    await simulate_ai_analyst_data()
    await test_real_time_streaming()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
