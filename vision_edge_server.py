"""
VISION EDGE HEALTHCARE - FASTAPI WEBSOCKET SERVER
Real-time streaming with PostgreSQL + Redis architecture
"""

import asyncio
import json
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import redis.asyncio as redis
import asyncpg
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# =====================================================
# CONFIGURATION
# =====================================================

# Database configuration
DATABASE_CONFIG = {
    "host": "postgres",
    "port": 5432,
    "database": "vision_edge_healthcare",
    "user": "postgres", 
    "password": "your_password"
}

# Redis configuration
REDIS_CONFIG = {
    "host": "redis",
    "port": 6379,
    "password": "vision_edge_redis_pass",
    "db": 0
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# DATA MODELS
# =====================================================

class DetectionEvent(BaseModel):
    user_id: str
    session_id: str
    image_url: str
    status: str  # Normal, Warning, Danger
    action: str
    location: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None
    ai_metadata: Optional[Dict[str, Any]] = None

class AIAnalysisResult(BaseModel):
    total_detection_sessions: int
    progress_compared_to_last_week: float
    daily_activity_log: List[Dict[str, Any]]
    ai_summary: str

class AlertMessage(BaseModel):
    user_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metadata: Optional[Dict[str, Any]] = None

# =====================================================
# DATABASE CONNECTION POOL
# =====================================================

class DatabaseManager:
    def __init__(self):
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(**DATABASE_CONFIG, min_size=5, max_size=20)
            logger.info("✅ PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Database pool closed")
    
    async def execute_query(self, query: str, *args):
        """Execute query and return results"""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_function(self, function_name: str, *args):
        """Execute PostgreSQL function"""
        placeholders = ", ".join(f"${i+1}" for i in range(len(args)))
        query = f"SELECT * FROM {function_name}({placeholders})"
        return await self.execute_query(query, *args)

# =====================================================
# REDIS CONNECTION MANAGER
# =====================================================

class RedisManager:
    def __init__(self):
        self.redis_client = None
        self.pubsub = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**REDIS_CONFIG)
            await self.redis_client.ping()
            logger.info("✅ Redis connection initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔒 Redis connection closed")
    
    async def publish_event(self, channel: str, data: Dict[str, Any]):
        """Publish event to Redis stream"""
        try:
            await self.redis_client.xadd(
                f"stream:{channel}",
                data,
                maxlen=1000  # Keep last 1000 events
            )
            # Also publish to pub/sub for real-time notifications
            await self.redis_client.publish(f"channel:{channel}", json.dumps(data))
            logger.info(f"📡 Published to {channel}: {data.get('event_type', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Failed to publish event: {e}")
    
    async def get_stream_data(self, channel: str, count: int = 10):
        """Get recent data from Redis stream"""
        try:
            result = await self.redis_client.xrevrange(f"stream:{channel}", count=count)
            return [{"id": entry_id, "data": data} for entry_id, data in result]
        except Exception as e:
            logger.error(f"❌ Failed to get stream data: {e}")
            return []
    
    async def cache_set(self, key: str, value: Any, expire: int = 3600):
        """Set cache with expiration"""
        await self.redis_client.setex(key, expire, json.dumps(value))
    
    async def cache_get(self, key: str):
        """Get cache data"""
        result = await self.redis_client.get(key)
        return json.loads(result) if result else None

# =====================================================
# WEBSOCKET CONNECTION MANAGER
# =====================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}  # websocket_id -> user_id
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect user to WebSocket"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        self.user_sessions[id(websocket)] = user_id
        
        logger.info(f"👤 User {user_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect user from WebSocket"""
        websocket_id = id(websocket)
        
        if websocket_id in self.user_sessions:
            user_id = self.user_sessions[websocket_id]
            
            if user_id in self.active_connections:
                try:
                    self.active_connections[user_id].remove(websocket)
                    if not self.active_connections[user_id]:
                        del self.active_connections[user_id]
                except ValueError:
                    pass
            
            del self.user_sessions[websocket_id]
            logger.info(f"👤 User {user_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            dead_connections = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    dead_connections.append(websocket)
            
            # Remove dead connections
            for dead_ws in dead_connections:
                self.disconnect(dead_ws)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected users"""
        for user_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, user_id)

# =====================================================
# GLOBAL INSTANCES
# =====================================================

db_manager = DatabaseManager()
redis_manager = RedisManager()
connection_manager = ConnectionManager()

# =====================================================
# FASTAPI APPLICATION
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await db_manager.initialize()
    await redis_manager.initialize()
    
    # Start Redis listener for real-time events
    asyncio.create_task(redis_event_listener())
    
    logger.info("🚀 Vision Edge Healthcare Server started")
    
    yield
    
    # Shutdown
    await db_manager.close()
    await redis_manager.close()
    logger.info("🛑 Vision Edge Healthcare Server stopped")

app = FastAPI(
    title="Vision Edge Healthcare API",
    description="Real-time healthcare monitoring with AI analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# REDIS EVENT LISTENER
# =====================================================

async def redis_event_listener():
    """Listen to Redis pub/sub for real-time events"""
    try:
        pubsub = redis_manager.redis_client.pubsub()
        await pubsub.subscribe("channel:detection_events", "channel:alerts")
        
        logger.info("👂 Started Redis event listener")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    channel = message["channel"].decode()
                    
                    if channel == "channel:detection_events":
                        await handle_detection_event(data)
                    elif channel == "channel:alerts":
                        await handle_alert_event(data)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing Redis message: {e}")
                    
    except Exception as e:
        logger.error(f"❌ Redis listener error: {e}")

async def handle_detection_event(data: Dict[str, Any]):
    """Handle detection event from Redis"""
    user_id = data.get("user_id")
    if user_id:
        message = {
            "type": "detection_event",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await connection_manager.send_personal_message(message, user_id)

async def handle_alert_event(data: Dict[str, Any]):
    """Handle alert event from Redis"""
    user_id = data.get("user_id")
    if user_id:
        message = {
            "type": "alert",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await connection_manager.send_personal_message(message, user_id)

# =====================================================
# WEBSOCKET ENDPOINTS
# =====================================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, user_id)
    
    try:
        # Send initial data
        await send_initial_data(websocket, user_id)
        
        # Listen for client messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client requests
            await handle_websocket_message(websocket, user_id, message)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(websocket)

async def send_initial_data(websocket: WebSocket, user_id: str):
    """Send initial data when user connects"""
    try:
        # Get recent detection events
        recent_events = await get_recent_detection_events(user_id, limit=10)
        
        # Get cached AI summary
        ai_summary = await redis_manager.cache_get(f"ai_summary:{user_id}")
        if not ai_summary:
            ai_summary = await get_ai_summary_from_db(user_id)
            await redis_manager.cache_set(f"ai_summary:{user_id}", ai_summary, expire=1800)
        
        initial_data = {
            "type": "initial_data",
            "data": {
                "recent_events": recent_events,
                "ai_summary": ai_summary
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_text(json.dumps(initial_data))
        
    except Exception as e:
        logger.error(f"❌ Error sending initial data: {e}")

async def handle_websocket_message(websocket: WebSocket, user_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    try:
        if message_type == "ping":
            await websocket.send_text(json.dumps({"type": "pong"}))
        
        elif message_type == "get_recent_events":
            limit = message.get("limit", 20)
            events = await get_recent_detection_events(user_id, limit)
            
            response = {
                "type": "recent_events",
                "data": events,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(response))
        
        elif message_type == "get_daily_activity":
            days = message.get("days", 7)
            activity = await get_daily_activity_from_db(user_id, days)
            
            response = {
                "type": "daily_activity",
                "data": activity,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(response))
            
    except Exception as e:
        logger.error(f"❌ Error handling WebSocket message: {e}")

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vision Edge Healthcare API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/detection-events")
async def create_detection_event(event: DetectionEvent):
    """Create new detection event (from AI Analyst module)"""
    try:
        # Insert into PostgreSQL
        event_id = await insert_detection_event_to_db(event)
        
        # Publish to Redis for real-time streaming
        event_data = {
            "event_id": str(event_id),
            "user_id": event.user_id,
            "session_id": event.session_id,
            "status": event.status,
            "action": event.action,
            "location": event.location,
            "timestamp": event.timestamp.isoformat(),
            "event_type": "detection_event"
        }
        
        await redis_manager.publish_event("detection_events", event_data)
        
        # Clear cached AI summary for this user
        await redis_manager.redis_client.delete(f"ai_summary:{event.user_id}")
        
        return {
            "event_id": str(event_id),
            "status": "created",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating detection event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/detection-events")
async def get_detection_events(user_id: str, limit: int = 50, offset: int = 0):
    """Get detection events for user"""
    try:
        events = await get_recent_detection_events(user_id, limit, offset)
        return {
            "user_id": user_id,
            "events": events,
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting detection events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/ai-summary")
async def get_ai_summary(user_id: str, period_days: int = 7):
    """Get AI summary for user"""
    try:
        # Try cache first
        cached_summary = await redis_manager.cache_get(f"ai_summary:{user_id}")
        if cached_summary:
            return cached_summary
        
        # Get from database
        summary = await get_ai_summary_from_db(user_id, period_days)
        
        # Cache for 30 minutes
        await redis_manager.cache_set(f"ai_summary:{user_id}", summary, expire=1800)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error getting AI summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts")
async def create_alert(alert: AlertMessage):
    """Create new alert"""
    try:
        # Insert into PostgreSQL
        alert_id = await insert_alert_to_db(alert)
        
        # Publish to Redis for real-time notification
        alert_data = {
            "alert_id": str(alert_id),
            "user_id": alert.user_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "metadata": alert.metadata,
            "timestamp": datetime.now().isoformat(),
            "event_type": "alert"
        }
        
        await redis_manager.publish_event("alerts", alert_data)
        
        return {
            "alert_id": str(alert_id),
            "status": "created",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check PostgreSQL
        pg_status = "ok"
        try:
            await db_manager.execute_query("SELECT 1")
        except:
            pg_status = "error"
        
        # Check Redis
        redis_status = "ok"
        try:
            await redis_manager.redis_client.ping()
        except:
            redis_status = "error"
        
        # Check WebSocket connections
        active_users = len(connection_manager.active_connections)
        
        return {
            "status": "healthy" if pg_status == "ok" and redis_status == "ok" else "unhealthy",
            "postgresql": pg_status,
            "redis": redis_status,
            "active_websocket_users": active_users,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =====================================================
# DATABASE HELPER FUNCTIONS
# =====================================================

async def get_recent_detection_events(user_id: str, limit: int = 50, offset: int = 0):
    """Get recent detection events from database"""
    try:
        results = await db_manager.execute_function(
            "get_detection_events", user_id, limit, offset
        )
        
        return [
            {
                "id": str(row["id"]),
                "session_id": row["session_id"],
                "image_url": row["image_url"],
                "status": row["status"],
                "action": row["action"],
                "location": row["location"],
                "confidence_score": float(row["confidence_score"]) if row["confidence_score"] else None,
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"❌ Error getting detection events: {e}")
        return []

async def get_ai_summary_from_db(user_id: str, period_days: int = 7):
    """Get AI summary from database"""
    try:
        results = await db_manager.execute_function(
            "get_ai_summary", user_id, period_days
        )
        
        if results:
            row = results[0]
            return {
                "total_detection_sessions": row["total_detection_sessions"],
                "progress_compared_to_last_week": float(row["progress_compared_to_last_week"]),
                "daily_activity_log": row["daily_activity"],
                "ai_summary": row["ai_summary"]
            }
        
        return {
            "total_detection_sessions": 0,
            "progress_compared_to_last_week": 0.0,
            "daily_activity_log": [],
            "ai_summary": "No data available"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting AI summary: {e}")
        return {}

async def get_daily_activity_from_db(user_id: str, days: int = 7):
    """Get daily activity from database"""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        results = await db_manager.execute_function(
            "get_daily_activity", user_id, start_date, end_date
        )
        
        return [
            {
                "date": row["date"].isoformat(),
                "status": row["status"],
                "total_detections": row["total_detections"],
                "activity_score": float(row["activity_score"]) if row["activity_score"] else 0.0
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"❌ Error getting daily activity: {e}")
        return []

async def insert_detection_event_to_db(event: DetectionEvent):
    """Insert detection event to database"""
    try:
        results = await db_manager.execute_function(
            "insert_detection_event",
            event.user_id,
            event.session_id,
            event.image_url,
            event.status,
            event.action,
            json.dumps(event.location),
            event.confidence_score,
            json.dumps(event.ai_metadata) if event.ai_metadata else None
        )
        
        return results[0]["insert_detection_event"] if results else None
        
    except Exception as e:
        logger.error(f"❌ Error inserting detection event: {e}")
        raise

async def insert_alert_to_db(alert: AlertMessage):
    """Insert alert to database"""
    try:
        query = """
        INSERT INTO alerts (user_id, alert_type, severity, message, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """
        
        results = await db_manager.execute_query(
            query,
            alert.user_id,
            alert.alert_type,
            alert.severity,
            alert.message,
            json.dumps(alert.metadata) if alert.metadata else None
        )
        
        return results[0]["id"] if results else None
        
    except Exception as e:
        logger.error(f"❌ Error inserting alert: {e}")
        raise

# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
