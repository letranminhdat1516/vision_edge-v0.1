"""
FastAPI Server for Alarm API
Expose REST API endpoints để trigger/stop alarm

Run: uvicorn alarm_fastapi_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

# Import services
from infrastructure.services.alarm_api import alarm_api
from service.postgresql_healthcare_service import PostgreSQLHealthcareService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Healthcare Alarm API",
    description="API đơn giản để bật/tắt alarm",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model - CHỈ 1 API DUY NHẤT
class AlarmControlRequest(BaseModel):
    event_id: str
    user_id: str
    camera_id: str
    enabled: bool  # true = bật alarm, false = tắt alarm

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Alarm API Server...")
    
    # Initialize PostgreSQL service
    try:
        postgresql_service = PostgreSQLHealthcareService()
        alarm_api.set_postgresql_service(postgresql_service)
        logger.info("✅ PostgreSQL service connected")
    except Exception as e:
        logger.error(f"❌ Failed to connect PostgreSQL: {e}")
    
    logger.info("✅ Alarm API Server ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Healthcare Alarm API",
        "version": "2.0.0"
    }

@app.post("/api/alarm/control")
async def control_alarm(request: AlarmControlRequest):
    """
    API DUY NHẤT để bật/tắt alarm
    
    Body:
    {
        "event_id": "uuid-of-event",
        "user_id": "uuid-of-user",
        "camera_id": "uuid-of-camera",
        "enabled": true  // true = BẬT alarm, false = TẮT alarm
    }
    
    Example:
    - BẬT: {"event_id": "abc", "user_id": "user1", "camera_id": "cam1", "enabled": true}
    - TẮT: {"event_id": "abc", "user_id": "user1", "camera_id": "cam1", "enabled": false}
    """
    try:
        if request.enabled:
            # BẬT ALARM
            result = alarm_api.trigger_alarm(
                event_id=request.event_id,
                user_id=request.user_id,
                camera_id=request.camera_id,
                triggered_by="api",
                reason="Manual trigger via API"
            )
            action = "BẬT"
        else:
            # TẮT ALARM
            result = alarm_api.stop_alarm(
                event_id=request.event_id,
                user_id=request.user_id,
                stopped_by="api",
                reason="Manual stop via API"
            )
            action = "TẮT"
        
        if result['success']:
            return {
                "success": True,
                "action": action,
                "message": f"Đã {action} alarm thành công",
                "data": result
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    
    except Exception as e:
        logger.error(f"Error in control_alarm endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alarm/status")
async def get_alarm_status():
    """
    Get current alarm status
    
    Returns:
    {
        "is_playing": true/false,
        "active_alarms": [...],
        "timestamp": "..."
    }
    """
    try:
        result = alarm_api.get_alarm_status()
        return result
    
    except Exception as e:
        logger.error(f"Error in get_alarm_status endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
