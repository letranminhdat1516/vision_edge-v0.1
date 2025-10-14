# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Cameras:
    camera_id: Optional[str] = None
    user_id: str
    camera_name: str
    camera_type: str
    status: str
    is_online: bool
    created_at: datetime
    updated_at: datetime
    ip_address: Optional[str] = None
    port: Optional[int] = None
    rtsp_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    location_in_room: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    last_ping: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None

