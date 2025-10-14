# config/settings.py
import os, json
from typing import List
from dotenv import load_dotenv
from models.generated_all import Cameras

load_dotenv(override=True)

def getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","on")

def getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    # camera monitor
    MONITOR_AUTO_START = getenv_bool("MONITOR_AUTO_START", False)
    MONITOR_ENABLED = getenv_bool("MONITOR_ENABLED", True)
    MONITOR_ALLOWED_CAM_IDS = [s.strip() for s in os.getenv("MONITOR_ALLOWED_CAM_IDS","").split(",") if s.strip()]
    MONITOR_SAMPLE_EVERY_N = getenv_int("MONITOR_SAMPLE_EVERY_N", 2)
    MONITOR_MOTION_PREFILTER = getenv_bool("MONITOR_MOTION_PREFILTER", True)
    MONITOR_MAX_WORKERS = getenv_int("MONITOR_MAX_WORKERS", 2)
    MONITOR_QUEUE_SIZE = getenv_int("MONITOR_QUEUE_SIZE", 200)

    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "").strip()
    try:
        CAMERA_USER_MAP = json.loads(os.getenv("CAMERA_USER_MAP_JSON","{}"))
    except Exception:
        CAMERA_USER_MAP = {}

settings = Settings()


def get_sample_cameras() -> List[Cameras]:
    """Tạo sample cameras cho testing."""
    import uuid
    from datetime import datetime
    
    cameras = []
    
    # Camera 1 - Webcam
    cam1 = Cameras()
    cam1.camera_id = uuid.uuid4()
    cam1.user_id = uuid.uuid4()  # Sample user ID
    cam1.camera_name = "Test Webcam"
    cam1.rtsp_url = "0"  # Default webcam
    cam1.location_in_room = "Living Room"
    cam1.status = "active"
    cam1.is_online = True
    cam1.created_at = datetime.now()
    cameras.append(cam1)
    
    # Camera 2 - RTSP Stream (if available)
    cam2 = Cameras()
    cam2.camera_id = uuid.uuid4()
    cam2.user_id = uuid.uuid4()  # Sample user ID
    cam2.camera_name = "RTSP Camera"
    cam2.rtsp_url = "rtsp://admin:password@192.168.1.100/stream1"
    cam2.location_in_room = "Bedroom"
    cam2.status = "active"
    cam2.is_online = True
    cam2.created_at = datetime.now()
    cameras.append(cam2)
    
    # Camera 3 - Sample video file (if exists)
    cam3 = Cameras()
    cam3.camera_id = uuid.uuid4()
    cam3.user_id = uuid.uuid4()  # Sample user ID
    cam3.camera_name = "Video File"
    cam3.rtsp_url = "data/sample_video.mp4"
    cam3.location_in_room = "Test Room"
    cam3.status = "active"
    cam3.is_online = True
    cam3.created_at = datetime.now()
    cameras.append(cam3)
    
    return cameras
