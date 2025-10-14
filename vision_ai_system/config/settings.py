# config/settings.py
import os, json
from dotenv import load_dotenv

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
