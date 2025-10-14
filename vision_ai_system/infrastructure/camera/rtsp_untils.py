# infrastructure/camera/rtsp_utils.py
from dataclasses import asdict
from typing import Optional
from domain.entities.generated.cameras import Cameras

def build_rtsp_url(cam: Cameras) -> Optional[str]:
    # Ưu tiên dùng rtsp_url có sẵn
    if cam.rtsp_url and cam.rtsp_url.strip():
        return cam.rtsp_url.strip()

    # Fallback: tự ghép từ các trường khác (nếu có)
    if cam.ip_address and cam.port:
        user = (cam.username or "").strip()
        pw   = (cam.password or "").strip()
        auth = f"{user}:{pw}@" if user or pw else ""
        ch   = "1"
        subtype = "0"
        # đổi path theo hãng camera nếu cần
        return f"rtsp://{auth}{cam.ip_address}:{cam.port}/cam/realmonitor?channel={ch}&subtype={subtype}"

    return None
