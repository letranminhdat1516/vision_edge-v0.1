from dataclasses import dataclass
from typing import Tuple

@dataclass
class CameraConfig:
    url: str
    buffer_size: int
    fps: int
    resolution: Tuple[int, int]
    auto_reconnect: bool
