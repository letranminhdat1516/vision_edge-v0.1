"""
Camera IMOU Package
Xử lý kết nối và stream từ camera IMOU
Now uses database-first approach for camera configuration
"""

from .config import IMOUCameraConfig
from .simple_camera import SimpleIMOUCamera

# Note: camera_config deprecated - use database-based configuration
__all__ = ['IMOUCameraConfig', 'SimpleIMOUCamera']
