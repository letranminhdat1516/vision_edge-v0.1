"""
Camera IMOU Configuration Module
Xử lý kết nối và cấu hình camera IMOU qua RTSP
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class IMOUCameraConfig:
    """Cấu hình camera IMOU"""
    
    # Thông tin kết nối camera
    rtsp_url: str
    username: str
    password: str
    ip_address: str
    port: int = 554
    
    # Cấu hình video stream (tối ưu cho performance)
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 20  # Giảm từ 30 xuống 20 để tăng tốc
    
    # Cấu hình hiển thị
    show_stream: bool = True
    show_keypoints: bool = True
    stream_window_name: str = "IMOU Camera - Live Stream"
    keypoints_window_name: str = "IMOU Camera - Key Points"
    
    # Cấu hình xử lý
    motion_threshold: int = 50
    confidence_threshold: float = 0.5
    
    @classmethod
    def from_database_camera(cls, camera_data: dict) -> 'IMOUCameraConfig':
        """Tạo config từ database camera data - PRIMARY METHOD"""
        # Parse resolution
        resolution = camera_data.get('resolution', '640x480')
        try:
            width, height = map(int, resolution.split('x'))
        except:
            width, height = 640, 480
        
        return cls(
            rtsp_url=camera_data.get('rtsp_url', ''),
            username=camera_data.get('username', 'admin'),  # fallback only
            password=camera_data.get('password', ''),
            ip_address=camera_data.get('ip_address', ''),
            port=camera_data.get('port', 554),
            frame_width=width,
            frame_height=height,
            fps=camera_data.get('fps', 30),
            motion_threshold=int(os.getenv('MOTION_DETECTION_THRESHOLD', '50')),
            confidence_threshold=float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.5'))
        )
    
    @classmethod
    def from_env(cls) -> 'IMOUCameraConfig':
        """DEPRECATED: Tạo config từ environment variables - use from_database_camera instead"""
        print("⚠️ Warning: from_env() is deprecated. Use from_database_camera() instead.")
        return cls(
            rtsp_url='',  # No hardcoded values
            username=os.getenv('CAMERA_USERNAME', 'admin'),
            password='',  # No hardcoded values
            ip_address='',  # No hardcoded values
            port=int(os.getenv('CAMERA_PORT', '554')),
            frame_width=int(os.getenv('VIDEO_FRAME_WIDTH', '640')),
            frame_height=int(os.getenv('VIDEO_FRAME_HEIGHT', '480')),
            fps=int(os.getenv('VIDEO_FPS', '30')),
            motion_threshold=int(os.getenv('MOTION_DETECTION_THRESHOLD', '50')),
            confidence_threshold=float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.5'))
        )
    
    def get_rtsp_url(self) -> str:
        """Lấy RTSP URL đầy đủ"""
        if self.rtsp_url and self.rtsp_url.startswith('rtsp://'):
            return self.rtsp_url
        
        # Tạo RTSP URL từ thông tin cơ bản nếu có đủ thông tin
        if self.username and self.password and self.ip_address:
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.port}/cam/realmonitor?channel=1&subtype=0"
        
        return ""
    
    def validate(self) -> bool:
        """Kiểm tra tính hợp lệ của config"""
        # Kiểm tra có RTSP URL hoặc đủ thông tin để tạo URL
        has_rtsp_url = self.rtsp_url and self.rtsp_url.startswith('rtsp://')
        has_connection_info = self.ip_address and self.username and self.password
        
        if not (has_rtsp_url or has_connection_info):
            return False
        
        if self.frame_width <= 0 or self.frame_height <= 0:
            return False
            
        if self.fps <= 0:
            return False
            
        return True

# Global config - will be replaced by database config at runtime
camera_config = None
