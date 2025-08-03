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
    def from_env(cls) -> 'IMOUCameraConfig':
        """Tạo config từ environment variables"""
        return cls(
            rtsp_url=os.getenv('CAMERA_RTSP_URL', 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1'),
            username=os.getenv('CAMERA_USERNAME', 'admin'),
            password=os.getenv('CAMERA_PASSWORD', 'L2C37340'),
            ip_address=os.getenv('CAMERA_IP', '192.168.8.122'),
            port=int(os.getenv('CAMERA_PORT', '554')),
            frame_width=int(os.getenv('VIDEO_FRAME_WIDTH', '640')),
            frame_height=int(os.getenv('VIDEO_FRAME_HEIGHT', '480')),
            fps=int(os.getenv('VIDEO_FPS', '30')),
            motion_threshold=int(os.getenv('MOTION_DETECTION_THRESHOLD', '50')),
            confidence_threshold=float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.5'))
        )
    
    def get_rtsp_url(self) -> str:
        """Lấy RTSP URL đầy đủ"""
        if self.rtsp_url.startswith('rtsp://'):
            return self.rtsp_url
        
        # Tạo RTSP URL từ thông tin cơ bản
        return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.port}/cam/realmonitor?channel=1&subtype=0"
    
    def validate(self) -> bool:
        """Kiểm tra tính hợp lệ của config"""
        if not self.ip_address or not self.username or not self.password:
            return False
        
        if self.frame_width <= 0 or self.frame_height <= 0:
            return False
            
        if self.fps <= 0:
            return False
            
        return True

# Tạo instance global config
camera_config = IMOUCameraConfig.from_env()
