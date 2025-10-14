# core/services/camera_service.py
from typing import List
from datetime import datetime
from models.generated_all import Cameras  # Import from generated_all

class CameraService:
    """Xử lý nghiệp vụ camera (load, kiểm tra trạng thái, ...)."""

    def __init__(self, repo):
        self.repo = repo  # repository đọc từ DB hoặc JSON test

    def get_active_cameras(self) -> List[Cameras]:
        """Trả về danh sách camera đang active."""
        cams = self.repo.get_all()
        return [c for c in cams if c.status == "active"]

    def update_status(self, camera_id: str, status: str):
        """Cập nhật trạng thái camera (active/offline)."""
        self.repo.update_status(camera_id, status)

    def process_frame(self, camera_id: str, frame, timestamp: datetime):
        """Process frame from camera"""
        # TODO: Implement AI model processing here
        pass
