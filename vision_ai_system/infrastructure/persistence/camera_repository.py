# infrastructure/persistence/camera_repository.py

from typing import List
from models.generated_all import Cameras as CameraModel  # Import from generated_all instead
from sqlalchemy.orm import Session

class SqlCameraRepository:
    """Triển khai repository để tương tác DB thật."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self) -> List[CameraModel]:
        """Lấy toàn bộ camera trong DB."""
        return self.session.query(CameraModel).all()

    def get_all_active(self) -> List[CameraModel]:
        """Lấy toàn bộ camera active trong DB."""
        return self.session.query(CameraModel).filter(CameraModel.status == 'active').all()

    def get_by_user_id(self, user_id: str) -> List[CameraModel]:
        """Lấy camera theo user_id."""
        return self.session.query(CameraModel).filter(CameraModel.user_id == user_id).all()

    def update_status(self, camera_id: str, status: str):
        """Cập nhật trạng thái camera (active/offline)."""
        self.session.query(CameraModel).filter(CameraModel.camera_id == camera_id).update({"status": status})
        self.session.commit()
