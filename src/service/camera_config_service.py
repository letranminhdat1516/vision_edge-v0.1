"""
Camera Configuration Service
Load camera configurations from database instead of hardcoded values
"""

import os
import sys
from typing import Dict, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add src to path for models import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from models.generated.cameras import Cameras

load_dotenv()

class CameraConfigService:
    """Service to manage camera configurations from database"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        self.Session = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection"""
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                self.Session = sessionmaker(bind=self.engine)
                print("✅ Camera Config Service: Database connected")
            except Exception as e:
                print(f"⚠️ Camera Config Service: Database connection failed: {e}")
                self.Session = None
    
    def get_all_cameras(self) -> List[Dict]:
        """Get all active cameras from database"""
        if not self.Session:
            return self._get_fallback_cameras()
        
        try:
            with self.Session() as session:
                cameras = session.query(Cameras).filter(
                    Cameras.status == 'active',
                    Cameras.is_online == True
                ).all()
                
                return [self._camera_to_dict(camera) for camera in cameras]
                
        except Exception as e:
            print(f"⚠️ Error loading cameras from database: {e}")
            return self._get_fallback_cameras()
    
    def get_camera_by_id(self, camera_id: str) -> Optional[Dict]:
        """Get specific camera by ID"""
        if not self.Session:
            fallback_cameras = self._get_fallback_cameras()
            return next((cam for cam in fallback_cameras if cam['id'] == camera_id), None)
        
        try:
            with self.Session() as session:
                camera = session.query(Cameras).filter(
                    Cameras.camera_id == camera_id,
                    Cameras.status == 'active'
                ).first()
                
                return self._camera_to_dict(camera) if camera else None
                
        except Exception as e:
            print(f"⚠️ Error loading camera {camera_id}: {e}")
            return None
    
    def get_allowed_camera_ids(self) -> List[str]:
        """Get list of allowed camera IDs from environment or database"""
        # First try environment variable
        env_ids = os.getenv('MONITOR_ALLOWED_CAM_IDS', '')
        if env_ids:
            return [cam_id.strip() for cam_id in env_ids.split(',') if cam_id.strip()]
        
        # Fallback to all active cameras
        cameras = self.get_all_cameras()
        return [cam['id'] for cam in cameras]
    
    def _camera_to_dict(self, camera: Cameras) -> Dict:
        """Convert database camera object to dictionary"""
        if not camera:
            return {}
            
        return {
            'id': camera.camera_id,
            'name': camera.camera_name,
            'rtsp_url': camera.rtsp_url,
            'type': camera.camera_type,
            'location': camera.location_in_room,
            'resolution': camera.resolution,
            'fps': camera.fps,
            'status': camera.status,
            'is_online': camera.is_online,
            'ip_address': camera.ip_address,
            'port': camera.port,
            'username': camera.username,
            'password': camera.password
        }
    
    def _get_fallback_cameras(self) -> List[Dict]:
        """Fallback camera configuration when database is not available - DISABLED"""
        print("❌ WARNING: Fallback cameras requested - this should not happen!")
        print("   System should use clean_camera_service instead")
        return []  # Return empty to force error and find where this is used

# Global instance
camera_config_service = CameraConfigService()