"""
Clean Camera Service using Generated SQLAlchemy Models
Only uses database cameras table - no config files
"""

import os
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Import generated model
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.generated.cameras import Cameras, Base

load_dotenv()

class CameraService:
    """Simple camera service using only database"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        self.SessionLocal = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database connection"""
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                self.SessionLocal = sessionmaker(bind=self.engine)
                print("✅ Camera Service: Database connected")
            except Exception as e:
                print(f"❌ Camera Service: Database error: {e}")
    
    def get_active_cameras(self) -> List[Dict]:
        """Get all active cameras"""
        if not self.SessionLocal:
            return []
        
        try:
            with self.SessionLocal() as session:
                cameras = session.query(Cameras).filter(
                    Cameras.status == 'active',
                    Cameras.is_online == True
                ).all()
                
                result = []
                for cam in cameras:
                    result.append({
                        'id': cam.camera_id,
                        'name': cam.camera_name,
                        'rtsp_url': cam.rtsp_url,
                        'location': cam.location_in_room,
                        'resolution': cam.resolution or '1920x1080',
                        'fps': cam.fps or 30,
                        'type': cam.camera_type,
                        'user_id': cam.user_id
                    })
                
                print(f"✅ Found {len(result)} active cameras")
                return result
                
        except Exception as e:
            print(f"❌ Error loading cameras: {e}")
            return []
    
    def get_cameras_for_user(self, user_id: str) -> List[Dict]:
        """Get cameras for specific user"""
        if not self.SessionLocal:
            return []
        
        try:
            with self.SessionLocal() as session:
                cameras = session.query(Cameras).filter(
                    Cameras.user_id == user_id,
                    Cameras.status == 'active'
                ).all()
                
                result = []
                for cam in cameras:
                    result.append({
                        'id': str(cam.camera_id),
                        'name': cam.camera_name,
                        'rtsp_url': cam.rtsp_url,
                        'location': cam.location_in_room,
                        'resolution': cam.resolution or '1920x1080',
                        'fps': cam.fps or 30,
                        'type': cam.camera_type,
                        'user_id': str(cam.user_id)
                    })
                
                print(f"✅ Found {len(result)} cameras for user {user_id}")
                return result
                
        except Exception as e:
            print(f"❌ Error loading cameras for user: {e}")
            return []
    
    def get_primary_camera(self, user_id: str) -> Optional[Dict]:
        """Get first available camera for user"""
        cameras = self.get_cameras_for_user(user_id)
        if cameras:
            return cameras[0]
        return None

# Global instance
camera_service = CameraService()