"""
Simple Database Camera Loader
Direct database access for camera configurations
"""

import os
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

class SimpleCameraDB:
    """Simple direct database access for cameras"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                print("✅ Database connected for camera config")
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
    
    def get_all_active_cameras(self) -> List[Dict]:
        """Get all active cameras from database"""
        if not self.engine:
            print("❌ No database connection")
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT camera_id, camera_name, rtsp_url, location_in_room, 
                           resolution, fps, camera_type, status, is_online
                    FROM cameras 
                    WHERE status = 'active' AND is_online = true
                    ORDER BY created_at ASC
                """))
                
                cameras = []
                for row in result:
                    cameras.append({
                        'id': row[0],
                        'name': row[1], 
                        'rtsp_url': row[2],
                        'location': row[3],
                        'resolution': row[4] or '640x480',
                        'fps': row[5] or 30,
                        'type': row[6],
                        'status': row[7],
                        'is_online': row[8]
                    })
                
                print(f"✅ Found {len(cameras)} active cameras in database")
                return cameras
                
        except Exception as e:
            print(f"❌ Error loading cameras: {e}")
            return []
    
    def get_camera_by_id(self, camera_id: str) -> Optional[Dict]:
        """Get specific camera by ID"""
        cameras = self.get_all_active_cameras()
        return next((cam for cam in cameras if cam['id'] == camera_id), None)
    
    def get_allowed_cameras(self) -> List[Dict]:
        """Get cameras based on MONITOR_ALLOWED_CAM_IDS"""
        all_cameras = self.get_all_active_cameras()
        
        # Debug: print all camera IDs
        print(f"📋 All camera IDs in database:")
        for cam in all_cameras:
            print(f"   - {cam['id']} ({cam['name']})")
        
        # Check if specific camera IDs are allowed
        allowed_ids = os.getenv('MONITOR_ALLOWED_CAM_IDS', '').strip()
        print(f"🔍 Allowed IDs from .env: {allowed_ids}")
        
        if allowed_ids:
            allowed_list = [cam_id.strip() for cam_id in allowed_ids.split(',')]
            print(f"🎯 Filtered to allowed IDs: {allowed_list}")
            filtered_cameras = [cam for cam in all_cameras if cam['id'] in allowed_list]
            print(f"✅ Found {len(filtered_cameras)} matching cameras")
            return filtered_cameras
        
        # Return all active cameras if no restriction
        print("✅ No ID restriction - returning all cameras")
        return all_cameras

# Global instance
camera_db = SimpleCameraDB()