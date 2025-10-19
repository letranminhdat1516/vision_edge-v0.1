"""
Snapshot Service for Healthcare Vision Edge System
Handles creating snapshots and snapshot images with MinIO integration.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker

from ..storage.minio_service import get_minio_service

# Import models with relative paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.generated.snapshots import Snapshots
from models.generated.snapshot_images import SnapshotImages

logger = logging.getLogger(__name__)

class SnapshotService:
    """Service for managing snapshots and snapshot images"""
    
    def __init__(self, database_url: str):
        """Initialize the snapshot service"""
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.minio_service = get_minio_service()
        logger.info("SnapshotService initialized")
    
    def create_detection_snapshot(
        self,
        camera_id: str,
        user_id: str,
        event_type: str,
        confidence: float,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Create snapshot and upload image when detection occurs
        
        Args:
            camera_id: Camera UUID
            user_id: User UUID
            event_type: Type of detection (fall, seizure, etc.)
            confidence: Detection confidence
            frame: OpenCV frame to save
            metadata: Additional metadata
        
        Returns:
            Tuple of (snapshot_id, image_id)
        """
        db = self.SessionLocal()
        try:
            # Generate UUIDs
            snapshot_id = str(uuid.uuid4())
            image_id = str(uuid.uuid4())
            
            # Upload image to MinIO
            logger.info(f"Uploading {event_type} detection image to MinIO...")
            object_name, cloud_url, file_size = self.minio_service.upload_frame_image(
                frame=frame,
                camera_id=camera_id,
                event_type=event_type,
                confidence=confidence,
                user_id=user_id,  # Pass user_id for folder organization
                metadata={
                    'snapshot_id': snapshot_id,
                    'user_id': user_id,
                    **(metadata or {})
                }
            )
            
            # Create snapshot record
            snapshot = Snapshots(
                snapshot_id=snapshot_id,
                camera_id=camera_id,
                user_id=user_id,
                metadata=str({
                    'event_type': event_type,
                    'confidence': confidence,
                    'detection_time': datetime.now().isoformat(),
                    **(metadata or {})
                }),
                capture_type=event_type,
                captured_at=datetime.now(),
                processed_at=datetime.now(),
                is_processed=True
            )
            
            # Create snapshot image record
            snapshot_image = SnapshotImages(
                image_id=image_id,
                snapshot_id=snapshot_id,
                image_path=object_name,  # MinIO object name
                cloud_url=cloud_url,
                created_at=datetime.now(),
                file_size=str(file_size)
            )
            
            # Save to database
            db.add(snapshot)
            db.add(snapshot_image)
            db.commit()
            
            logger.info(f"✅ Successfully created {event_type} snapshot: {snapshot_id}")
            logger.info(f"📸 Image uploaded to MinIO: {object_name}")
            logger.info(f"🔗 Cloud URL: {cloud_url}")
            
            return snapshot_id, image_id
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating detection snapshot: {e}")
            raise
        finally:
            db.close()
    
    def create_manual_snapshot(
        self,
        camera_id: str,
        user_id: str,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Create manual snapshot (not from detection)
        
        Args:
            camera_id: Camera UUID
            user_id: User UUID
            frame: OpenCV frame to save
            metadata: Additional metadata
        
        Returns:
            Tuple of (snapshot_id, image_id)
        """
        return self.create_detection_snapshot(
            camera_id=camera_id,
            user_id=user_id,
            event_type="manual",
            confidence=1.0,
            frame=frame,
            metadata=metadata
        )
    
    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot details by ID"""
        db = self.SessionLocal()
        try:
            snapshot = db.query(Snapshots).filter(Snapshots.snapshot_id == snapshot_id).first()
            if not snapshot:
                return None
            
            # Get associated images
            images = db.query(SnapshotImages).filter(SnapshotImages.snapshot_id == snapshot_id).all()
            
            return {
                'snapshot': snapshot.to_dict(),
                'images': [img.to_dict() for img in images]
            }
        finally:
            db.close()
    
    def get_user_snapshots(
        self,
        user_id: str,
        limit: int = 50,
        event_type: Optional[str] = None
    ) -> list:
        """Get snapshots for a user"""
        db = self.SessionLocal()
        try:
            query = db.query(Snapshots).filter(Snapshots.user_id == user_id)
            
            if event_type:
                query = query.filter(Snapshots.capture_type == event_type)
            
            snapshots = query.order_by(Snapshots.captured_at.desc()).limit(limit).all()
            
            result = []
            for snapshot in snapshots:
                # Get associated images
                images = db.query(SnapshotImages).filter(
                    SnapshotImages.snapshot_id == snapshot.snapshot_id
                ).all()
                
                result.append({
                    'snapshot': snapshot.to_dict(),
                    'images': [img.to_dict() for img in images]
                })
            
            return result
        finally:
            db.close()
    
    def get_camera_snapshots(
        self,
        camera_id: str,
        limit: int = 50,
        event_type: Optional[str] = None
    ) -> list:
        """Get snapshots for a camera"""
        db = self.SessionLocal()
        try:
            query = db.query(Snapshots).filter(Snapshots.camera_id == camera_id)
            
            if event_type:
                query = query.filter(Snapshots.capture_type == event_type)
            
            snapshots = query.order_by(Snapshots.captured_at.desc()).limit(limit).all()
            
            result = []
            for snapshot in snapshots:
                # Get associated images
                images = db.query(SnapshotImages).filter(
                    SnapshotImages.snapshot_id == snapshot.snapshot_id
                ).all()
                
                result.append({
                    'snapshot': snapshot.to_dict(),
                    'images': [img.to_dict() for img in images]
                })
            
            return result
        finally:
            db.close()
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete snapshot and associated images"""
        db = self.SessionLocal()
        try:
            # Get snapshot images to delete from MinIO
            images = db.query(SnapshotImages).filter(SnapshotImages.snapshot_id == snapshot_id).all()
            
            # Delete images from MinIO
            for image in images:
                image_path = getattr(image, 'image_path', None)
                if image_path:
                    self.minio_service.delete_image(str(image_path))
            
            # Delete from database
            db.query(SnapshotImages).filter(SnapshotImages.snapshot_id == snapshot_id).delete()
            db.query(Snapshots).filter(Snapshots.snapshot_id == snapshot_id).delete()
            db.commit()
            
            logger.info(f"Successfully deleted snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting snapshot: {e}")
            return False
        finally:
            db.close()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        db = self.SessionLocal()
        try:
            # Database stats
            total_snapshots = db.query(Snapshots).count()
            total_images = db.query(SnapshotImages).count()
            
            # Event type breakdown
            fall_snapshots = db.query(Snapshots).filter(Snapshots.capture_type == 'fall').count()
            seizure_snapshots = db.query(Snapshots).filter(Snapshots.capture_type == 'seizure').count()
            manual_snapshots = db.query(Snapshots).filter(Snapshots.capture_type == 'manual').count()
            
            # MinIO stats
            minio_stats = self.minio_service.get_storage_stats()
            
            return {
                'database': {
                    'total_snapshots': total_snapshots,
                    'total_images': total_images,
                    'fall_snapshots': fall_snapshots,
                    'seizure_snapshots': seizure_snapshots,
                    'manual_snapshots': manual_snapshots
                },
                'minio': minio_stats
            }
        finally:
            db.close()

# Singleton instance
_snapshot_service = None

def get_snapshot_service(database_url: Optional[str] = None) -> SnapshotService:
    """Get singleton snapshot service instance"""
    global _snapshot_service
    if _snapshot_service is None:
        if not database_url:
            import os
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found in environment variables")
        _snapshot_service = SnapshotService(database_url)
    return _snapshot_service