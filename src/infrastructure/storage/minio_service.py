"""
MinIO Storage Service for Healthcare Vision Edge System
Handles image upload, storage, and management in MinIO object storage.
"""

import os
import io
import cv2
import uuid
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)

class MinIOService:
    """Service for managing image storage in MinIO"""
    
    def __init__(self):
        """Initialize MinIO client"""
        self.endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = os.getenv('MINIO_SECRET_KEY')
        self.bucket_name = os.getenv('MINIO_BUCKET_NAME', 'healthcare-snapshots')
        self.secure = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
        
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO credentials not found in environment variables")
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        logger.info(f"MinIO Service initialized - Endpoint: {self.endpoint}, Bucket: {self.bucket_name}")
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket already exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise
    
    def upload_frame_image(
        self,
        frame: np.ndarray,
        camera_id: str,
        event_type: str,
        confidence: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, int]:
        """
        Upload frame image to MinIO with user-organized folder structure
        
        Args:
            frame: OpenCV frame (numpy array)
            camera_id: Camera identifier
            event_type: Type of event (fall, seizure, etc.)
            confidence: Detection confidence
            user_id: User identifier for folder organization
            metadata: Additional metadata
        
        Returns:
            Tuple of (object_name, cloud_url, file_size)
        """
        try:
            # Generate unique filename with enhanced naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_id = str(uuid.uuid4())[:8]
            
            # Simplified structure: user_id/filename with detailed filename
            # Filename format: {event_type}_{camera_id}_{timestamp}_{id}_{confidence}.jpg
            filename = f"{event_type}_{camera_id}_{timestamp}_{image_id}_{confidence:.3f}.jpg"
            
            if user_id:
                object_name = f"{user_id}/{filename}"
            else:
                # Fallback to old structure if no user_id
                object_name = f"unknown_user/{filename}"
            
            # Convert frame to JPEG bytes
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("Failed to encode frame to JPEG")
            
            # Convert to bytes
            image_bytes = buffer.tobytes()
            file_size = len(image_bytes)
            
            # Create file-like object
            image_stream = io.BytesIO(image_bytes)
            
            # Upload to MinIO with user-organized structure
            result = self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=image_stream,
                length=file_size,
                content_type='image/jpeg',
                metadata={
                    'user_id': user_id or 'unknown',
                    'camera_id': camera_id,
                    'event_type': event_type,
                    'confidence': str(confidence),
                    'uploaded_at': datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
            
            # Generate cloud URL
            cloud_url = f"http{'s' if self.secure else ''}://{self.endpoint}/{self.bucket_name}/{object_name}"
            
            logger.info(f"Successfully uploaded image: {object_name} ({file_size} bytes)")
            return object_name, cloud_url, file_size
            
        except Exception as e:
            logger.error(f"Error uploading image to MinIO: {e}")
            raise
    
    def upload_raw_image(
        self,
        image_path: str,
        camera_id: str,
        event_type: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, int]:
        """
        Upload existing image file to MinIO with user-organized folder structure
        
        Args:
            image_path: Path to image file
            camera_id: Camera identifier
            event_type: Type of event
            user_id: User identifier for folder organization
            metadata: Additional metadata
        
        Returns:
            Tuple of (object_name, cloud_url, file_size)
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Generate unique filename with enhanced naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(image_path)[1]
            
            # Simplified structure: user_id/filename with detailed filename
            # Filename format: {event_type}_{camera_id}_{timestamp}_{id}.{ext}
            filename = f"{event_type}_{camera_id}_{timestamp}_{image_id}{file_extension}"
            
            if user_id:
                object_name = f"{user_id}/{filename}"
            else:
                # Fallback to old structure if no user_id
                object_name = f"unknown_user/{filename}"
            
            # Get file size
            file_size = os.path.getsize(image_path)
            
            # Upload to MinIO
            result = self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=image_path,
                content_type='image/jpeg',
                metadata={
                    'user_id': user_id or 'unknown',
                    'camera_id': camera_id,
                    'event_type': event_type,
                    'uploaded_at': datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
            
            # Generate cloud URL
            cloud_url = f"http{'s' if self.secure else ''}://{self.endpoint}/{self.bucket_name}/{object_name}"
            
            logger.info(f"Successfully uploaded image: {object_name} ({file_size} bytes)")
            return object_name, cloud_url, file_size
            
        except Exception as e:
            logger.error(f"Error uploading image file to MinIO: {e}")
            raise
    
    def delete_image(self, object_name: str) -> bool:
        """
        Delete image from MinIO
        
        Args:
            object_name: Name of object to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Successfully deleted image: {object_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting image from MinIO: {e}")
            return False
    
    def get_image_url(self, object_name: str, expires_hours: int = 24) -> str:
        """
        Get presigned URL for image access
        
        Args:
            object_name: Name of object
            expires_hours: URL expiration time in hours
        
        Returns:
            Presigned URL
        """
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(hours=expires_hours)
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return ""
    
    def list_images(
        self,
        prefix: str = "",
        limit: int = 100
    ) -> list:
        """
        List images in bucket
        
        Args:
            prefix: Object name prefix filter
            limit: Maximum number of objects to return
        
        Returns:
            List of object information
        """
        try:
            objects = []
            for obj in self.client.list_objects(self.bucket_name, prefix=prefix):
                objects.append({
                    'object_name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                })
                if len(objects) >= limit:
                    break
            return objects
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return []
    
    def list_user_images(
        self,
        user_id: str,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        List images for a specific user with optional filters
        
        Args:
            user_id: User identifier
            event_type: Optional event type filter (fall, seizure, etc.)
            camera_id: Optional camera filter
            limit: Maximum number of objects to return
        
        Returns:
            List of user's images
        """
        try:
            # Search in user folder
            prefix = f"{user_id}/"
            
            objects = []
            for obj in self.client.list_objects(self.bucket_name, prefix=prefix):
                # Parse filename to extract metadata
                filename = (obj.object_name or "").split('/')[-1] if obj.object_name else ""
                
                # Extract metadata from filename: {event_type}_{camera_id}_{timestamp}_{id}_{confidence}.jpg
                filename_parts = filename.split('_')
                
                object_info = {
                    'object_name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag,
                    'cloud_url': f"http{'s' if self.secure else ''}://{self.endpoint}/{self.bucket_name}/{obj.object_name}",
                    'filename': filename
                }
                
                # Extract metadata from filename if properly formatted
                if len(filename_parts) >= 3:  # event_type_camera_id_timestamp_...
                    extracted_event_type = filename_parts[0]
                    extracted_camera_id = filename_parts[1]
                    
                    object_info.update({
                        'user_id': user_id,
                        'event_type': extracted_event_type,
                        'camera_id': extracted_camera_id
                    })
                    
                    # Apply filters
                    if event_type and extracted_event_type != event_type:
                        continue
                    if camera_id and extracted_camera_id != camera_id:
                        continue
                
                objects.append(object_info)
                if len(objects) >= limit:
                    break
            
            return objects
        except Exception as e:
            logger.error(f"Error listing user images: {e}")
            return []
    
    def get_user_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get storage statistics for a specific user
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with user storage stats
        """
        try:
            prefix = f"{user_id}/"
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix))
            total_size = sum(obj.size or 0 for obj in objects)
            
            # Count by event type from filename
            event_counts = {}
            camera_counts = {}
            
            for obj in objects:
                filename = (obj.object_name or "").split('/')[-1] if obj.object_name else ""
                # Extract from filename: {event_type}_{camera_id}_{timestamp}_{id}_{confidence}.jpg
                filename_parts = filename.split('_')
                
                if len(filename_parts) >= 2:
                    event_type = filename_parts[0]
                    camera_id = filename_parts[1]
                    
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                    camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
            
            return {
                'user_id': user_id,
                'total_objects': len(objects),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'event_type_counts': event_counts,
                'camera_counts': camera_counts,
                'bucket_name': self.bucket_name
            }
        except Exception as e:
            logger.error(f"Error getting user storage stats: {e}")
            return {}
    
    def ensure_user_folder(self, user_id: str) -> bool:
        """
        Ensure user folder exists (MinIO creates folders automatically when uploading)
        This is mainly for documentation purposes as MinIO handles folder creation automatically
        
        Args:
            user_id: User identifier
        
        Returns:
            True if user can upload to their folder
        """
        try:
            # MinIO creates folders automatically when uploading files
            # So we just need to check if we can access the bucket
            return self.client.bucket_exists(self.bucket_name)
        except Exception as e:
            logger.error(f"Error checking user folder access: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage stats
        """
        try:
            objects = list(self.client.list_objects(self.bucket_name))
            total_size = sum(obj.size or 0 for obj in objects)  # Handle None values
            
            return {
                'bucket_name': self.bucket_name,
                'total_objects': len(objects),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'endpoint': self.endpoint
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

# Singleton instance
_minio_service = None

def get_minio_service() -> MinIOService:
    """Get singleton MinIO service instance"""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinIOService()
    return _minio_service