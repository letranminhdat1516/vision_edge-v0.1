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
            logger.error("MinIO credentials not found in environment variables")
            self.client = None
            return
        
        # Initialize MinIO client
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                logger.info(f"Creating bucket: {self.bucket_name}")
                self.client.make_bucket(self.bucket_name)
            else:
                logger.info(f"Bucket exists: {self.bucket_name}")
                
        except Exception as e:
            logger.error(f"MinIO client initialization failed: {e}")
            self.client = None

    def upload_frame_image(self, frame: np.ndarray, user_id: str, camera_id: str, 
                          event_type: str, confidence: float, 
                          metadata: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, str, int]]:
        """Upload frame image to MinIO with simplified folder structure: user_id/filename
        
        Returns:
            Tuple of (object_name, cloud_url, file_size) if successful, None if failed
        """
        
        if not self.client:
            logger.error("MinIO client not available")
            return None
            
        try:
            # Generate simplified object name: user_id/event_camera_timestamp_id_confidence.jpg
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            confidence_str = f'{confidence:.3f}'
            
            object_name = f"{user_id}/{event_type}_{camera_id}_{timestamp}_{unique_id}_{confidence_str}.jpg"
            
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                logger.error("Failed to encode frame as JPEG")
                return None
            
            # Convert to bytes
            image_bytes = buffer.tobytes()
            file_size = len(image_bytes)
            
            # Debug: Log image info
            logger.info(f"Image encoded successfully: size={file_size} bytes, frame_shape={frame.shape}")
            
            # Upload to MinIO with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create fresh stream for each attempt
                    image_stream = io.BytesIO(image_bytes)
                    
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
                            'upload_time': datetime.now().strftime('%Y%m%d_%H%M%S')  # Simplified timestamp
                        }
                    )
                    
                    # Generate cloud URL using public domain
                    cloud_url = f"https://nas.cicca.dpdns.org/cdn-image/{object_name}"
                    
                    logger.info(f"Successfully uploaded image to MinIO: {object_name}")
                    return object_name, cloud_url, file_size
                    
                except Exception as upload_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"Upload attempt {attempt + 1} failed: {upload_error}, retrying...")
                        continue
                    else:
                        logger.error(f"Error uploading image to MinIO: {upload_error}")
                        return None
            
        except Exception as e:
            logger.error(f"Unexpected error in upload_frame_image: {e}")
            return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from MinIO"""
        if not self.client:
            return {'error': 'MinIO client not available'}
            
        try:
            # Get bucket statistics
            objects = list(self.client.list_objects(self.bucket_name))
            total_objects = len(objects)
            total_size = sum(obj.size for obj in objects if obj.size)
            
            return {
                'bucket_name': self.bucket_name,
                'total_objects': total_objects,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'connection_status': 'connected'
            }
        except Exception as e:
            return {'error': f'Failed to get storage stats: {e}'}

    def test_connection(self) -> bool:
        """Test MinIO connection"""
        if not self.client:
            return False
            
        try:
            self.client.bucket_exists(self.bucket_name)
            return True
        except Exception as e:
            logger.error(f"MinIO connection test failed: {e}")
            return False

# Global instance
_minio_service_instance = None

def get_minio_service() -> Optional[MinIOService]:
    """Get or create MinIO service instance"""
    global _minio_service_instance
    if _minio_service_instance is None:
        try:
            _minio_service_instance = MinIOService()
            if _minio_service_instance.client is None:
                logger.warning("MinIO service created but client is not available")
        except Exception as e:
            logger.error(f"Failed to create MinIO service: {e}")
            _minio_service_instance = None
    
    return _minio_service_instance