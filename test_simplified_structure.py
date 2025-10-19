"""
Test Simplified MinIO User Folder Structure
"""

import os
import cv2
import numpy as np
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

try:
    from infrastructure.storage.minio_service import get_minio_service
    
    print("🧪 Testing Simplified MinIO User Folder Structure...")
    
    # Get MinIO service
    minio_service = get_minio_service()
    print("✅ MinIO service connected!")
    
    # Test user info
    test_user_id = os.getenv('DEFAULT_USER_ID', 'test-user-123')
    test_camera_id = 'camera-001'
    
    print(f"👤 Test User ID: {test_user_id[:12]}...")
    print(f"📹 Test Camera ID: {test_camera_id}")
    
    # Test different event types
    test_events = ['fall', 'seizure', 'manual']
    uploaded_files = []
    
    print(f"\n📸 Testing image uploads...")
    for i, event_type in enumerate(test_events):
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, f"{event_type.upper()} TEST", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(test_frame, f"Camera: {test_camera_id}", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(test_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Test upload with simplified structure
        object_name, cloud_url, file_size = minio_service.upload_frame_image(
            frame=test_frame,
            camera_id=test_camera_id,
            event_type=event_type,
            confidence=0.80 + (i * 0.05),  # Different confidence for each
            user_id=test_user_id,
            metadata={'test': True, 'sequence': i + 1}
        )
        
        uploaded_files.append({
            'event_type': event_type,
            'object_name': object_name,
            'cloud_url': cloud_url,
            'file_size': file_size
        })
        
        print(f"   ✅ {event_type}: {object_name}")
    
    print(f"\n📁 New Folder Structure:")
    print(f"   📦 {os.getenv('MINIO_BUCKET_NAME')}/")
    print(f"   └── {test_user_id[:8]}.../ (user folder)")
    
    for file_info in uploaded_files:
        filename = file_info['object_name'].split('/')[-1]
        print(f"       ├── {filename}")
    
    # Test user image listing with filters
    print(f"\n🔍 Testing image listing and filtering...")
    
    # List all user images
    all_images = minio_service.list_user_images(test_user_id)
    print(f"   📋 Total images for user: {len(all_images)}")
    
    # List by event type
    for event_type in test_events:
        event_images = minio_service.list_user_images(test_user_id, event_type=event_type)
        print(f"   📋 {event_type} images: {len(event_images)}")
        if event_images:
            filename = event_images[0]['filename']
            print(f"       Example: {filename}")
    
    # List by camera
    camera_images = minio_service.list_user_images(test_user_id, camera_id=test_camera_id)
    print(f"   📋 Images from {test_camera_id}: {len(camera_images)}")
    
    # Test storage stats
    print(f"\n📊 User Storage Statistics:")
    stats = minio_service.get_user_storage_stats(test_user_id)
    print(f"   Total Images: {stats.get('total_objects', 0)}")
    print(f"   Total Storage: {stats.get('total_size_mb', 0)} MB")
    print(f"   Event Breakdown: {stats.get('event_type_counts', {})}")
    print(f"   Camera Breakdown: {stats.get('camera_counts', {})}")
    
    print(f"\n🎯 Filename Convention Analysis:")
    for file_info in uploaded_files:
        filename = file_info['object_name'].split('/')[-1]
        parts = filename.split('_')
        if len(parts) >= 5:
            event_type = parts[0]
            camera_id = parts[1]
            timestamp = f"{parts[2]}_{parts[3]}"
            image_id = parts[4]
            confidence = parts[5].replace('.jpg', '')
            
            print(f"   📄 {filename}")
            print(f"       Event: {event_type} | Camera: {camera_id}")
            print(f"       Time: {timestamp} | ID: {image_id} | Conf: {confidence}")
    
    print(f"\n🌐 Access Examples:")
    for file_info in uploaded_files[:2]:  # Show first 2
        print(f"   🔗 {file_info['cloud_url']}")
    
    print(f"\n✅ Simplified folder structure test completed!")
    print(f"📝 Benefits:")
    print(f"   ✅ Cleaner folder structure: user_id/filename")
    print(f"   ✅ All info in filename: event_camera_time_id_confidence")
    print(f"   ✅ Easy filtering by event type or camera")
    print(f"   ✅ Simplified navigation and management")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()