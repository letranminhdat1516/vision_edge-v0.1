"""
Test snapshot creation separately
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.postgresql_healthcare_service import postgresql_service

def test_create_snapshot():
    """Test creating a snapshot"""
    print("🔍 Testing snapshot creation...")
    
    # Use real IDs
    camera_id = '3c0b0000-0000-4000-8000-000000000001'
    room_id = '2d0a0000-0000-4000-8000-000000000001'
    user_id = '34e92ef3-1300-40d0-a0e0-72989cf30121'
    
    snapshot_id = postgresql_service._create_default_snapshot(
        camera_id=camera_id,
        room_id=room_id,
        user_id=user_id
    )
    
    print(f"Snapshot ID result: {snapshot_id} (type: {type(snapshot_id)})")
    
    if snapshot_id:
        print(f"✅ Snapshot created successfully: {snapshot_id}")
    else:
        print("❌ Failed to create snapshot")

if __name__ == "__main__":
    test_create_snapshot()
