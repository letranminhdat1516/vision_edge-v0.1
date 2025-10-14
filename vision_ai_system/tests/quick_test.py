# quick_test.py
"""
Quick test script để kiểm tra AI system
"""
import sys
sys.path.append('.')

def test_imports():
    """Test all imports work correctly."""
    try:
        from core.entities.pose_detection import Keypoint, PoseDetectionResult, FallDetectionEvent
        from infrastructure.ai_models.pose_detection_engine import PoseDetectionEngine
        from core.services.pose_detection_service import PoseDetectionDomainService
        from core.services.camera_service import CameraService
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ai_engine():
    """Test AI engine basic functionality."""
    try:
        import numpy as np
        from infrastructure.ai_models.pose_detection_engine import PoseDetectionEngine
        
        engine = PoseDetectionEngine()
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = engine.detect_pose(test_frame, "test_cam")
        print(f"✅ AI engine test passed - detected {len(result.keypoints)} keypoints")
        return True
    except Exception as e:
        print(f"❌ AI engine test failed: {e}")
        return False

def main():
    print("🧪 Quick AI System Test")
    print("=" * 30)
    
    success = True
    
    # Test imports
    print("1. Testing imports...")
    success &= test_imports()
    
    # Test AI engine
    print("\n2. Testing AI engine...")
    success &= test_ai_engine()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 All tests PASSED! System ready.")
        print("Run: python main.py")
    else:
        print("❌ Some tests FAILED!")

if __name__ == "__main__":
    main()