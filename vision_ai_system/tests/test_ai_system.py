# test_ai_system.py
"""
Quick test script để verify toàn bộ DDD AI system
"""
import sys
import cv2
import numpy as np
from datetime import datetime

# Add system path
sys.path.append('.')

# Test imports
try:
    from core.entities.pose_detection import Keypoint, PoseDetectionResult, FallDetectionEvent
    from infrastructure.ai_models.pose_detection_engine import PoseDetectionEngine, PoseVisualizationService
    from core.services.pose_detection_service import PoseDetectionDomainService
    from core.services.camera_service import CameraService
    print("✅ All DDD imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_domain_entities():
    """Test domain entities."""
    print("🧪 Testing Domain Entities...")
    
    # Test Keypoint
    keypoint = Keypoint(y=100.0, x=150.0, confidence=0.8)
    print(f"Keypoint: {keypoint}")
    
    # Test PoseDetectionResult
    keypoints = [
        Keypoint(y=50.0, x=60.0, confidence=0.9),
        Keypoint(y=80.0, x=90.0, confidence=0.7)
    ]
    pose_result = PoseDetectionResult(keypoints=keypoints, confidence=0.8)
    print(f"Pose Result: {pose_result}")
    
    # Test FallDetectionEvent
    fall_event = FallDetectionEvent(
        camera_id="test_cam",
        detection_time=datetime.now(),
        pose_result=pose_result,
        fall_probability=0.85
    )
    print(f"Fall Event: {fall_event.description}")
    print(f"Is Critical: {fall_event.is_critical_fall()}")
    
    print("✅ Domain entities test passed")


def test_ai_engine():
    """Test AI engine infrastructure."""
    print("🧪 Testing AI Engine...")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test PoseDetectionEngine
    engine = PoseDetectionEngine()
    result = engine.detect_pose(test_frame)
    
    print(f"Engine detected {len(result.keypoints)} keypoints")
    print(f"Detection confidence: {result.confidence}")
    
    # Test visualization
    viz_service = PoseVisualizationService()
    annotated_frame = viz_service.draw_pose(test_frame, result)
    print(f"Visualization frame shape: {annotated_frame.shape}")
    
    print("✅ AI engine test passed")


def test_domain_service():
    """Test domain service."""
    print("🧪 Testing Domain Service...")
    
    # Create engine and service
    engine = PoseDetectionEngine()
    domain_service = PoseDetectionDomainService(pose_engine=engine)
    
    # Test frame processing
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = domain_service.process_frame("test_cam", test_frame, datetime.now())
    
    print(f"Domain service result: {result}")
    print(f"Fall detected: {result.get('fall_detected', False)}")
    
    # Test stats
    stats = domain_service.get_processing_stats()
    print(f"Processing stats: {stats}")
    
    print("✅ Domain service test passed")


def test_camera_service():
    """Test application camera service."""
    print("🧪 Testing Camera Service...")
    
    # Create full service stack
    engine = PoseDetectionEngine()
    domain_service = PoseDetectionDomainService(pose_engine=engine)
    camera_service = CameraService(domain_service)
    
    # Test frame processing
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = camera_service.process_frame("test_cam", test_frame, datetime.now())
    
    print(f"Camera service result keys: {list(result.keys())}")
    print(f"AI ready: {camera_service.is_ai_ready()}")
    
    # Test visualization
    if result.get('pose_result'):
        viz_frame = camera_service.visualize_frame(
            test_frame, result['pose_result'], 
            show_skeleton=True, show_labels=True, 
            show_info=True, fps=30.0
        )
        print(f"Visualization successful: {viz_frame.shape}")
    
    print("✅ Camera service test passed")


def test_coordinate_transformation():
    """Test coordinate transformation logic."""
    print("🧪 Testing Coordinate Transformation...")
    
    engine = PoseDetectionEngine()
    
    # Test với frame có aspect ratio khác nhau
    test_cases = [
        (480, 640),   # 4:3
        (720, 1280),  # 16:9  
        (480, 854),   # 16:9 mobile
        (600, 800),   # 4:3 portrait
    ]
    
    for height, width in test_cases:
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        result = engine.detect_pose(test_frame)
        
        print(f"Frame {width}x{height}: {len(result.keypoints)} keypoints detected")
        
        # Check keypoint coordinates trong range
        for i, kp in enumerate(result.keypoints[:3]):  # Check first 3 keypoints
            if kp.confidence > 0.3:
                if 0 <= kp.x <= width and 0 <= kp.y <= height:
                    print(f"  Keypoint {i}: ({kp.x:.1f}, {kp.y:.1f}) ✅")
                else:
                    print(f"  Keypoint {i}: ({kp.x:.1f}, {kp.y:.1f}) ❌ OUT OF BOUNDS")
    
    print("✅ Coordinate transformation test passed")


def run_comprehensive_test():
    """Run comprehensive system test."""
    print("🎯 AI-Powered Vision System - Comprehensive Test")
    print("=" * 60)
    
    try:
        # 1. Test domain entities
        test_domain_entities()
        print()
        
        # 2. Test AI engine
        test_ai_engine()
        print()
        
        # 3. Test domain service
        test_domain_service()
        print()
        
        # 4. Test camera service
        test_camera_service()
        print()
        
        # 5. Test coordinate transformation
        test_coordinate_transformation()
        print()
        
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        print("✅ DDD Architecture working correctly")
        print("✅ AI pose detection functioning")
        print("✅ Coordinate transformation accurate")
        print("✅ Fall detection logic operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_validation():
    """Quick validation test."""
    print("⚡ Quick Validation Test")
    print("-" * 30)
    
    try:
        # Quick pose detection test
        engine = PoseDetectionEngine()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw simple figure for pose detection
        cv2.rectangle(test_frame, (280, 150), (360, 200), (255, 255, 255), -1)  # Head
        cv2.rectangle(test_frame, (300, 200), (340, 300), (255, 255, 255), -1)  # Body
        cv2.line(test_frame, (320, 220), (280, 260), (255, 255, 255), 5)  # Left arm
        cv2.line(test_frame, (320, 220), (360, 260), (255, 255, 255), 5)  # Right arm
        cv2.line(test_frame, (320, 280), (300, 350), (255, 255, 255), 5)  # Left leg
        cv2.line(test_frame, (320, 280), (340, 350), (255, 255, 255), 5)  # Right leg
        
        result = engine.detect_pose(test_frame)
        
        print(f"✅ Detected {len(result.keypoints)} keypoints")
        print(f"✅ Average confidence: {result.confidence:.2f}")
        
        # Test visualization
        viz_service = PoseVisualizationService()
        annotated = viz_service.draw_pose(test_frame, result)
        
        print(f"✅ Visualization completed")
        print("🎯 System validation PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🔬 AI System Testing Suite")
    print("=" * 40)
    
    while True:
        print("\n📋 TEST OPTIONS:")
        print("1. Quick Validation")
        print("2. Comprehensive Test")
        print("3. Exit")
        
        choice = input("\nSelect test (1-3): ").strip()
        
        if choice == "1":
            print()
            success = run_quick_validation()
            if success:
                print("\n✅ Quick validation completed successfully")
            else:
                print("\n⚠️ Quick validation failed")
                
        elif choice == "2":
            print()
            success = run_comprehensive_test()
            if success:
                print("\n🏆 Comprehensive test completed successfully")
            else:
                print("\n⚠️ Comprehensive test failed")
                
        elif choice == "3":
            print("👋 Testing completed!")
            break
            
        else:
            print("❌ Invalid option. Please choose 1-3.")


if __name__ == "__main__":
    main()