# main.py
"""
Vision AI System - DDD Clean Architecture with AI-Powered Fall Detection
Main entry point for the healthcare monitoring system
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.services.camera_service import CameraService
from core.services.pose_detection_service import PoseDetectionDomainService
from infrastructure.ai_models.pose_detection_engine_threadsafe import PoseDetectionEngine, PoseVisualizationService
from infrastructure.camera.camera_manager import CameraManager
from infrastructure.persistence.camera_repository import SqlCameraRepository
from runtime.monitor_runner import MonitorRunner
from config.settings import get_sample_cameras
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.generated_all import Base
# Import all models to ensure relationships are properly registered
from models import generated_all  # This imports all models and relationships

# Database setup from environment
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    print("⚠️ DATABASE_URL not configured, using sample cameras mode")
    engine = None
    SessionLocal = None

def create_ai_services_for_database(cam_repo) -> CameraService:
    """Tạo AI services cho database mode."""
    print("🔧 Initializing AI-powered services for database mode...")
    
    # Infrastructure Layer - AI Engine
    pose_engine = PoseDetectionEngine()
    
    # Domain Layer - Business Logic  
    pose_detection_service = PoseDetectionDomainService(pose_engine=pose_engine)
    
    # Application Layer - Camera Service với cả repo và AI
    camera_service = CameraService(
        repo=cam_repo, 
        pose_detection_service=pose_detection_service
    )
    
    print("✅ AI services for database mode initialized successfully")
    return camera_service


def create_ai_services_for_samples() -> CameraService:
    """Tạo AI services cho sample mode (không cần database)."""
    print("🔧 Initializing AI-powered services for sample mode...")
    
    # Infrastructure Layer - AI Engine
    pose_engine = PoseDetectionEngine()
    
    # Domain Layer - Business Logic  
    pose_detection_service = PoseDetectionDomainService(pose_engine=pose_engine)
    
    # Application Layer - Camera Service chỉ với AI (không có repo)
    camera_service = CameraService(
        repo=None,
        pose_detection_service=pose_detection_service
    )
    
    print("✅ AI services for sample mode initialized successfully")
    return camera_service


def run_with_database():
    """Run system with database connection."""
    print("🗄️ Running with database mode...")
    
    if not DATABASE_URL or not SessionLocal:
        raise ValueError("DATABASE_URL environment variable is required for database mode")
    
    db = SessionLocal()
    try:
        cam_repo = SqlCameraRepository(db)
        cam_service = create_ai_services_for_database(cam_repo)  # Use AI-powered service with repo
        
        # Create enhanced camera manager
        cam_manager = CameraManager(cam_service)
        
        # Create and configure monitor runner
        monitor_runner = MonitorRunner(cam_service, cam_manager)
        
        print("🎯 AI-Powered monitoring system starting...")
        monitor_runner.run()
        
        print("Press Ctrl+C to stop...")
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping system...")
        if 'monitor_runner' in locals():
            monitor_runner.stop()
    finally:
        db.close()
        print("✅ Database connection closed")


def run_sample_mode():
    """Run system with sample cameras (no database)."""
    print("📹 Running in sample cameras mode...")
    
    try:
        # Create AI services
        camera_service = create_ai_services_for_samples()
        
        # Get sample cameras
        cameras = get_sample_cameras()
        print(f"📋 Loaded {len(cameras)} sample cameras:")
        for cam in cameras:
            print(f"   - {cam.camera_name}: {cam.rtsp_url}")
        
        # Create camera manager
        camera_manager = CameraManager(camera_service)
        
        # Add cameras
        success_count = 0
        for cam in cameras:
            if camera_manager.add_camera(cam):
                success_count += 1
        
        if success_count == 0:
            print("❌ No cameras were successfully added")
            return
        
        print(f"✅ Successfully added {success_count}/{len(cameras)} cameras")
        
        # Display system info
        print("\n" + "=" * 60)
        print("🤖 AI SYSTEM STATUS")
        print("=" * 60)
        print(f"🧠 AI Engine: {'READY' if camera_service.is_ai_ready() else 'NOT READY'}")
        print(f"📹 Cameras: {success_count} active")
        print(f"🎯 Pose Detection: ENABLED")
        print(f"🚨 Fall Detection: ENABLED")
        print("=" * 60)
        
        # Start monitoring
        print("\n🎬 Starting real-time monitoring...")
        print("Controls:")
        print("  S - Toggle skeleton visualization")
        print("  L - Toggle keypoint labels")  
        print("  V - Toggle full visualization")
        print("  I - Toggle info display")
        print("  Q - Quit system")
        print("=" * 60)
        
        camera_manager.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n⏹️ System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


def run():
    """Main application entry point với AI integration."""
    print("🚀 Vision AI System - Fall Detection Enabled")
    print("=" * 60)
    
    # Check if monitor is enabled
    if not os.getenv("MONITOR_ENABLED", "true").lower() == "true":
        print("⚠️ Monitor is disabled in .env")
        return
    
    # Show menu options
    print("\n📋 SYSTEM MODES:")
    print("1. Database Mode (requires DATABASE_URL)")
    print("2. Sample Cameras Mode (no database)")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            if DATABASE_URL:
                run_with_database()
                break
            else:
                print("❌ DATABASE_URL not configured. Please set DATABASE_URL environment variable.")
                continue
                
        elif choice == "2":
            run_sample_mode()
            break
            
        elif choice == "3":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please choose 1-3.")
    
    print("🔚 System shutdown complete")

if __name__ == "__main__":
    run()
