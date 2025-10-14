# main.py
"""
Vision AI System - DDD Clean Architecture
Main entry point for the healthcare monitoring system
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.services.camera_service import CameraService
from infrastructure.camera.camera_manager import CameraManager
from infrastructure.persistence.camera_repository import SqlCameraRepository
from runtime.monitor_runner import MonitorRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.generated_all import Base
# Import all models to ensure relationships are properly registered
from models import generated_all  # This imports all models and relationships

# Database setup from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def run():
    """Main application entry point"""
    print("Starting Vision AI System...")
    
    # Check if monitor is enabled
    if not os.getenv("MONITOR_ENABLED", "true").lower() == "true":
        print("Monitor is disabled in .env")
        return
    
    db = SessionLocal()
    cam_repo = SqlCameraRepository(db)
    cam_service = CameraService(cam_repo)
    cam_manager = CameraManager(on_frame=lambda *args: None)  # Placeholder
    
    # Create and configure monitor runner
    monitor_runner = MonitorRunner(cam_service, cam_manager)
    
    try:
        monitor_runner.run()
        
        print("Press Ctrl+C to stop...")
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping system...")
        monitor_runner.stop()
    finally:
        db.close()
        print("System shutdown complete")

if __name__ == "__main__":
    run()
