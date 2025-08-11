#!/usr/bin/env python3
"""
Auto Healthcare Monitor - Chạy với default config
"""

import sys
import os
from pathlib import Path

# Add examples to path
sys.path.append(str(Path(__file__).parent / "examples"))

from advanced_healthcare_monitor import AdvancedHealthcareMonitor

def main():
    print("🏥 Auto Healthcare Monitor Starting...")
    print("✅ Configuration: Keypoints ON, Statistics ON")
    print("🌐 WebSocket: ws://localhost:8086")
    print("📺 Dual windows will open")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Initialize monitor with default settings
    monitor = AdvancedHealthcareMonitor(
        show_keypoints=True,
        show_statistics=True,
        websocket_url="ws://localhost:8086",
        enable_streaming=True,
        enable_api_integration=True
    )
    
    # Run monitoring
    success = monitor.run_monitoring()
    
    if success:
        print("\n✅ Healthcare monitoring completed successfully!")
    else:
        print("\n❌ Healthcare monitoring failed!")

if __name__ == "__main__":
    main()
