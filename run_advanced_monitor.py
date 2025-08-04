#!/usr/bin/env python3
"""
Advanced Healthcare Monitor - Run Script
Dual Detection System: Fall + Seizure Detection
"""

import sys
import os
from pathlib import Path

def main():
    """Run the advanced healthcare monitor"""
    # Change to examples directory
    examples_dir = Path(__file__).parent
    os.chdir(examples_dir)
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(examples_dir.parent))
    
    try:
        # Import and run the advanced monitor
        from examples.advanced_healthcare_monitor import main as monitor_main
        monitor_main()
        
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error running healthcare monitor: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure camera is connected and accessible")
        print("2. Check if all required packages are installed")
        print("3. Verify YOLO model is available")
        print("4. Check VSViG model files in models/VSViG/")

if __name__ == "__main__":
    main()
