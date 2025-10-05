#!/usr/bin/env python3
"""
Test Multi-Camera Connection
Quick test to verify both cameras are accessible
"""

import cv2
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_camera_connection(name, rtsp_url):
    """Test individual camera connection"""
    print(f"📹 Testing {name}...")
    print(f"   URL: {rtsp_url}")
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"❌ {name}: Cannot open RTSP stream")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ {name}: Cannot read frame")
            cap.release()
            return False
        
        print(f"✅ {name}: Connected successfully!")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ {name}: Connection error - {e}")
        return False

def test_multi_camera_system():
    """Test the complete multi-camera system"""
    print("🎥 Multi-Camera Connection Test")
    print("="*50)
    
    # Camera configurations
    cameras = {
        "Living Room Camera": "rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1",
        "Bedroom Camera": "rtsp://admin:L2C37340@192.168.8.123:554/cam/realmonitor?channel=1&subtype=1"
    }
    
    # Test each camera
    results = {}
    for name, url in cameras.items():
        results[name] = test_camera_connection(name, url)
        print()
    
    # Summary
    print("📊 CONNECTION SUMMARY:")
    print("-" * 30)
    connected_count = sum(results.values())
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅ Connected" if success else "❌ Failed"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {connected_count}/{total_count} cameras connected")
    
    if connected_count == 0:
        print("\n⚠️ NO CAMERAS CONNECTED!")
        print("Please check:")
        print("1. Camera IP addresses (192.168.8.122, 192.168.8.123)")
        print("2. Network connectivity")
        print("3. RTSP credentials (admin:L2C37340)")
        print("4. Camera RTSP service enabled")
        return False
    elif connected_count == 1:
        print("\n⚠️ ONLY 1 CAMERA CONNECTED!")
        print("Multi-camera system will work but with reduced redundancy")
        print("Check the failed camera connection")
        return True
    else:
        print("\n✅ ALL CAMERAS CONNECTED!")
        print("Multi-camera system ready to start")
        return True

def test_frame_quality():
    """Test frame quality from both cameras"""
    print("\n🔍 TESTING FRAME QUALITY...")
    print("-" * 30)
    
    cameras = {
        "Camera 1": "rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1",
        "Camera 2": "rtsp://admin:L2C37340@192.168.8.123:554/cam/realmonitor?channel=1&subtype=1"
    }
    
    for name, url in cameras.items():
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = gray.mean()
                    
                    # Simple sharpness test
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness = laplacian.var()
                    
                    print(f"{name}:")
                    print(f"  Brightness: {brightness:.1f} {'✅' if 80 <= brightness <= 180 else '⚠️'}")
                    print(f"  Sharpness: {sharpness:.1f} {'✅' if sharpness > 100 else '⚠️'}")
                    
                cap.release()
        except Exception as e:
            print(f"{name}: Quality test failed - {e}")

def main():
    """Main test function"""
    print("🚀 MULTI-CAMERA SYSTEM TEST")
    print("="*50)
    
    # Test connections
    connection_success = test_multi_camera_system()
    
    if connection_success:
        # Test frame quality
        test_frame_quality()
        
        print("\n🎯 NEXT STEPS:")
        if connection_success:
            print("✅ Run the multi-camera healthcare system:")
            print("   python examples/multi_camera_healthcare_system.py")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
