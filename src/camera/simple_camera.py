"""
Simple IMOU Camera Stream Handler - No external dependencies
Xử lý kết nối và stream video từ camera IMOU
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple


class SimpleIMOUCamera:
    """Simple IMOU Camera Stream Handler - không dùng loguru"""
    
    def __init__(self, config):
        """Initialize camera với config"""
        self.config = config if hasattr(config, 'get') else {'url': config}
        
        # Camera properties
        self.cap = None
        self.connected = False
        self.streaming = False
        
        # Frame properties
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.stream_thread = None
        
        # Stats
        self.frame_count = 0
        self.failed_frames = 0
        
    def connect(self) -> bool:
        """Kết nối tới camera IMOU với enhanced error handling"""
        try:
            # Extract URL properly
            if hasattr(self.config, 'get'):
                url = self.config.get('url', '')
            else:
                url = str(self.config) if self.config else ''
                
            if not url:
                print("❌ No camera URL provided")
                return False
                
            print(f"📹 Connecting to camera: {url}")
            
            # Enhanced RTSP connection with proper type handling
            self.cap = cv2.VideoCapture(url)
            
            # Thiết lập timeout cho RTSP connection
            if self.cap and hasattr(self.cap, 'set'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Note: Timeout properties may not be available in all OpenCV versions
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)  # 3 second read timeout
                except:
                    pass  # Ignore if timeout properties not available
                
                # Thiết lập FPS nếu có
                if hasattr(self.config, 'get'):
                    fps = self.config.get('fps')
                    if fps and isinstance(fps, (int, float)):
                        self.cap.set(cv2.CAP_PROP_FPS, float(fps))
                
                # Thiết lập resolution nếu có
                if hasattr(self.config, 'get'):
                    resolution = self.config.get('resolution')
                    if resolution and isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolution[0]))
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolution[1]))
            
            # Kiểm tra kết nối với detailed error reporting
            if not self.cap or not self.cap.isOpened():
                print("❌ Cannot connect to camera")
                print("   💡 Possible issues:")
                print("   - Wrong IP address or credentials")
                print("   - Camera is offline or unreachable")
                print("   - RTSP port is blocked")
                print("   - Authentication failed (401 Unauthorized)")
                print(f"   📝 URL used: {url}")
                self._suggest_troubleshooting(url)
                return False
            
            # Test đọc frame với timeout
            import time
            timeout_start = time.time()
            timeout_duration = 10  # 10 seconds
            
            ret, frame = None, None
            while time.time() < timeout_start + timeout_duration:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.1)
            
            if not ret or frame is None:
                print("❌ Cannot read frame from camera")
                print("   💡 Camera connected but no video stream available")
                print("   - Check camera stream settings")
                print("   - Verify RTSP stream path")
                return False
            
            print("✅ Camera connected successfully!")
            print(f"   📐 Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
            self.connected = True
            
            # Start streaming thread
            self.start_stream()
            
            return True
            
        except Exception as e:
            print(f"❌ Camera connection error: {e}")
            print("   💡 Troubleshooting:")
            print("   - Check network connectivity")
            print("   - Verify camera credentials")
            print("   - Test camera URL in VLC player")
            return False
    
    def _suggest_troubleshooting(self, url: str):
        """Suggest specific troubleshooting based on URL"""
        if "401" in str(url) or "Unauthorized" in str(url):
            print("   🔐 Authentication Issue:")
            print("   - Check username and password")
            print("   - Verify camera login credentials")
        if "192.168" in url:
            print("   🌐 Network Issue:")
            print("   - Ping the camera IP address")
            print("   - Check if camera is on same network")
        print(f"   🧪 Test in VLC: Open Network Stream -> {url}")
    
    def start_stream(self):
        """Bắt đầu stream camera"""
        if self.streaming:
            return
            
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print("📹 Camera streaming started")
    
    def stop_stream(self):
        """Dừng stream camera"""
        self.streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        print("📹 Camera streaming stopped")
    
    def _stream_loop(self):
        """Main stream loop"""
        retry_count = 0
        max_retries = 5
        
        while self.streaming and self.connected:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # Update current frame
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        self.frame_count += 1
                        retry_count = 0  # Reset retry count on success
                        
                    else:
                        self.failed_frames += 1
                        print(f"⚠️ Failed to read frame (failed: {self.failed_frames})")
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            print("⚠️ Too many failed frames, attempting reconnect...")
                            if not self._attempt_reconnect():
                                break
                            retry_count = 0
                        
                        time.sleep(0.1)  # Small delay on failure
                
                else:
                    print("⚠️ Camera not available")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Stream loop error: {e}")
                time.sleep(1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy frame hiện tại"""
        if not self.connected:
            print("❌ Camera not connected")
            return None
            
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _attempt_reconnect(self) -> bool:
        """Thử kết nối lại camera"""
        try:
            print("🔄 Attempting camera reconnect...")
            
            # Đóng connection cũ
            if self.cap:
                self.cap.release()
            
            time.sleep(2)  # Wait before reconnect
            
            # Kết nối lại
            url = self.config.get('url', self.config) if hasattr(self.config, 'get') else self.config
            self.cap = cv2.VideoCapture(url)
            
            if self.cap.isOpened():
                # Test frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print("✅ Camera reconnected successfully!")
                    return True
            
            print("❌ Camera reconnect failed")
            return False
            
        except Exception as e:
            print(f"❌ Reconnect error: {e}")
            return False
    
    def disconnect(self):
        """Ngắt kết nối camera"""
        print("🔌 Disconnecting camera...")
        
        self.streaming = False
        self.connected = False
        
        # Stop stream thread
        self.stop_stream()
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("🔌 Camera disconnected")
    
    def get_stats(self) -> dict:
        """Lấy thống kê camera"""
        return {
            'connected': self.connected,
            'streaming': self.streaming,
            'frame_count': self.frame_count,
            'failed_frames': self.failed_frames,
            'frame_shape': self.current_frame.shape if self.current_frame is not None else None
        }


# Alias for compatibility
IMOUCameraStream = SimpleIMOUCamera
