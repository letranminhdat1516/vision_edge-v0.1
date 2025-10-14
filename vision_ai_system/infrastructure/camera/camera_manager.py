# infrastructure/camera/camera_manager.py
import cv2
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque

from models.generated_all import Cameras
from infrastructure.camera.camera_device import CameraDevice
from core.services.camera_service import CameraService


class CameraManager:
    """Quản lý nhiều camera với real-time pose detection và monitoring."""

    def __init__(self, camera_service: CameraService):
        self.camera_service = camera_service
        self.devices: Dict[str, CameraDevice] = {}
        self.running = False
        self.threads: Dict[str, threading.Thread] = {}
        
        # Performance tracking
        self.fps_counters: Dict[str, deque] = {}
        self.last_frame_times: Dict[str, float] = {}
        
        # Monitoring settings
        self.show_visualization = True
        self.show_skeleton = True
        self.show_labels = False
        self.show_info = True

    def add_camera(self, cam: Cameras) -> bool:
        """Thêm camera vào quản lý."""
        try:
            device = CameraDevice(cam)
            if device.open():
                camera_id = str(cam.camera_id)
                self.devices[camera_id] = device
                self.fps_counters[camera_id] = deque(maxlen=30)
                self.last_frame_times[camera_id] = time.time()
                print(f"✅ Camera {cam.camera_name} ({camera_id}) added successfully")
                return True
            else:
                print(f"❌ Failed to open camera {cam.camera_name}")
                return False
        except Exception as e:
            print(f"❌ Error adding camera {cam.camera_name}: {e}")
            return False

    def remove_camera(self, camera_id: str):
        """Xóa camera khỏi quản lý."""
        if camera_id in self.devices:
            self.devices[camera_id].release()
            del self.devices[camera_id]
            
            if camera_id in self.fps_counters:
                del self.fps_counters[camera_id]
            if camera_id in self.last_frame_times:
                del self.last_frame_times[camera_id]
                
            print(f"✅ Camera {camera_id} removed")

    def start_monitoring(self):
        """Bắt đầu monitoring tất cả camera."""
        if not self.camera_service.is_ai_ready():
            print("❌ AI engine not ready. Cannot start monitoring.")
            return False
            
        print(f"🎥 Starting monitoring for {len(self.devices)} cameras...")
        print(f"📋 Visualization settings:")
        print(f"   Show visualization: {self.show_visualization}")
        print(f"   Show skeleton: {self.show_skeleton}")  
        print(f"   Show labels: {self.show_labels}")
        print(f"   Show info: {self.show_info}")
        print(f"📋 Controls: S-skeleton, L-labels, V-visualization, I-info, Q-quit")
        
        self.running = True
        
        # Tạo thread cho mỗi camera
        for camera_id, device in self.devices.items():
            thread = threading.Thread(
                target=self._monitor_camera,
                args=(camera_id, device),
                daemon=True
            )
            self.threads[camera_id] = thread
            thread.start()
            print(f"🎯 Started monitoring thread for camera {camera_id}")
        
        # Main control loop
        self._main_control_loop()
        
        return True

    def stop_monitoring(self):
        """Dừng monitoring."""
        print("⏹️ Stopping camera monitoring...")
        self.running = False
        
        # Wait for threads to finish
        for camera_id, thread in self.threads.items():
            if thread.is_alive():
                print(f"⏳ Waiting for camera {camera_id} thread to finish...")
                thread.join(timeout=2.0)
        
        self.threads.clear()
        cv2.destroyAllWindows()
        print("✅ Camera monitoring stopped")

    def _monitor_camera(self, camera_id: str, device: CameraDevice):
        """Monitor một camera cụ thể."""
        window_name = f"Camera {device.meta.camera_name} ({camera_id})"
        
        frame_skip_count = 0
        max_frame_skip = 2  # Process every 3rd frame để tăng FPS
        
        try:
            while self.running:
                start_time = time.time()
                
                # Đọc frame
                ret, frame = device.read()
                if not ret or frame is None:
                    print(f"❌ Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Skip frames để tăng performance
                frame_skip_count += 1
                should_process = frame_skip_count >= max_frame_skip
                
                if should_process:
                    frame_skip_count = 0
                    
                    # Process frame với AI
                    result = self.camera_service.process_frame(
                        camera_id, frame, datetime.now()
                    )
                    
                    # Call MonitorRunner callback if set
                    if hasattr(self, '_on_frame') and self._on_frame:
                        self._on_frame(camera_id, frame, datetime.now())
                    
                    # Visualization
                    if self.show_visualization and result.get('pose_result'):
                        frame = self.camera_service.visualize_frame(
                            frame,
                            result['pose_result'],
                            self.show_skeleton,
                            self.show_labels,
                            self.show_info,
                            self._get_fps(camera_id)
                        )
                    elif self.show_info:
                        # Chỉ vẽ FPS info
                        fps = self._get_fps(camera_id)
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Frame bị skip, chỉ vẽ FPS
                    if self.show_info:
                        fps = self._get_fps(camera_id)
                        cv2.putText(frame, f"FPS: {fps:.1f} (SKIP)", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Hiển thị frame
                cv2.imshow(window_name, frame)
                
                # Track FPS
                frame_time = time.time() - start_time
                self.fps_counters[camera_id].append(frame_time)
                self.last_frame_times[camera_id] = time.time()
                
                # Handle window events
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    self._handle_key_press(key)
                
                # Limit FPS để không quá tải CPU
                time.sleep(0.01)
                
        except Exception as e:
            print(f"❌ Error in camera {camera_id} monitoring: {e}")
        finally:
            cv2.destroyWindow(window_name)

    def _main_control_loop(self):
        """Main control loop để handle global controls."""
        try:
            while self.running:
                time.sleep(0.1)
                
                # Check if any thread is still alive
                alive_threads = [t for t in self.threads.values() if t.is_alive()]
                if not alive_threads:
                    print("⚠️ All camera threads stopped")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        finally:
            self.stop_monitoring()

    def _handle_key_press(self, key: int):
        """Handle keyboard controls."""
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
        elif key == ord('s'):  # Toggle skeleton
            self.show_skeleton = not self.show_skeleton
            print(f"Skeleton: {'ON' if self.show_skeleton else 'OFF'}")
        elif key == ord('l'):  # Toggle labels
            self.show_labels = not self.show_labels
            print(f"Labels: {'ON' if self.show_labels else 'OFF'}")
        elif key == ord('v'):  # Toggle visualization
            self.show_visualization = not self.show_visualization
            print(f"Visualization: {'ON' if self.show_visualization else 'OFF'}")
        elif key == ord('i'):  # Toggle info
            self.show_info = not self.show_info
            print(f"Info: {'ON' if self.show_info else 'OFF'}")

    def _get_fps(self, camera_id: str) -> float:
        """Tính FPS cho camera."""
        if camera_id not in self.fps_counters or len(self.fps_counters[camera_id]) == 0:
            return 0.0
        
        frame_times = list(self.fps_counters[camera_id])
        if len(frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(frame_times) / len(frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_monitoring_stats(self) -> dict:
        """Lấy thống kê monitoring."""
        stats = {
            'total_cameras': len(self.devices),
            'running': self.running,
            'camera_fps': {},
            'camera_service_stats': self.camera_service.get_processing_stats()
        }
        
        for camera_id in self.devices:
            stats['camera_fps'][camera_id] = self._get_fps(camera_id)
        
        return stats

    def reload_cameras(self, cameras: List[Cameras]):
        """Reload danh sách camera."""
        print("🔄 Reloading cameras...")
        
        # Stop current monitoring
        was_running = self.running
        if was_running:
            self.stop_monitoring()
        
        # Clear current devices
        for camera_id in list(self.devices.keys()):
            self.remove_camera(camera_id)
        
        # Add new cameras
        success_count = 0
        for cam in cameras:
            if self.add_camera(cam):
                success_count += 1
        
        print(f"✅ Reloaded {success_count}/{len(cameras)} cameras")
        
        # Restart monitoring if was running
        if was_running and success_count > 0:
            self.start_monitoring()
        
        return success_count > 0

    # Compatibility methods for old MonitorRunner interface
    def load_from_list(self, cameras: List[Cameras]):
        """Load cameras from list (compatibility method)."""
        print(f"📋 Loading {len(cameras)} cameras...")
        for cam in cameras:
            self.add_camera(cam)
    
    def start_all(self):
        """Start all cameras (compatibility method)."""
        return self.start_monitoring()
    
    def stop_all(self):
        """Stop all cameras (compatibility method)."""
        return self.stop_monitoring()
    
    # For MonitorRunner callback compatibility
    def set_on_frame_callback(self, callback):
        """Set frame callback (compatibility method)."""
        self.on_frame = callback
    
    @property 
    def on_frame(self):
        """Get current frame callback."""
        return getattr(self, '_on_frame', None)
    
    @on_frame.setter
    def on_frame(self, callback):
        """Set frame callback."""
        self._on_frame = callback
