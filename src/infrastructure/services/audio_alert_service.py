"""
Audio Alert Service
Handles audio playback through external audio devices (Bluetooth speakers, USB speakers, etc.)
Supports automatic device detection and emergency alarm playback
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioAlertService:
    """
    Service để phát cảnh báo âm thanh qua thiết bị audio bên ngoài
    Tự động nhận diện Bluetooth speaker, USB speaker, headphones
    """
    
    def __init__(self):
        self.enabled = os.getenv('AUDIO_ALERT_ENABLED', 'true').lower() == 'true'
        self.volume = float(os.getenv('EMERGENCY_ALERT_VOLUME', '1.0'))
        
        # FIX: Use absolute path relative to this file
        current_dir = Path(__file__).parent.parent.parent  # Go up to src/
        default_sounds_dir = current_dir / 'sounds'
        self.sounds_dir = Path(os.getenv('SOUNDS_DIRECTORY', str(default_sounds_dir)))
        
        self.alert_duration = int(os.getenv('ALERT_DURATION_SECONDS', '30'))
        
        self.is_playing = False
        self.current_sound = None
        self.audio_backend = None
        self.available_devices = []
        
        if self.enabled:
            self._initialize_audio()
    
    def _initialize_audio(self):
        """Khởi tạo audio backend và detect devices"""
        try:
            # Try pygame first (cross-platform, dễ dùng)
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.audio_backend = 'pygame'
            logger.info("✅ Audio backend: pygame initialized")
            
            # Detect available devices
            self._detect_audio_devices()
            
            # Create sounds directory if not exists
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            
        except ImportError:
            logger.warning("pygame not available, trying pydub + simpleaudio")
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                import simpleaudio
                self.audio_backend = 'pydub'
                logger.info("✅ Audio backend: pydub initialized")
            except ImportError:
                logger.error("No audio backend available! Install: pip install pygame")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            self.enabled = False
    
    def _detect_audio_devices(self):
        """Phát hiện các thiết bị audio khả dụng"""
        try:
            if os.name == 'posix':  # Linux/Mac
                self._detect_linux_devices()
            elif os.name == 'nt':  # Windows
                self._detect_windows_devices()
        except Exception as e:
            logger.error(f"Failed to detect audio devices: {e}")
    
    def _detect_linux_devices(self):
        """Phát hiện audio devices trên Linux (Raspberry Pi)"""
        try:
            import subprocess
            
            # Sử dụng aplay -l để list devices
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout
                devices = []
                
                for line in output.split('\n'):
                    if 'card' in line.lower():
                        devices.append(line.strip())
                
                self.available_devices = devices
                
                logger.info("📻 Available audio devices:")
                for device in devices:
                    logger.info(f"   - {device}")
                
                # Check for Bluetooth devices
                bluetooth_devices = [d for d in devices if 'blue' in d.lower()]
                if bluetooth_devices:
                    logger.info(f"🔊 Found {len(bluetooth_devices)} Bluetooth audio device(s)")
                
        except FileNotFoundError:
            logger.warning("aplay not found, using default audio device")
        except Exception as e:
            logger.error(f"Failed to detect Linux audio devices: {e}")
    
    def _detect_windows_devices(self):
        """Phát hiện audio devices trên Windows"""
        try:
            from pycaw.pycaw import AudioUtilities
            
            devices = AudioUtilities.GetSpeakers()
            
            if devices:
                device_name = devices.GetFriendlyName()
                self.available_devices = [device_name]
                logger.info(f"📻 Default audio device: {device_name}")
            
        except ImportError:
            logger.warning("pycaw not available, using default audio device")
        except Exception as e:
            logger.error(f"Failed to detect Windows audio devices: {e}")
    
    def get_available_devices(self) -> List[str]:
        """Lấy danh sách thiết bị audio khả dụng"""
        return self.available_devices
    
    def _load_sound(self, sound_name: str = "emergency_siren.mp3"):
        """Load file âm thanh"""
        sound_path = self.sounds_dir / sound_name
        
        if not sound_path.exists():
            logger.error(f"Sound file not found: {sound_path}")
            return None
        
        try:
            if self.audio_backend == 'pygame':
                import pygame
                sound = pygame.mixer.Sound(str(sound_path))
                sound.set_volume(self.volume)
                return sound
            
            elif self.audio_backend == 'pydub':
                from pydub import AudioSegment
                sound = AudioSegment.from_file(str(sound_path))
                # Adjust volume (pydub uses dB)
                volume_db = (self.volume - 1) * 20  # Convert 0-1 to dB
                sound = sound + volume_db
                return sound
            
        except Exception as e:
            logger.error(f"Failed to load sound {sound_name}: {e}")
            return None
    
    async def play_emergency_alarm(self, user_id: str, triggered_by: str = "mobile_app", duration: int = 0) -> Dict[str, Any]:
        """
        Phát báo động khẩn cấp
        
        Args:
            user_id: ID người dùng kích hoạt
            triggered_by: Nguồn kích hoạt (mobile_app, ai_detection, manual)
            duration: Override duration in seconds (0 = use default)
        
        Returns:
            Dict với status và message
        """
        if not self.enabled:
            logger.warning("Audio alert service is disabled")
            return {"success": False, "message": "Audio service disabled"}
        
        if self.is_playing:
            logger.info("Alert already playing, stopping current alert first")
            await self.stop_alarm()
        
        # Override duration if specified
        original_duration = self.alert_duration
        if duration > 0:
            self.alert_duration = duration
        
        try:
            # Load emergency sound
            sound = self._load_sound("emergency_siren.mp3")
            
            if not sound:
                # Try fallback sound
                sound = self._load_sound("emergency_alert.wav")
            
            if not sound:
                return {"success": False, "message": "No sound file available"}
            
            logger.info(f"🚨 EMERGENCY ALARM ACTIVATED")
            logger.info(f"   User ID: {user_id}")
            logger.info(f"   Triggered by: {triggered_by}")
            logger.info(f"   Volume: {self.volume * 100:.0f}%")
            logger.info(f"   Duration: {self.alert_duration}s")
            
            # Play based on backend
            if self.audio_backend == 'pygame':
                import pygame
                sound.play(loops=-1)  # Loop indefinitely
                self.is_playing = True
                self.current_sound = sound
                
                # Schedule auto-stop
                import asyncio
                asyncio.create_task(self._auto_stop_after_duration())
            
            elif self.audio_backend == 'pydub':
                from pydub.playback import play
                import threading
                
                def play_loop():
                    while self.is_playing:
                        play(sound)
                
                self.is_playing = True
                play_thread = threading.Thread(target=play_loop, daemon=True)
                play_thread.start()
                
                # Schedule auto-stop
                import asyncio
                asyncio.create_task(self._auto_stop_after_duration())
            
            # Restore original duration
            actual_duration = self.alert_duration
            if duration > 0:
                self.alert_duration = original_duration
            
            return {
                "success": True,
                "message": "Emergency alarm activated",
                "duration": actual_duration,
                "volume": self.volume,
                "devices": len(self.available_devices),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to play emergency alarm: {e}")
            # Restore duration on error too
            if duration > 0:
                self.alert_duration = original_duration
            return {"success": False, "message": str(e)}
    
    async def _auto_stop_after_duration(self):
        """Tự động dừng sau duration"""
        import asyncio
        try:
            await asyncio.sleep(self.alert_duration)
            if self.is_playing:
                await self.stop_alarm()
                logger.info(f"⏰ Auto-stopped alarm after {self.alert_duration}s")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in auto-stop: {e}")
    
    async def stop_alarm(self) -> Dict[str, Any]:
        """
        Dừng báo động
        
        Returns:
            Dict với status và message
        """
        if not self.is_playing:
            return {"success": False, "message": "No alarm is playing"}
        
        try:
            if self.audio_backend == 'pygame':
                import pygame
                pygame.mixer.stop()
            
            self.is_playing = False
            self.current_sound = None
            
            logger.info("✅ Emergency alarm stopped")
            
            return {
                "success": True,
                "message": "Alarm stopped successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stop alarm: {e}")
            return {"success": False, "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại của service"""
        return {
            "enabled": self.enabled,
            "is_playing": self.is_playing,
            "volume": self.volume,
            "alert_duration": self.alert_duration,
            "audio_backend": self.audio_backend,
            "available_devices": len(self.available_devices),
            "sounds_directory": str(self.sounds_dir),
            "devices": self.available_devices
        }
    
    def test_audio(self) -> bool:
        """
        Test xem audio có hoạt động không
        
        Returns:
            True nếu test thành công
        """
        try:
            logger.info("🔊 Testing audio playback...")
            
            # Try to play a short beep
            if self.audio_backend == 'pygame':
                import pygame
                import numpy as np
                
                # Generate a simple beep sound
                sample_rate = 44100
                duration = 0.5  # seconds
                frequency = 440  # Hz (A note)
                
                # Generate samples
                samples = np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)
                samples = (samples * 32767).astype(np.int16)
                
                # Create stereo sound
                stereo_samples = np.column_stack((samples, samples))
                
                sound = pygame.sndarray.make_sound(stereo_samples)
                sound.play()
                
                import time
                time.sleep(0.6)
                
                logger.info("✅ Audio test successful")
                return True
            
            else:
                logger.warning("Audio test not available for current backend")
                return True
                
        except Exception as e:
            logger.error(f"Audio test failed: {e}")
            return False

# Global instance
audio_alert_service = AudioAlertService()
