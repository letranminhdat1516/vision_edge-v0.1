"""
Test Audio with System Beep + Pygame
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 80)
print("🔊 AUDIO DEVICE TEST")
print("=" * 80)

# Test 1: System beep
print("\n🔔 Test 1: System Beep")
print("   You should hear a beep sound...")
try:
    import winsound
    winsound.Beep(1000, 500)  # 1000Hz for 500ms
    print("   ✅ System beep sent!")
except Exception as e:
    print(f"   ❌ System beep failed: {e}")

input("\n👂 Did you hear the beep? Press Enter to continue...")

# Test 2: Pygame with different sound
print("\n🎵 Test 2: Pygame Audio")
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    
    print(f"   Pygame initialized!")
    print(f"   Mixer: {pygame.mixer.get_init()}")
    
    # Try loading our sound file
    from pathlib import Path
    sound_path = Path(__file__).parent.parent / 'src' / 'sounds' / 'emergency_siren.mp3'
    
    print(f"\n   Sound file: {sound_path}")
    print(f"   Exists: {sound_path.exists()}")
    
    if sound_path.exists():
        print(f"\n🔊 Playing emergency_siren.mp3...")
        sound = pygame.mixer.Sound(str(sound_path))
        sound.set_volume(1.0)  # Max volume
        
        print(f"   Sound length: {sound.get_length():.2f}s")
        print(f"   Volume: 100%")
        print(f"\n   Playing NOW! Listen carefully... 👂")
        
        sound.play()
        
        import time
        time.sleep(3)
        
        pygame.mixer.stop()
        print(f"\n   ✅ Playback finished!")
    else:
        print(f"\n   ❌ Sound file not found!")
        print(f"   Create the file or use a different sound")
        
except Exception as e:
    print(f"\n   ❌ Pygame test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check Windows audio settings
print("\n" + "=" * 80)
print("💡 TROUBLESHOOTING TIPS:")
print("=" * 80)
print("\n1. Check Windows Volume:")
print("   - Right-click speaker icon in taskbar")
print("   - Check 'Volume Mixer'")
print("   - Make sure Python is not muted")

print("\n2. Check Default Audio Device:")
print("   - Settings > System > Sound")
print("   - Make sure correct output device is selected")

print("\n3. Test with Windows Media Player:")
print(f"   - Try playing: {sound_path}")
print("   - If WMP works but Python doesn't = pygame issue")

print("\n4. Update Audio Driver:")
print("   - Device Manager > Sound controllers")
print("   - Update audio driver")

print("\n" + "=" * 80)
