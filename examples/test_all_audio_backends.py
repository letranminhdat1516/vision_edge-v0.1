"""
Test với nhiều audio backends khác nhau
"""

import os
from pathlib import Path

sound_path = Path(__file__).parent.parent / 'src' / 'sounds' / 'emergency_siren.mp3'

print("=" * 80)
print("🔊 TESTING MULTIPLE AUDIO BACKENDS")
print("=" * 80)
print(f"\nSound file: {sound_path}")
print(f"Exists: {sound_path.exists()}")
print(f"Size: {sound_path.stat().st_size / 1024:.1f} KB")

if not sound_path.exists():
    print("\n❌ Sound file not found!")
    exit(1)

# Test 1: playsound (simplest)
print("\n" + "=" * 80)
print("Test 1: playsound library")
print("=" * 80)
try:
    from playsound import playsound
    print("🔊 Playing with playsound... (3 seconds)")
    import threading
    
    def play_sound():
        playsound(str(sound_path))
    
    thread = threading.Thread(target=play_sound)
    thread.start()
    
    import time
    time.sleep(3)
    
    print("✅ playsound test complete")
except ImportError:
    print("⚠️ playsound not installed")
    print("   Install: pip install playsound")
except Exception as e:
    print(f"❌ playsound error: {e}")

input("\n👂 Did you hear sound from playsound? Press Enter...")

# Test 2: pygame
print("\n" + "=" * 80)
print("Test 2: pygame library")
print("=" * 80)
try:
    import pygame
    pygame.mixer.quit()  # Reset mixer
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    
    print(f"Mixer initialized: {pygame.mixer.get_init()}")
    
    sound = pygame.mixer.Sound(str(sound_path))
    print(f"Sound loaded: {sound.get_length():.2f}s")
    
    sound.set_volume(1.0)
    print(f"Volume: 100%")
    
    print("🔊 Playing with pygame...")
    channel = sound.play()
    
    time.sleep(3)
    pygame.mixer.stop()
    
    print("✅ pygame test complete")
except Exception as e:
    print(f"❌ pygame error: {e}")
    import traceback
    traceback.print_exc()

input("\n👂 Did you hear sound from pygame? Press Enter...")

# Test 3: pydub + simpleaudio
print("\n" + "=" * 80)
print("Test 3: pydub + simpleaudio")
print("=" * 80)
try:
    from pydub import AudioSegment
    from pydub.playback import play
    
    print("Loading audio file...")
    audio = AudioSegment.from_mp3(str(sound_path))
    
    print(f"Duration: {len(audio)/1000:.2f}s")
    print(f"Channels: {audio.channels}")
    print(f"Sample rate: {audio.frame_rate}")
    
    print("🔊 Playing with pydub (3s)...")
    
    # Play first 3 seconds
    play(audio[:3000])
    
    print("✅ pydub test complete")
except ImportError:
    print("⚠️ pydub/simpleaudio not installed")
    print("   Install: pip install pydub simpleaudio")
except Exception as e:
    print(f"❌ pydub error: {e}")

input("\n👂 Did you hear sound from pydub? Press Enter...")

# Test 4: winsound (Windows only - beep)
print("\n" + "=" * 80)
print("Test 4: winsound (Windows beeps)")
print("=" * 80)
try:
    import winsound
    
    print("🔔 Beeping 3 times...")
    for i in range(3):
        print(f"   Beep {i+1}...")
        winsound.Beep(1000, 300)  # 1000Hz, 300ms
        time.sleep(0.2)
    
    print("✅ winsound test complete")
except Exception as e:
    print(f"❌ winsound error: {e}")

input("\n👂 Did you hear the beeps? Press Enter...")

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print("\nIf you heard:")
print("✅ Beeps but NOT music → Audio output works, pygame/file issue")
print("✅ Windows Media Player → File is valid, Python audio issue")
print("❌ Nothing at all → Check:")
print("   1. Volume not muted")
print("   2. Correct output device (speakers/headphones)")
print("   3. Audio drivers installed")

print("\n💡 RECOMMENDATION:")
print("If pygame doesn't work, try playsound:")
print("   pip install playsound")
print("   # Then modify audio_alert_service.py to use playsound")

print("\n" + "=" * 80)
