"""
Test Audio với Bluetooth Speaker - Windows winsound
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import winsound
from pathlib import Path

print("=" * 80)
print("🔊 BLUETOOTH SPEAKER TEST - winsound")
print("=" * 80)

sound_path = Path(__file__).parent.parent / 'src' / 'sounds' / 'emergency_siren.mp3'

print(f"\nSound file: {sound_path}")
print(f"Exists: {sound_path.exists()}")

# Test 1: System beep qua Bluetooth
print("\n" + "=" * 80)
print("Test 1: System Beep (should play on Bluetooth)")
print("=" * 80)
print("\n🔔 Beeping 3 times through default audio device...")
print("   (Should play on your Bluetooth speaker)")

for i in range(3):
    print(f"\n   Beep {i+1}/3 - Frequency: {500 + i*500}Hz")
    winsound.Beep(500 + i*500, 800)  # Different frequencies
    import time
    time.sleep(0.3)

print("\n✅ Beep test complete!")

heard_beep = input("\n👂 Did you hear the beeps on Bluetooth speaker? (y/n): ").strip().lower()

if heard_beep != 'y':
    print("\n❌ Bluetooth speaker not working!")
    print("\n💡 TROUBLESHOOTING:")
    print("   1. Check Bluetooth speaker is connected and paired")
    print("   2. Open Windows Sound settings:")
    print("      - Right-click speaker icon → Sound settings")
    print("      - Choose your Bluetooth speaker as Output device")
    print("   3. Test with music/YouTube to confirm Bluetooth works")
    print("   4. Volume on Bluetooth speaker turned up")
    exit(1)

print("\n✅ Great! Bluetooth speaker works with beeps")

# Test 2: WAV file (winsound only supports WAV, not MP3)
print("\n" + "=" * 80)
print("Test 2: WAV File Playback")
print("=" * 80)

# Check if we have WAV file, if not convert MP3 to WAV
wav_path = Path(__file__).parent.parent / 'src' / 'sounds' / 'emergency_siren.wav'

if not wav_path.exists():
    print(f"\n⚠️ WAV file not found: {wav_path.name}")
    print("   Converting MP3 to WAV...")
    
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_mp3(str(sound_path))
        audio.export(str(wav_path), format='wav')
        
        print(f"   ✅ Converted to WAV: {wav_path.name}")
        
    except ImportError:
        print("   ❌ pydub not installed")
        print("   Install: pip install pydub")
        print("\n💡 ALTERNATIVE: Use online converter to create .wav file")
        print(f"   1. Convert {sound_path.name} to .wav online")
        print(f"   2. Save as: {wav_path}")
        exit(1)
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        exit(1)

if wav_path.exists():
    print(f"\n🔊 Playing WAV file on Bluetooth speaker...")
    print("   Listen carefully!")
    
    try:
        # SND_FILENAME: Play file
        # SND_ASYNC: Don't wait for completion
        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        
        print(f"   ✅ Playing... (3 seconds)")
        time.sleep(3)
        
        # Stop
        winsound.PlaySound(None, winsound.SND_PURGE)
        
        print(f"   ✅ Playback complete!")
        
    except Exception as e:
        print(f"   ❌ winsound error: {e}")

print("\n" + "=" * 80)
print("📊 RESULT")
print("=" * 80)

heard_wav = input("\n👂 Did you hear the alarm sound on Bluetooth? (y/n): ").strip().lower()

if heard_wav == 'y':
    print("\n✅ SUCCESS! winsound works with Bluetooth!")
    print("\n💡 SOLUTION: Modify audio_alert_service.py")
    print("   Replace pygame with winsound.PlaySound()")
    print("   Or ensure pygame outputs to default device")
else:
    print("\n❌ Still no sound...")
    print("\n💡 FINAL CHECKS:")
    print("   1. Bluetooth speaker volume")
    print("   2. Windows Sound settings → Output device")
    print("   3. Try playing music/video to confirm Bluetooth works")
    print("   4. Reconnect Bluetooth speaker")

print("\n" + "=" * 80)
