"""
Quick Audio Test - Test if alarm sound plays
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("🔊 QUICK AUDIO TEST")
print("=" * 80)

# Import audio service
from infrastructure.services.audio_alert_service import audio_alert_service

# Check status
status = audio_alert_service.get_status()

print("\n📊 Audio Service Status:")
print(f"   Enabled: {status['enabled']}")
print(f"   Backend: {status.get('audio_backend', 'N/A')}")
print(f"   Volume: {status.get('volume', 1.0) * 100:.0f}%")
print(f"   Sound file exists: {status.get('sound_file_exists', False)}")

if status['available_devices']:
    print(f"\n🔈 Audio Devices ({len(status['available_devices'])}):")
    for device in status['available_devices'][:5]:
        print(f"   - {device}")

if not status['enabled']:
    print("\n❌ Audio service is DISABLED!")
    print("   Check .env: AUDIO_ALERT_ENABLED=true")
    print("   Install pygame: pip install pygame")
    exit(1)

# Test sound playback
print("\n" + "=" * 80)
print("🎵 TESTING SOUND PLAYBACK")
print("=" * 80)

user_id = os.getenv('DEFAULT_USER_ID', 'test-user')

try:
    print("\n🔊 Playing alarm for 3 seconds...")
    print("   Listen carefully! 👂")
    
    # Play alarm
    result = asyncio.run(audio_alert_service.play_emergency_alarm(
        user_id=user_id,
        triggered_by='quick_test',
        duration=3  # Play for 3 seconds only
    ))
    
    if result['success']:
        print(f"\n✅ Alarm started!")
        print(f"   Volume: {result.get('volume', 1.0) * 100:.0f}%")
        print(f"   Duration: {result.get('duration', 0)}s")
        print(f"\n⏳ Playing... (wait {result.get('duration', 3)}s)")
        
        # Wait for alarm to finish
        time.sleep(result.get('duration', 3) + 0.5)
        
        print(f"\n✅ Test complete!")
        print(f"   Did you hear the alarm sound? 🔊")
        
    else:
        print(f"\n❌ Failed to play alarm!")
        print(f"   Error: {result['message']}")
        
        # Debug info
        print(f"\n🔍 Debug Info:")
        print(f"   Audio backend: {audio_alert_service.audio_backend}")
        print(f"   Sound directory: {audio_alert_service.sounds_dir}")
        print(f"   Check if emergency_siren.mp3 exists in sounds folder")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

# Ask for feedback
if status['enabled']:
    heard = input("\n👂 Did you hear the alarm sound? (y/n): ").strip().lower()
    
    if heard == 'y':
        print("\n✅ Audio system working correctly!")
    else:
        print("\n❌ Troubleshooting:")
        print("   1. Check your speaker/headphone connection")
        print("   2. Check system volume is not muted")
        print("   3. Try: pip install --upgrade pygame")
        print("   4. Check sound file exists:")
        print(f"      {audio_alert_service.sounds_dir / 'emergency_siren.mp3'}")

print("=" * 80)
