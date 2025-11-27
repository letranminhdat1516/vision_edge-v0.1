"""
Test Bluetooth Speaker với winsound backend (sau khi cài ffmpeg)
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.services.audio_alert_service import AudioAlertService
import asyncio

async def main():
    print("=" * 80)
    print("🔊 BLUETOOTH SPEAKER TEST (winsound + ffmpeg)")
    print("=" * 80)
    
    # Initialize audio service
    audio_service = AudioAlertService()
    
    # Check status
    status = audio_service.get_status()
    backend = audio_service.audio_backend if hasattr(audio_service, 'audio_backend') else 'unknown'
    print(f"\n📊 Audio Backend: {backend}")
    print(f"   Sound file: {'✅ Found' if status.get('sound_file_exists') else '❌ Not found'}")
    
    if backend == 'winsound':
        print("\n✅ winsound backend active - Will use Bluetooth speaker!")
        print("   Make sure your Bluetooth speaker is:")
        print("   1. Connected to Windows")
        print("   2. Set as default audio output device")
    else:
        print(f"\n⚠️ Using {backend} backend (not winsound)")
        print("   Check if ffmpeg is installed and restart terminal")
    
    print("\n" + "=" * 80)
    print("🎵 TESTING BLUETOOTH PLAYBACK")
    print("=" * 80)
    
    input("\n👂 Ready? Press ENTER to play 5-second alarm through Bluetooth...")
    
    # Play alarm
    result = await audio_service.play_emergency_alarm(user_id="test_bluetooth", duration=5)
    
    if result['success']:
        print(f"\n🔊 ALARM PLAYING for {result['duration']} seconds...")
        print("   Listen on your Bluetooth speaker! Should be LOUD 🔊")
        
        # Wait for duration
        await asyncio.sleep(result['duration'])
        
        # Stop alarm
        await audio_service.stop_alarm()
        
        print("\n✅ Test complete!")
    else:
        print(f"\n❌ Failed: {result['message']}")
    
    print("\n" + "=" * 80)
    response = input("👂 Did you hear LOUD alarm on Bluetooth speaker? (y/n): ")
    
    if response.lower() == 'y':
        print("\n🎉 SUCCESS! Bluetooth speaker working with winsound!")
        print("   Your alarm system will now use Bluetooth for loud alerts!")
    else:
        print("\n⚠️ Troubleshooting:")
        print("   1. Check Windows Sound Settings → Output Device = Bluetooth")
        print("   2. Restart terminal after installing ffmpeg")
        print("   3. Verify ffmpeg installed: ffmpeg -version")
        print("   4. Try playing emergency_siren.mp3 in Windows Media Player")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
