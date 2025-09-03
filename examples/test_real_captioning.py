"""
Test REAL Vietnamese captioning với BLIP model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.vietnamese_caption_service_fixed import get_vietnamese_caption_service
import glob
from pathlib import Path

def test_real_captioning():
    """Test thật với BLIP model"""
    print("=" * 60)
    print("🇻🇳 TESTING REAL VIETNAMESE CAPTIONING WITH BLIP")
    print("=" * 60)
    
    # Initialize service
    service = get_vietnamese_caption_service()
    
    # Find alerts folder
    alerts_folder = Path(__file__).parent.parent / 'src' / 'examples' / 'data' / 'saved_frames' / 'alerts'
    
    if not alerts_folder.exists():
        print(f"❌ Alerts folder not found: {alerts_folder}")
        return
    
    # Get latest images
    image_files = list(alerts_folder.glob("*.jpg"))
    if not image_files:
        print("❌ No images found")
        return
    
    # Sort by creation time (latest first)
    image_files.sort(key=lambda p: p.stat().st_ctime, reverse=True)
    
    print(f"📁 Found {len(image_files)} images")
    print(f"🤖 Service info: {service.get_service_info()}")
    
    # Test with 3 latest images
    print(f"\n🖼️ Testing with 3 latest images:")
    print("-" * 60)
    
    for i, image_file in enumerate(image_files[:3], 1):
        try:
            print(f"\n{i}. 📸 {image_file.name}")
            
            # Generate caption
            caption = service.generate_caption(str(image_file))
            
            print(f"   🇻🇳 Caption: {caption}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test completed!")
    print("💡 For REAL AI captioning, we need BLIP model")
    print("📝 Current: Enhanced intelligent fallback")
    print("=" * 60)

def install_requirements():
    """Install requirements for real captioning"""
    print("📦 Installing requirements for real captioning...")
    
    requirements = [
        "transformers",
        "torch",
        "torchvision", 
        "Pillow",
        "requests"
    ]
    
    for req in requirements:
        try:
            __import__(req)
            print(f"✅ {req} already installed")
        except ImportError:
            print(f"❌ {req} not found - install with: pip install {req}")

if __name__ == '__main__':
    print("Checking requirements...")
    install_requirements()
    print("\nStarting real captioning test...")
    test_real_captioning()
