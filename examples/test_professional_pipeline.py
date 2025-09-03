"""
Test Professional Vietnamese Caption Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from service.professional_vietnamese_caption import get_professional_caption_pipeline
import glob
from pathlib import Path

def test_professional_pipeline():
    """Test professional pipeline với ảnh thật"""
    print("🇻🇳 TESTING PROFESSIONAL VIETNAMESE CAPTION PIPELINE")
    print("=" * 65)
    
    # Get pipeline
    pipeline = get_professional_caption_pipeline()
    
    # Find alerts folder
    alerts_folder = Path(__file__).parent.parent / 'src' / 'examples' / 'data' / 'saved_frames' / 'alerts'
    
    if not alerts_folder.exists():
        print(f"❌ Alerts folder not found: {alerts_folder}")
        return
    
    # Get images
    image_files = list(alerts_folder.glob("*.jpg"))
    if not image_files:
        print("❌ No images found")
        return
    
    # Sort by creation time (latest first)
    latest_images = sorted(image_files, key=lambda p: p.stat().st_ctime, reverse=True)[:3]
    
    print(f"📁 Found {len(image_files)} images, testing latest 3")
    print(f"🤖 BLIP Loaded: {pipeline.blip_loaded}")
    print(f"🔄 Translator Loaded: {pipeline.translator_loaded}")
    
    print(f"\n🖼️ Professional Caption Results:")
    print("-" * 65)
    
    for i, image_path in enumerate(latest_images, 1):
        try:
            print(f"\n{i}. 📸 {image_path.name}")
            
            # Generate professional caption
            caption, metadata = pipeline.generate_professional_caption(str(image_path))
            
            print(f"   🇻🇳 Final Caption: {caption}")
            print(f"   🔧 Pipeline: {' → '.join(metadata.get('pipeline_steps', []))}")
            print(f"   ✅ Success: {metadata.get('success', False)}")
            
            if 'english_caption' in metadata:
                print(f"   🌍 English: {metadata['english_caption']}")
            
            if 'vietnamese_base' in metadata:
                print(f"   📝 Vietnamese Base: {metadata['vietnamese_base']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 65)
    print("🎯 Professional Pipeline Summary:")
    print(f"   • BLIP Vision: {'✅ Active' if pipeline.blip_loaded else '❌ Not Available'}")
    print(f"   • AI Translation: {'✅ Active' if pipeline.translator_loaded else '📝 Rule-based Fallback'}")
    print(f"   • Medical Context: ✅ Enhanced")
    print("=" * 65)

if __name__ == '__main__':
    test_professional_pipeline()
