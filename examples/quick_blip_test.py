"""
Quick implementation: BLIP model for real image captioning
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

def test_blip_real():
    """Test BLIP model with real image"""
    try:
        # Load BLIP model
        print("📥 Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Test với ảnh mới nhất
        alerts_folder = "src/examples/data/saved_frames/alerts"
        import glob
        image_files = glob.glob(os.path.join(alerts_folder, "*.jpg"))
        
        if image_files:
            latest_image = max(image_files, key=os.path.getctime)
            print(f"Testing with: {os.path.basename(latest_image)}")
            
            # Load image
            image = Image.open(latest_image).convert('RGB')
            
            # Generate caption
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"🌍 English caption: {caption}")
            
            # Simple translation
            vietnamese_caption = translate_to_vietnamese(caption)
            print(f"🇻🇳 Vietnamese caption: {vietnamese_caption}")
            
        else:
            print("❌ No images found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install requirements: pip install transformers Pillow")

def translate_to_vietnamese(english_text):
    """Simple translation mapping"""
    translations = {
        "a person": "một người",
        "person": "người",
        "lying": "đang nằm",
        "on the": "trên",
        "floor": "sàn nhà",
        "bed": "giường",
        "room": "phòng",
        "white": "trắng",
        "wearing": "mặc",
        "shirt": "áo"
    }
    
    result = english_text.lower()
    for eng, vie in translations.items():
        result = result.replace(eng, vie)
    
    return result.capitalize()

if __name__ == '__main__':
    test_blip_real()
