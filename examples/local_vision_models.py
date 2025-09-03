"""
Sử dụng các local vision models khác để generate descriptions
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class LocalVisionModels:
    """Sử dụng local pre-trained vision models"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
    
    def load_blip_model(self):
        """Load BLIP model cho image captioning"""
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            self.processors['blip'] = processor
            self.models['blip'] = model
            
            logger.info("✅ BLIP model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP: {e}")
            return False
    
    def load_vit_gpt2_model(self):
        """Load ViT-GPT2 model"""
        try:
            model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            
            self.models['vit_gpt2'] = model
            self.processors['vit_gpt2'] = {
                'feature_extractor': feature_extractor,
                'tokenizer': tokenizer
            }
            
            logger.info("✅ ViT-GPT2 model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load ViT-GPT2: {e}")
            return False
    
    def generate_caption_blip(self, image_path):
        """Generate caption using BLIP"""
        try:
            if 'blip' not in self.models:
                if not self.load_blip_model():
                    return None
            
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processors['blip'](image, return_tensors="pt")
            
            # Generate caption
            out = self.models['blip'].generate(**inputs, max_length=50)
            caption = self.processors['blip'].decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"BLIP caption error: {e}")
            return None
    
    def generate_caption_vit_gpt2(self, image_path):
        """Generate caption using ViT-GPT2"""
        try:
            if 'vit_gpt2' not in self.models:
                if not self.load_vit_gpt2_model():
                    return None
            
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            pixel_values = self.processors['vit_gpt2']['feature_extractor'](
                image, return_tensors="pt"
            ).pixel_values
            
            # Generate caption
            output_ids = self.models['vit_gpt2'].generate(
                pixel_values, max_length=50, num_beams=4
            )
            
            caption = self.processors['vit_gpt2']['tokenizer'].decode(
                output_ids[0], skip_special_tokens=True
            )
            
            return caption
            
        except Exception as e:
            logger.error(f"ViT-GPT2 caption error: {e}")
            return None
    
    def translate_to_vietnamese(self, english_caption):
        """Translate English caption to Vietnamese"""
        try:
            # Có thể sử dụng translation model
            from transformers import pipeline
            
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")
            result = translator(english_caption)
            
            return result[0]['translation_text']
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fallback simple mapping
            translations = {
                "person": "người",
                "lying": "nằm",
                "floor": "sàn nhà",
                "medical": "y tế",
                "emergency": "khẩn cấp",
                "danger": "nguy hiểm"
            }
            
            vietnamese_caption = english_caption
            for eng, vie in translations.items():
                vietnamese_caption = vietnamese_caption.replace(eng, vie)
            
            return vietnamese_caption
    
    def generate_vietnamese_caption(self, image_path, model_type='blip'):
        """Generate Vietnamese caption"""
        try:
            # Generate English caption
            if model_type == 'blip':
                english_caption = self.generate_caption_blip(image_path)
            elif model_type == 'vit_gpt2':
                english_caption = self.generate_caption_vit_gpt2(image_path)
            else:
                return None
            
            if english_caption:
                # Translate to Vietnamese
                vietnamese_caption = self.translate_to_vietnamese(english_caption)
                return vietnamese_caption
            
            return None
            
        except Exception as e:
            logger.error(f"Vietnamese caption generation error: {e}")
            return None

def demo_local_models():
    """Demo local vision models"""
    models = LocalVisionModels()
    
    # Test với ảnh mẫu
    test_image = "path/to/test/image.jpg"
    
    print("=== Testing Local Vision Models ===")
    
    # Test BLIP
    blip_caption = models.generate_vietnamese_caption(test_image, 'blip')
    print(f"BLIP Caption: {blip_caption}")
    
    # Test ViT-GPT2
    vit_caption = models.generate_vietnamese_caption(test_image, 'vit_gpt2')
    print(f"ViT-GPT2 Caption: {vit_caption}")

if __name__ == '__main__':
    demo_local_models()
