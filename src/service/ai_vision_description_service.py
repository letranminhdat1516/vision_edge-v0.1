"""
Professional Vietnamese Caption Pipeline
BLIP → Translation Model → High Quality Vietnamese Caption
"""

import torch
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProfessionalVietnameseCaptionPipeline:
    """Pipeline BLIP + Translation Model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vision model (BLIP)
        self.blip_model = None
        self.blip_processor = None
        self.blip_loaded = False
        
        # Translation model 
        self.translator = None
        self.translator_loaded = False
        
        # Load models
        self._load_vision_model()
        self._load_translation_model()
    
    def _load_vision_model(self):
        """Load BLIP vision model"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            logger.info("📥 Loading BLIP vision model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            self.blip_loaded = True
            logger.info("✅ BLIP vision model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load BLIP: {e}")
            self.blip_loaded = False
    
    def _load_translation_model(self):
        """Load English → Vietnamese translation model"""
        try:
            from transformers import pipeline
            
            logger.info("📥 Loading translation model...")
            
            # Try better models in order of preference
            models_to_try = [
                "facebook/nllb-200-distilled-600M",  # NLLB model - better quality
                "Helsinki-NLP/opus-mt-en-vi",         # Fallback
            ]
            
            for model_name in models_to_try:
                try:
                    if "nllb" in model_name:
                        # NLLB model requires specific parameters
                        self.translator = pipeline(
                            "translation",
                            model=model_name,
                            src_lang="eng_Latn",
                            tgt_lang="vie_Latn",
                            device=0 if torch.cuda.is_available() else -1
                        )
                    else:
                        # Standard translation pipeline
                        self.translator = pipeline(
                            "translation", 
                            model=model_name,
                            device=0 if torch.cuda.is_available() else -1
                        )
                    
                    self.translator_loaded = True
                    logger.info(f"✅ EN→VI translation model loaded: {model_name}")
                    return
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ Failed to load {model_name}: {model_error}")
                    continue
            
            # If all models fail, try alternative
            raise Exception("All primary translation models failed")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load translation model: {e}")
            logger.info("💡 Trying alternative translation methods...")
            self._load_alternative_translator()
    
    def _load_alternative_translator(self):
        """Load alternative translation methods"""
        try:
            # Option 2: VinAI PhoBERT-based translation
            from transformers import pipeline
            
            self.translator = pipeline(
                "translation",
                model="VietAI/envit5-translation", 
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.translator_loaded = True
            logger.info("✅ VietAI translation model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Alternative translation failed: {e}")
            logger.info("📝 Will use rule-based translation")
            self.translator_loaded = False
    
    def generate_english_caption(self, image_path):
        """Generate English caption using BLIP"""
        try:
            if not self.blip_loaded:
                return None, "BLIP model not available"
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process with BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                output = self.blip_model.generate(**inputs, max_length=50)
                english_caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            
            return english_caption, "success"
            
        except Exception as e:
            logger.error(f"❌ BLIP generation failed: {e}")
            return None, str(e)
    
    def translate_to_vietnamese(self, english_text):
        """Translate English to Vietnamese using AI model"""
        try:
            if not self.translator_loaded:
                return self._rule_based_translation(english_text), "rule_based"
            
            # Use AI translation model
            if hasattr(self.translator, 'model') and "nllb" in str(self.translator.model.name_or_path):
                # NLLB model
                result = self.translator(english_text, src_lang="eng_Latn", tgt_lang="vie_Latn")
            else:
                # Standard translation model
                result = self.translator(english_text)
            
            if isinstance(result, list) and len(result) > 0:
                vietnamese_text = result[0]['translation_text']
                
                # Clean up any weird symbols from translation
                vietnamese_text = vietnamese_text.replace('♪', '').replace('_', '').strip()
                
                return vietnamese_text, "ai_model"
            else:
                return self._rule_based_translation(english_text), "fallback"
                
        except Exception as e:
            logger.error(f"❌ AI translation failed: {e}")
            return self._rule_based_translation(english_text), "error_fallback"
    
    def _rule_based_translation(self, english_text):
        """Rule-based translation as fallback"""
        translations = {
            # People - Fixed order for better matching
            "a woman": "một phụ nữ",
            "a man": "một người đàn ông", 
            "a person": "một người",
            "woman": "phụ nữ",
            "man": "người đàn ông",
            "person": "người", 
            "people": "những người",
            "child": "đứa trẻ",
            
            # Actions - Fixed order
            "is walking": "đang đi",
            "is standing": "đang đứng",
            "is sitting": "đang ngồi",
            "is lying": "đang nằm",
            "is sleeping": "đang ngủ",
            "is holding": "đang cầm",
            "is wearing": "đang mặc",
            "walking": "đi bộ",
            "standing": "đứng",
            "sitting": "ngồi", 
            "lying": "nằm",
            "sleeping": "ngủ",
            "holding": "cầm",
            "wearing": "mặc",
            
            # Movement directions
            "walking down": "đi xuống",
            "walking up": "đi lên", 
            "down": "xuống",
            "up": "lên",
            
            # Places - More specific first
            "in a hotel": "trong khách sạn",
            "in the hallway": "trong hành lang",
            "in a room": "trong phòng",
            "on the floor": "trên sàn nhà",
            "on the bed": "trên giường",
            "in the hospital": "trong bệnh viện",
            "at home": "ở nhà",
            "hallway": "hành lang",
            "hotel": "khách sạn",
            
            # Objects
            "bed": "giường",
            "chair": "ghế", 
            "table": "bàn",
            "window": "cửa sổ",
            "door": "cửa",
            "phone": "điện thoại",
            
            # Clothing
            "dress": "váy",
            "shirt": "áo", 
            "pants": "quần",
            "jacket": "áo khoác",
            
            # Colors
            "white": "màu trắng",
            "black": "màu đen",
            "red": "màu đỏ", 
            "blue": "màu xanh",
            "green": "màu xanh lá",
            "pink": "màu hồng",
            "yellow": "màu vàng",
            
            # Common words - Most specific first
            " the ": " ",
            " a ": " một ",
            " an ": " một ",
            " is ": " là ",
            " and ": " và ",
            " in ": " trong ",
            " on ": " trên ",
            " at ": " tại "
        }
        
        result = english_text.lower()
        
        # Apply translations in order (most specific first)
        for en, vi in translations.items():
            result = result.replace(en, vi)
        
        # Clean up multiple spaces
        import re
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        
        return result
    
    def enhance_medical_context(self, base_caption, image_path):
        """Add medical context based on detection results"""
        filename = Path(image_path).name.lower()
        
        medical_additions = []
        
        # Detect emergency type
        if 'fall' in filename:
            medical_additions.append("⚠️ Cảnh báo: Phát hiện ngã đổ")
        elif 'seizure' in filename:
            medical_additions.append("🚨 Cảnh báo: Phát hiện co giật")
        
        # Extract confidence (REMOVED - to avoid duplicate confidence display)
        # if 'conf_' in filename:
        #     try:
        #         conf_part = filename.split('conf_')[1].split('.')[0]
        #         confidence = float(conf_part)
        #         medical_additions.append(f"- Độ tin cậy: {confidence:.1%}")
        #     except:
        #         pass
        
        # Combine base caption with medical context
        if medical_additions:
            enhanced = f"{base_caption}. {' - '.join(medical_additions)}"
            return enhanced
        
        return base_caption
    
    def generate_professional_caption(self, image_path):
        """Generate professional Vietnamese caption"""
        metadata = {
            "pipeline_steps": [],
            "success": False,
            "image_path": image_path
        }
        
        try:
            # Step 1: Generate English caption with BLIP
            english_caption, blip_status = self.generate_english_caption(image_path)
            metadata["pipeline_steps"].append(f"BLIP: {blip_status}")
            
            if english_caption:
                metadata["english_caption"] = english_caption
                
                # Step 2: Translate to Vietnamese
                vietnamese_caption, translation_method = self.translate_to_vietnamese(english_caption)
                metadata["pipeline_steps"].append(f"Translation: {translation_method}")
                metadata["vietnamese_base"] = vietnamese_caption
                
                # Step 3: Enhance with medical context
                final_caption = self.enhance_medical_context(vietnamese_caption, image_path)
                metadata["final_caption"] = final_caption
                metadata["success"] = True
                
                logger.info(f"🎯 Professional Pipeline: {english_caption} → {final_caption}")
                
                return final_caption, metadata
            
            else:
                # Fallback when BLIP fails
                fallback = self._generate_fallback_caption(image_path)
                metadata["pipeline_steps"].append("Fallback: filename_based")
                
                return fallback, metadata
                
        except Exception as e:
            logger.error(f"❌ Professional pipeline failed: {e}")
            metadata["error"] = str(e)
            
            fallback = "Không thể tạo mô tả cho ảnh này"
            return fallback, metadata
    
    def _generate_fallback_caption(self, image_path):
        """Generate fallback caption when AI fails"""
        filename = Path(image_path).name.lower()
        
        if 'fall' in filename:
            return "Phát hiện tình huống ngã đổ, cần kiểm tra an toàn"
        elif 'seizure' in filename:
            return "Phát hiện co giật, cần hỗ trợ y tế khẩn cấp"
        else:
            return "Tình huống y tế cần theo dõi"

# Global pipeline instance
_professional_pipeline = None

def get_professional_caption_pipeline():
    """Get singleton professional caption pipeline"""
    global _professional_pipeline
    if _professional_pipeline is None:
        _professional_pipeline = ProfessionalVietnameseCaptionPipeline()
    return _professional_pipeline

def generate_professional_vietnamese_caption(image_path):
    """Generate professional Vietnamese caption"""
    pipeline = get_professional_caption_pipeline()
    caption, metadata = pipeline.generate_professional_caption(image_path)
    return caption

if __name__ == '__main__':
    # Test professional pipeline
    import glob
    import os
    
    print("🇻🇳 TESTING PROFESSIONAL VIETNAMESE CAPTION PIPELINE")
    print("=" * 60)
    
    pipeline = get_professional_caption_pipeline()
    
    # Test với ảnh alerts
    alerts_folder = "../src/examples/data/saved_frames/alerts"
    
    if os.path.exists(alerts_folder):
        image_files = glob.glob(os.path.join(alerts_folder, "*.jpg"))
        
        if image_files:
            # Test 3 ảnh mới nhất
            latest_images = sorted(image_files, key=os.path.getctime, reverse=True)[:3]
            
            for i, image_path in enumerate(latest_images, 1):
                print(f"\n{i}. 📸 {os.path.basename(image_path)}")
                
                caption, metadata = pipeline.generate_professional_caption(image_path)
                
                print(f"   🇻🇳 Final Caption: {caption}")
                print(f"   🔧 Pipeline: {' → '.join(metadata['pipeline_steps'])}")
                
                if 'english_caption' in metadata:
                    print(f"   🌍 English: {metadata['english_caption']}")
                if 'vietnamese_base' in metadata:
                    print(f"   📝 Vietnamese Base: {metadata['vietnamese_base']}")
        else:
            print("❌ No images found")
    else:
        print(f"❌ Alerts folder not found: {alerts_folder}")
    
    print(f"\n✅ Professional pipeline test completed!")
    print(f"💡 Install models: pip install transformers torch")
