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
                
                # Post-process: Replace bending patterns with stroke warning
                vietnamese_text = self._replace_bending_patterns(vietnamese_text)
                
                return vietnamese_text, "ai_model"
            else:
                return self._rule_based_translation(english_text), "fallback"
                
        except Exception as e:
            logger.error(f"❌ AI translation failed: {e}")
            return self._rule_based_translation(english_text), "error_fallback"
    
    def _remove_gender_terms(self, vietnamese_text):
        """Remove gender-specific terms for privacy (replace with generic 'một người')"""
        import re
        
        # Replace gender-specific terms with neutral "một người"
        gender_replacements = [
            (r'\bmột người phụ nữ\b', 'một người'),
            (r'\bmột người đàn ông\b', 'một người'),
            (r'\bngười phụ nữ\b', 'người'),
            (r'\bngười đàn ông\b', 'người'),
            (r'\bphụ nữ\b', 'người'),
            (r'\bđàn ông\b', 'người'),
            (r'\bcô ấy\b', 'họ'),
            (r'\banh ấy\b', 'họ'),
        ]
        
        for pattern, replacement in gender_replacements:
            vietnamese_text = re.sub(pattern, replacement, vietnamese_text, flags=re.IGNORECASE)
        
        return vietnamese_text
    
    def _replace_bending_patterns(self, vietnamese_text):
        """Replace bending/leaning patterns with stroke warning"""
        import re
        
        # Patterns indicating bending forward (stroke symptom)
        bending_patterns = [
            r'cúi\s+(người|xuống|thấp)',
            r'chúi\s+người',
            r'cong\s+người',
            r'nghiêng\s+(người|về\s+phía\s+trước)',
            r'gập\s+người',
            r'đang\s+cúi',
            r'đang\s+chúi',
            r'đang\s+cong',
            r'đang\s+nghiêng',
            r'đang\s+gập'
        ]
        
        for pattern in bending_patterns:
            vietnamese_text = re.sub(pattern, 'có dấu hiệu đột quỵ', vietnamese_text, flags=re.IGNORECASE)
        
        return vietnamese_text
    
    def _rule_based_translation(self, english_text):
        """Rule-based translation as fallback"""
        translations = {
            # People - Dùng "bệnh nhân" cho hệ thống healthcare
            "two people": "hai người",        # 2 người trở lên
            "three people": "ba người",       # 3 người
            "several people": "nhiều người",  # Nhiều người
            "people": "những người",          # Nhiều người (fallback)
            
            "a woman": "bệnh nhân",   # Healthcare context - 1 người
            "a man": "bệnh nhân",      # Healthcare context - 1 người
            "a person": "bệnh nhân",
            "woman": "bệnh nhân",      # Healthcare context
            "man": "bệnh nhân",        # Healthcare context
            "person": "bệnh nhân", 
            
            "child": "đứa trẻ",
            
            # Actions - Fixed order
            "is bending": "có dấu hiệu đột quỵ",
            "is leaning forward": "có dấu hiệu đột quỵ",
            "is stooping": "có dấu hiệu đột quỵ",
            "bending over": "có dấu hiệu đột quỵ",
            "bending down": "có dấu hiệu đột quỵ",
            "leaning forward": "có dấu hiệu đột quỵ",
            "bending": "có dấu hiệu đột quỵ",
            "stooping": "có dấu hiệu đột quỵ",
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
        
        # Post-process: Replace bending patterns with stroke warning
        result = self._replace_bending_patterns(result)
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        
        return result
    
    def enhance_medical_context(self, base_caption, image_path, event_type=None, camera_name=None, confidence=None):
        """
        Add medical context based on event_type or detection results
        
        Args:
            base_caption: Base Vietnamese caption
            image_path: Path to image file
            event_type: Actual event type ('fall', 'seizure', 'abnormal_behavior', etc.)
            camera_name: Optional camera name to add location context
            confidence: Optional confidence score to determine if caption should be modified
        """
        filename = Path(image_path).name.lower()
        
        # 🔥 STEP 1: SMART Caption Replacement based on CONFIDENCE
        # LOGIC THÔNG MINH:
        # - Confidence CAO → Event THẬT → Thay đổi caption để phản ánh sự kiện
        # - Confidence THẤP → False positive → Giữ nguyên pose description từ BLIP
        #
        # THRESHOLD:
        # - Fall: confidence >= 0.60 → Té thật → Thay "ngồi/nằm" thành "ngã"
        # - Seizure: confidence >= 0.75 → Co giật thật → Thay "ngồi/nằm" thành "co giật"
        
        if confidence is not None and event_type:
            if event_type == 'fall' and confidence >= 0.60:
                # Confidence cao → Té THẬT → Thay đổi caption
                base_caption = base_caption.replace("đang ngồi", "đang ngã")
                base_caption = base_caption.replace("đang nằm", "đang nằm sau khi té")
                base_caption = base_caption.replace("đang quỳ", "đang ngã")
                base_caption = base_caption.replace("quỳ gối", "ngã")
                # Giữ "nằm" nếu đã có "sau khi té", không thay nữa
                if "sau khi té" not in base_caption:
                    base_caption = base_caption.replace("nằm trên", "ngã trên")
                    
            elif event_type in ['seizure', 'abnormal_behavior'] and confidence >= 0.75:
                # Confidence cao → Co giật THẬT → Thay đổi caption
                base_caption = base_caption.replace("đang ngồi", "đang co giật")
                base_caption = base_caption.replace("đang nằm", "đang co giật")
                base_caption = base_caption.replace("đang quỳ", "đang co giật")
                base_caption = base_caption.replace("quỳ gối", "co giật")
                base_caption = base_caption.replace("ngồi trên", "co giật trên")
                base_caption = base_caption.replace("nằm trên", "co giật trên")
            # else: Confidence thấp → GIỮ NGUYÊN caption gốc từ BLIP
        
        # STEP 2: Add or replace location with camera name
        if camera_name:
            # Replace existing location phrases
            base_caption = base_caption.replace("trong phòng của mình", f"trong {camera_name}")
            base_caption = base_caption.replace("in a room", f"trong {camera_name}")
            base_caption = base_caption.replace("in the room", f"trong {camera_name}")
            base_caption = base_caption.replace("trong một căn phòng", f"trong {camera_name}")
            
            # If no location found, append camera name at the end
            location_keywords = ["trong phòng", "trong một căn", "trong {", "in a room", "in the room"]
            has_location = any(keyword in base_caption for keyword in location_keywords)
            
            if not has_location:
                # Add location before the last punctuation or at the end
                if base_caption.endswith("."):
                    base_caption = base_caption[:-1] + f" trong {camera_name}."
                else:
                    base_caption = base_caption + f" trong {camera_name}"
        
        medical_additions = []
        
        # STEP 3: Detect emergency type - prioritize event_type parameter over filename
        if event_type:
            # Use actual event_type (most reliable source)
            if event_type == 'fall':
                medical_additions.append("⚠️ Cảnh báo: Phát hiện ngã đổ")
            elif event_type in ['seizure', 'abnormal_behavior']:
                medical_additions.append("🚨 Cảnh báo: Phát hiện co giật")
        else:
            # Fallback to filename detection (less reliable)
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
    
    def generate_professional_caption(self, image_path, event_type=None, camera_name=None, confidence=None):
        """
        Generate professional Vietnamese caption
        
        Args:
            image_path: Path to image file
            event_type: Optional event type for accurate medical context ('fall', 'seizure', etc.)
            camera_name: Optional camera name to include in location context
            confidence: Optional confidence score to determine if caption should be modified
        """
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
                
                # Step 2.5: Remove gender-specific terms (privacy protection)
                vietnamese_caption = self._remove_gender_terms(vietnamese_caption)
                
                # Step 3: Enhance with medical context (pass event_type + camera_name + confidence)
                final_caption = self.enhance_medical_context(vietnamese_caption, image_path, event_type, camera_name, confidence)
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
