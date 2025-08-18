"""
Vietnamese Image Captioning Service for Healthcare Events
Generates Vietnamese descriptions for emergency images using pre-trained model
"""

import torch
import logging
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class VietnameseCaptionService:
    """Service to generate Vietnamese captions for healthcare emergency images"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Vietnamese Caption Service
        
        Args:
            model_path: Path to the Vietnamese captioning model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.transform = None
        self.is_initialized = False
        
        # Default model path
        if model_path is None:
            self.model_path = Path(__file__).parent.parent.parent / "models" / "Vietnamese-Image-Captioning" / "best_image_captioning_model_vietnamese.pth.tar"
        else:
            self.model_path = Path(model_path)
            
        # Initialize transform
        self._setup_transform()
        
        # Try to initialize model
        self._initialize_model()
    
    def _setup_transform(self):
        """Setup image transformation pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        logger.info("📸 Vietnamese caption image transform initialized")
    
    def _initialize_model(self):
        """Initialize the Vietnamese captioning model"""
        try:
            if not self.model_path.exists():
                logger.warning(f"❌ Vietnamese captioning model not found at: {self.model_path}")
                logger.info("📝 Will use fallback text descriptions instead")
                return
            
            # Import model classes with dependency check
            try:
                from .image_caption import ImageCaptioningModel, Vocabulary, TRANSFORMERS_AVAILABLE
                
                if not TRANSFORMERS_AVAILABLE:
                    logger.warning("⚠️ transformers library not available")
                    logger.info("📝 Using enhanced descriptive captions instead of AI model")
                    return
                    
            except ImportError as e:
                logger.warning(f"⚠️ Import error: {e}")
                logger.info("📝 Using enhanced descriptive captions instead")
                return
            
            model_name = "vinai/bartpho-syllable"
            
            # Initialize vocabulary
            logger.info("🔤 Loading Vietnamese vocabulary...")
            self.vocab = Vocabulary(model_name=model_name)
            
            # Initialize model
            logger.info("🧠 Loading Vietnamese Image Captioning model...")
            self.model = ImageCaptioningModel(
                embed_size=768, 
                bartpho_model_name=model_name,
                train_CNN=False, 
                freeze_bartpho=False
            ).to(self.device)
            
            # Load checkpoint
            logger.info(f"📦 Loading model checkpoint from: {self.model_path}")
            if self.model.load_checkpoint(self.model_path, self.device):
                self.model.eval()
                logger.info("🇻🇳 Vietnamese captioning model initialized successfully")
                self.is_initialized = True
            else:
                logger.warning("❌ Failed to load checkpoint, using fallback mode")
                self.model = None
                self.vocab = None
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vietnamese captioning model: {e}")
            logger.info("📝 Will use fallback text descriptions instead")
            self.model = None
            self.vocab = None
    
    def generate_caption(self, image_path: str, event_type: str = "unknown", confidence: float = 0.0) -> str:
        """
        Generate Vietnamese caption for emergency image
        
        Args:
            image_path: Path to the emergency image
            event_type: Type of event (fall, seizure, etc.)
            confidence: Detection confidence
            
        Returns:
            Vietnamese description of the emergency scene
        """
        try:
            # Check if image path is provided and exists
            if not image_path or not Path(image_path).exists():
                return self._generate_enhanced_description(event_type, confidence, image_path or "unknown")
                
            if not self.is_initialized:
                return self._generate_fallback_description(event_type, confidence)
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image_tensor = self.transform(image)
                if hasattr(image_tensor, 'unsqueeze'):
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                else:
                    return self._generate_fallback_description(event_type, confidence)
            else:
                return self._generate_fallback_description(event_type, confidence)
            
            # Generate caption using Vietnamese model
            with torch.no_grad():
                if self.model is not None and self.vocab is not None:
                    # Generate actual Vietnamese caption using the model
                    caption = self.model.predict(image_tensor, self.vocab, max_length=50)
                    enhanced_caption = self._enhance_caption_with_context(caption, event_type, confidence)
                    logger.info(f"🇻🇳 Generated Vietnamese caption from model: {enhanced_caption}")
                    return enhanced_caption
            
            # Fallback to context-aware description
            return self._generate_enhanced_description(event_type, confidence, image_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating Vietnamese caption: {e}")
            return self._generate_fallback_description(event_type, confidence)
    
    def _generate_enhanced_description(self, event_type: str, confidence: float, image_path: str) -> str:
        """Generate enhanced Vietnamese description with context"""
        
        # Base descriptions by event type
        base_descriptions = {
            "fall": [
                "Phát hiện người bệnh ngã xuống sàn trong phòng",
                "Cảnh báo: Người bệnh đã bị té ngã, cần hỗ trợ ngay",
                "Tình huống khẩn cấp: Phát hiện sự cố ngã của bệnh nhân"
            ],
            "abnormal_behavior": [
                "Phát hiện hành vi bất thường của người bệnh",
                "Cảnh báo: Người bệnh có biểu hiện co giật hoặc судороги",
                "Tình huống khẩn cấp: Phát hiện triệu chứng bất thường"
            ],
            "seizure": [
                "Phát hiện người bệnh có biểu hiện co giật",
                "Cảnh báo: Cơn động kinh hoặc co giật đang xảy ra",
                "Tình huống khẩn cấp: Phát hiện cơn судороги của bệnh nhân"
            ]
        }
        
        # Select description based on confidence
        descriptions = base_descriptions.get(event_type, base_descriptions["abnormal_behavior"])
        
        if confidence >= 0.8:
            desc_index = 2  # High confidence - emergency
        elif confidence >= 0.5:
            desc_index = 1  # Medium confidence - warning
        else:
            desc_index = 0  # Low confidence - normal
            
        base_desc = descriptions[desc_index]
        
        # Add confidence and timestamp info
        confidence_text = f"(Độ tin cậy: {confidence:.1%})"
        
        # Add location context if available
        if "room" in image_path.lower():
            location_text = "trong phòng bệnh"
        elif "entrance" in image_path.lower():
            location_text = "ở khu vực lối vào"
        else:
            location_text = "tại khu vực giám sát"
            
        return f"{base_desc} {location_text}. {confidence_text}"
    
    def _generate_fallback_description(self, event_type: str, confidence: float) -> str:
        """Generate simple fallback description in Vietnamese"""
        
        event_names = {
            "fall": "té ngã",
            "abnormal_behavior": "hành vi bất thường", 
            "seizure": "co giật",
            "emergency": "tình huống khẩn cấp"
        }
        
        event_name = event_names.get(event_type, "sự kiện y tế")
        confidence_text = f"(Độ tin cậy: {confidence:.1%})"
        
        return f"Phát hiện {event_name} {confidence_text}"
    
    def _enhance_caption_with_context(self, caption: str, event_type: str, confidence: float) -> str:
        """Enhance AI-generated caption with healthcare context"""
        
        # Add healthcare context to the generated caption
        context_prefixes = {
            "fall": "🚨 Cảnh báo ngã:",
            "abnormal_behavior": "⚠️ Hành vi bất thường:",
            "seizure": "🆘 Phát hiện co giật:",
            "emergency": "🚨 Tình huống khẩn cấp:"
        }
        
        prefix = context_prefixes.get(event_type, "📋 Phát hiện:")
        confidence_text = f"(Tin cậy: {confidence:.1%})"
        
        return f"{prefix} {caption} {confidence_text}"
    
    def is_available(self) -> bool:
        """Check if Vietnamese captioning service is available"""
        return self.is_initialized and self.model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "initialized": self.is_initialized,
            "model_available": self.model is not None,
            "model_path": str(self.model_path),
            "device": str(self.device)
        }
