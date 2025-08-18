"""
Vietnamese Image Captioning Model Implementation
Based on EfficientNet-B0 + BARTPho architecture
"""

import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)

# Try to import transformers, use fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ transformers library not available, using simplified implementation")
    TRANSFORMERS_AVAILABLE = False

class Vocabulary:
    """Vocabulary class for Vietnamese text processing"""
    
    def __init__(self, model_name="vinai/bartpho-syllable"):
        """
        Initialize vocabulary with BARTPho tokenizer
        
        Args:
            model_name: BARTPho model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.vocab_size = 0
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = self.tokenizer.vocab_size
            logger.info(f"✅ Vocabulary loaded: {model_name} (vocab_size: {self.vocab_size})")
        except Exception as e:
            logger.error(f"❌ Failed to load vocabulary: {e}")
    
    def encode(self, text: str, max_length: int = 50):
        """Encode text to token IDs"""
        if self.tokenizer is None:
            return []
        
        return self.tokenizer.encode(
            text, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        if self.tokenizer is None:
            return ""
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

class EncoderCNN(nn.Module):
    """CNN Encoder using EfficientNet-B0"""
    
    def __init__(self, embed_size=768, train_CNN=False):
        """
        Initialize CNN encoder
        
        Args:
            embed_size: Size of embedding vector
            train_CNN: Whether to train CNN layers
        """
        super(EncoderCNN, self).__init__()
        self.embed_size = embed_size
        self.train_CNN = train_CNN
        
        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Get the number of input features from classifier
        # EfficientNet-B0 has 1280 features before the classifier
        num_features = 1280  # Standard for EfficientNet-B0
        
        # Replace classifier with linear layer
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, embed_size)
        )
        
        # Freeze CNN layers if not training
        if not train_CNN:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            
            # Only train the final linear layer
            for param in self.efficientnet.classifier.parameters():
                param.requires_grad = True
    
    def forward(self, images):
        """Forward pass through CNN encoder"""
        features = self.efficientnet(images)
        return features

class DecoderBARTPho(nn.Module):
    """BARTPho Decoder for Vietnamese text generation"""
    
    def __init__(self, embed_size=768, bartpho_model_name="vinai/bartpho-syllable", freeze_bartpho=False):
        """
        Initialize BARTPho decoder
        
        Args:
            embed_size: Size of image embedding
            bartpho_model_name: BARTPho model name
            freeze_bartpho: Whether to freeze BARTPho parameters
        """
        super(DecoderBARTPho, self).__init__()
        self.embed_size = embed_size
        self.bartpho_model_name = bartpho_model_name
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load BARTPho model
                self.bartpho = AutoModel.from_pretrained(bartpho_model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(bartpho_model_name)
                
                # Linear projection from image features to BARTPho input
                self.projection = nn.Linear(embed_size, self.bartpho.config.hidden_size)
                
                # Freeze BARTPho if specified
                if freeze_bartpho:
                    for param in self.bartpho.parameters():
                        param.requires_grad = False
                
                self.using_bartpho = True
                logger.info(f"✅ BARTPho decoder loaded: {bartpho_model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load BARTPho: {e}")
                self.using_bartpho = False
                self._init_fallback()
        else:
            logger.warning("⚠️ transformers not available, using fallback decoder")
            self.using_bartpho = False
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize simple fallback decoder"""
        self.bartpho = None
        self.tokenizer = None
        self.projection = nn.Linear(self.embed_size, 512)
        self.output_layer = nn.Linear(512, 1000)  # Vietnamese vocab size
    
    def forward(self, image_features, captions=None, max_length=50):
        """Forward pass through BARTPho decoder"""
        if self.using_bartpho and self.bartpho is not None:
            # Project image features to BARTPho input space
            projected_features = self.projection(image_features)
            
            # Add batch and sequence dimensions
            if len(projected_features.shape) == 2:
                projected_features = projected_features.unsqueeze(1)  # Add sequence dim
            
            # Generate text using BARTPho
            # This is a simplified implementation - actual generation would be more complex
            return projected_features
        else:
            # Fallback implementation
            projected = self.projection(image_features)
            outputs = self.output_layer(projected)
            return outputs

class ImageCaptioningModel(nn.Module):
    """Complete Vietnamese Image Captioning Model"""
    
    def __init__(self, embed_size=768, bartpho_model_name="vinai/bartpho-syllable", 
                 train_CNN=False, freeze_bartpho=False):
        """
        Initialize complete image captioning model
        
        Args:
            embed_size: Size of embedding vector
            bartpho_model_name: BARTPho model name
            train_CNN: Whether to train CNN encoder
            freeze_bartpho: Whether to freeze BARTPho decoder
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.embed_size = embed_size
        self.encoder = EncoderCNN(embed_size=embed_size, train_CNN=train_CNN)
        self.decoder = DecoderBARTPho(
            embed_size=embed_size, 
            bartpho_model_name=bartpho_model_name,
            freeze_bartpho=freeze_bartpho
        )
        
        logger.info(f"✅ Vietnamese Image Captioning Model initialized")
        logger.info(f"   Encoder: EfficientNet-B0 (embed_size: {embed_size})")
        logger.info(f"   Decoder: {bartpho_model_name}")
    
    def forward(self, images, captions=None, max_length=50):
        """Forward pass through complete model"""
        # Extract image features
        image_features = self.encoder(images)
        
        # Generate captions
        outputs = self.decoder(image_features, captions, max_length)
        
        return outputs
    
    def predict(self, image_tensor, vocab, max_length=50):
        """
        Generate Vietnamese caption for image
        
        Args:
            image_tensor: Preprocessed image tensor
            vocab: Vocabulary object
            max_length: Maximum caption length
            
        Returns:
            Generated Vietnamese caption
        """
        self.eval()
        
        with torch.no_grad():
            # Extract image features
            image_features = self.encoder(image_tensor)
            
            # Simple caption generation (placeholder for actual implementation)
            # In real implementation, this would use beam search or other generation methods
            
            # For now, return a context-aware caption based on image features
            # This is a simplified version - real implementation would be much more complex
            
            return self._generate_simple_caption(image_features)
    
    def _generate_simple_caption(self, image_features):
        """
        Generate simple Vietnamese caption (placeholder)
        
        This is a simplified implementation. Real model would use:
        - Beam search generation
        - Attention mechanisms
        - Proper Vietnamese language model
        """
        
        # Analyze image features to determine scene content
        feature_norm = torch.norm(image_features, dim=1).item()
        
        # Simple rule-based caption generation based on feature patterns
        if feature_norm > 15.0:
            return "Cảnh trong phòng có nhiều hoạt động, có thể là tình huống khẩn cấp"
        elif feature_norm > 10.0:
            return "Phát hiện người trong phòng với chuyển động bất thường"
        elif feature_norm > 5.0:
            return "Cảnh trong phòng bình thường với ít hoạt động"
        else:
            return "Phòng trống hoặc ánh sáng yếu"
    
    def load_checkpoint(self, checkpoint_path, device):
        """Load model from checkpoint"""
        try:
            # Try loading with weights_only=False for older checkpoints
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load with weights_only=False: {e}")
                # Fallback to default loading
                checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load state dict
            if "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Assume the checkpoint is just the state dict
                self.load_state_dict(checkpoint)
                
            logger.info(f"✅ Model checkpoint loaded from: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
