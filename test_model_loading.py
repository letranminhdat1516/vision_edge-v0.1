"""
Test Model Loading - Kiểm tra BLIP và Translation models có load được không
"""

import sys
import os
sys.path.append('src')

def test_model_loading():
    print("🔍 TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        from service.image_caption_service import ProfessionalVietnameseCaptionPipeline
        
        print("📥 Initializing pipeline...")
        pipeline = ProfessionalVietnameseCaptionPipeline()
        
        print(f"\n📊 MODEL STATUS:")
        print(f"   🤖 BLIP Model Loaded: {pipeline.blip_loaded}")
        print(f"   🌍 Translation Model Loaded: {pipeline.translator_loaded}")
        print(f"   💻 Device: {pipeline.device}")
        
        if pipeline.blip_loaded:
            print(f"   ✅ BLIP Processor: {type(pipeline.blip_processor).__name__}")
            print(f"   ✅ BLIP Model: {type(pipeline.blip_model).__name__}")
        else:
            print(f"   ❌ BLIP Model: Not loaded")
            
        if pipeline.translator_loaded:
            print(f"   ✅ Translator: {type(pipeline.translator).__name__}")
            print(f"   ✅ Translator Model: {pipeline.translator.model.name_or_path if hasattr(pipeline.translator, 'model') else 'Unknown'}")
        else:
            print(f"   ❌ Translator: Not loaded")
        
        # Test với text đơn giản
        print(f"\n🧪 TESTING TRANSLATION:")
        test_text = "a man is walking in the room"
        
        if pipeline.translator_loaded:
            print(f"   Input: {test_text}")
            result = pipeline.translator(test_text)
            print(f"   AI Translation: {result}")
        else:
            print(f"   Using rule-based translation...")
            result = pipeline._rule_based_translation(test_text)
            print(f"   Rule-based result: {result}")
        
        # Test dependencies
        print(f"\n📦 TESTING DEPENDENCIES:")
        try:
            import torch
            print(f"   ✅ PyTorch: {torch.__version__}")
            print(f"   🔥 CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print(f"   ❌ PyTorch: Not installed")
            
        try:
            import transformers
            print(f"   ✅ Transformers: {transformers.__version__}")
        except ImportError:
            print(f"   ❌ Transformers: Not installed")
            
        try:
            from PIL import Image
            print(f"   ✅ PIL: Available")
        except ImportError:
            print(f"   ❌ PIL: Not installed")
        
        # Test với ảnh giả lập nếu có BLIP
        if pipeline.blip_loaded:
            print(f"\n🖼️ TESTING IMAGE CAPTION (Simulated):")
            try:
                # Tạo ảnh test đơn giản
                from PIL import Image
                import numpy as np
                
                # Tạo ảnh trắng 224x224
                test_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
                test_path = "temp_test_image.jpg"
                test_image.save(test_path)
                
                print(f"   📸 Created test image: {test_path}")
                
                english_caption, status = pipeline.generate_english_caption(test_path)
                print(f"   🌍 English Caption: {english_caption}")
                print(f"   📊 Status: {status}")
                
                if english_caption:
                    vietnamese_caption, method = pipeline.translate_to_vietnamese(english_caption)
                    print(f"   🇻🇳 Vietnamese Caption: {vietnamese_caption}")
                    print(f"   🔧 Translation Method: {method}")
                
                # Cleanup
                os.remove(test_path)
                
            except Exception as e:
                print(f"   ❌ Image test failed: {e}")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        print(f"   This indicates models are not properly loaded")
        
        # Check specific issues
        print(f"\n🔍 CHECKING SPECIFIC ISSUES:")
        
        try:
            from transformers import BlipProcessor
            print(f"   ✅ Can import BlipProcessor")
        except ImportError as e:
            print(f"   ❌ Cannot import BlipProcessor: {e}")
            
        try:
            from transformers import pipeline
            print(f"   ✅ Can import transformers pipeline")
        except ImportError as e:
            print(f"   ❌ Cannot import transformers pipeline: {e}")

if __name__ == "__main__":
    test_model_loading()
