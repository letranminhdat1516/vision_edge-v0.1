import torch
from PIL import Image
from torchvision import transforms
import os
import glob

# Đường dẫn tới model và thư mục ảnh
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/Vietnamese-Image-Captioning/best_image_captioning_model_vietnamese.pth.tar')
ALERTS_FOLDER = os.path.join(os.path.dirname(__file__), '../src/examples/data/saved_frames/alerts')

def get_latest_image():
    """Lấy ảnh mới nhất từ thư mục alerts"""
    image_files = glob.glob(os.path.join(ALERTS_FOLDER, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"Không tìm thấy file ảnh nào trong {ALERTS_FOLDER}")
    
    # Sắp xếp theo thời gian tạo file (mới nhất đầu tiên)
    latest_image = max(image_files, key=os.path.getctime)
    return latest_image

def load_model_checkpoint():
    """Load checkpoint và trả về state_dict"""
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    print(f"Checkpoint loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best CIDEr score: {checkpoint.get('best_cider_score', 'N/A')}")
    
    state_dict = checkpoint['state_dict']
    print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
    return state_dict

def preprocess_image(image_path):
    """Tiền xử lý ảnh theo format mà model cần"""
    image = Image.open(image_path).convert('RGB')
    
    # Transform theo README: Resize(256) → CenterCrop(224) → Normalize(ImageNet)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image)
    if isinstance(tensor, torch.Tensor):
        return tensor.unsqueeze(0)  # Thêm batch dimension
    else:
        raise TypeError('transform(image) did not return a Tensor')

def analyze_model_architecture(state_dict):
    """Phân tích architecture của model từ state_dict"""
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder')]
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder')]
    other_keys = [k for k in state_dict.keys() if not k.startswith(('encoder', 'decoder'))]
    
    print(f"\n=== Model Architecture Analysis ===")
    print(f"Encoder layers: {len(encoder_keys)}")
    print(f"Decoder layers: {len(decoder_keys)}")
    print(f"Other layers: {len(other_keys)}")
    
    print(f"\nSample encoder keys:")
    for key in encoder_keys[:3]:
        print(f"  - {key}: {state_dict[key].shape}")
        
    print(f"\nSample decoder keys:")
    for key in decoder_keys[:3]:
        print(f"  - {key}: {state_dict[key].shape}")

if __name__ == '__main__':
    try:
        # 1. Load và phân tích model
        print("=== Loading Model Checkpoint ===")
        state_dict = load_model_checkpoint()
        
        # 2. Phân tích architecture
        analyze_model_architecture(state_dict)
        
        # 3. Kiểm tra ảnh
        print(f"\n=== Processing Latest Image ===")
        latest_image_path = get_latest_image()
        print(f'Using latest image: {os.path.basename(latest_image_path)}')
        
        # 4. Tiền xử lý ảnh
        image_tensor = preprocess_image(latest_image_path)
        print(f'Image tensor shape: {image_tensor.shape}')
        
        print(f"\n✅ Model checkpoint và ảnh được load thành công!")
        print(f"💡 Để sử dụng model, bạn cần:")
        print(f"   1. Tạo model architecture (ImageCaptioningModel)")
        print(f"   2. Load state_dict vào model")
        print(f"   3. Tạo vocabulary cho BARTPho")
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
