import torch
from torchvision import transforms
from PIL import Image
import os
import glob

# Đường dẫn tới model và thư mục chứa ảnh test
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

# Hàm load model

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.eval()
    return model

# Hàm tiền xử lý ảnh

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image)
    if isinstance(tensor, torch.Tensor):
        return tensor.unsqueeze(0)
    else:
        raise TypeError('transform(image) did not return a Tensor')

# Hàm sinh caption

def generate_caption(model, image_tensor):
    # Giả sử model có phương thức generate_caption
    if hasattr(model, 'generate_caption'):
        return model.generate_caption(image_tensor)
    # Nếu không, thử forward
    output = model(image_tensor)
    # Chuyển output thành caption (tùy thuộc vào model)
    if isinstance(output, str):
        return output
    elif isinstance(output, dict) and 'caption' in output:
        return output['caption']
    else:
        return str(output)

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f'Model file not found: {MODEL_PATH}')
    else:
        try:
            # Lấy ảnh mới nhất từ thư mục alerts
            latest_image_path = get_latest_image()
            print(f'Using latest image: {os.path.basename(latest_image_path)}')
            
            # Load model và xử lý ảnh
            model = load_model(MODEL_PATH)
            image_tensor = preprocess_image(latest_image_path)
            caption = generate_caption(model, image_tensor)
            print('Generated caption:', caption)
        except Exception as e:
            print(f'Error: {e}')
