import torch
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/Vietnamese-Image-Captioning/best_image_captioning_model_vietnamese.pth.tar')

def debug_model_structure():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    
    print("Checkpoint type:", type(checkpoint))
    print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            print(f"{key}: {type(value)}")
            if hasattr(value, 'keys') and callable(getattr(value, 'keys')):
                print(f"  -> {key} keys: {list(value.keys())[:5]}...")  # First 5 keys

if __name__ == '__main__':
    debug_model_structure()
