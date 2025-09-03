"""
Sử dụng external APIs để generate natural language descriptions
"""

import requests
import base64
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ExternalVisionAPI:
    """Sử dụng external vision APIs để generate descriptions"""
    
    def __init__(self):
        self.apis = {
            'openai': self._openai_vision,
            'google': self._google_vision,
            'azure': self._azure_vision
        }
    
    def encode_image_to_base64(self, image_path):
        """Encode ảnh thành base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _openai_vision(self, image_path, api_key):
        """Sử dụng OpenAI GPT-4V để describe ảnh"""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hãy mô tả ảnh này bằng tiếng Việt, tập trung vào tình trạng sức khỏe và an toàn của người trong ảnh. Nếu có dấu hiệu nguy hiểm, hãy đề cập rõ ràng."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return None
    
    def _google_vision(self, image_path, api_key):
        """Sử dụng Google Vision API"""
        try:
            # Google Vision API implementation
            # Cần google-cloud-vision package
            pass
        except Exception as e:
            logger.error(f"Google Vision error: {e}")
            return None
    
    def _azure_vision(self, image_path, api_key):
        """Sử dụng Azure Computer Vision"""
        try:
            # Azure Cognitive Services implementation
            pass
        except Exception as e:
            logger.error(f"Azure Vision error: {e}")
            return None
    
    def generate_description(self, image_path, provider='openai', api_key=None):
        """Generate description using external API"""
        if provider in self.apis and api_key:
            return self.apis[provider](image_path, api_key)
        else:
            logger.warning("API key required for external vision services")
            return None

# Example usage
def demo_external_api():
    """Demo sử dụng external API"""
    api = ExternalVisionAPI()
    
    # Cần API key thật
    # api_key = "your-openai-api-key"
    # description = api.generate_description("path/to/image.jpg", "openai", api_key)
    
    print("External API demo - cần API key thật để test")

if __name__ == '__main__':
    demo_external_api()
