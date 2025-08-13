import cv2
from pathlib import Path
from datetime import datetime
import json

class ImageService:
    def __init__(self, alerts_folder):
        self.alerts_folder = Path(alerts_folder)
        self.alerts_folder.mkdir(parents=True, exist_ok=True)
    def save_alert_image(self, frame, alert_type, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        filename = f"alert_{timestamp}_{alert_type}_conf_{confidence:.3f}.jpg"
        filepath = self.alerts_folder / filename
        success = cv2.imwrite(str(filepath), frame)
        if success:
            metadata = {
                "alert_type": alert_type,
                "confidence": str(confidence),
                "keyframe_confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            metadata_file = filepath.with_suffix('.jpg_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return filename
        return None
