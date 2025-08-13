
import cv2
from pathlib import Path
from datetime import datetime
from src.domain.alert_image_metadata import AlertImageMetadata
from typing import Optional

class ImageSaver:
    def __init__(self, alerts_folder: str):
        self.alerts_folder = Path(alerts_folder)
        self.alerts_folder.mkdir(parents=True, exist_ok=True)

    def save_alert_image(self, frame, alert_type: str, confidence: float) -> 'Optional[str]':
        """
        Lưu ảnh cảnh báo và metadata vào thư mục alerts
        Returns: filename nếu thành công, None nếu lỗi
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"alert_{timestamp}_{alert_type}_conf_{confidence:.3f}.jpg"
            filepath = self.alerts_folder / filename
            success = cv2.imwrite(str(filepath), frame)
            if success:
                metadata = AlertImageMetadata(
                    alert_type=alert_type,
                    confidence=confidence,
                    keyframe_confidence=0.0,
                    timestamp=datetime.now().isoformat()
                )
                metadata_file = filepath.with_suffix('.jpg_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    f.write(str(metadata))
                return filename
            return None
        except Exception as e:
            print(f"❌ Lỗi lưu ảnh cảnh báo: {e}")
            return None
