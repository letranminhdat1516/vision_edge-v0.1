from fall_detection.simple_fall_detector import SimpleFallDetector
import time

class FallDetectionService:
    def __init__(self, confidence_threshold=0.40):  # TĂNG 0.15→0.40 để giảm false positive khi ngồi xuống
        self.detector = SimpleFallDetector(confidence_threshold=confidence_threshold)
    
    def detect_fall(self, frame, person=None, timestamp=None, person_bbox=None):
        """
        Detect fall in frame with flexible parameter handling.
        
        Args:
            frame: Current video frame
            person: Person detection dict (for backward compatibility)
            timestamp: Frame timestamp (optional)
            person_bbox: Person bounding box [x1, y1, x2, y2] (optional)
            
        Returns:
            dict: Fall detection result
        """
        # Handle timestamp
        if timestamp is None:
            timestamp = time.time()
        
        # Handle person_bbox - extract from person dict if needed
        if person_bbox is None and person is not None:
            if isinstance(person, dict) and 'bbox' in person:
                # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                bbox = person['bbox']
                person_bbox = [
                    int(bbox[0]), int(bbox[1]),
                    int(bbox[0] + bbox[2]),
                    int(bbox[1] + bbox[3])
                ]
        
        # Call the underlying detector with correct parameters
        return self.detector.detect_fall(frame, timestamp, person_bbox)
