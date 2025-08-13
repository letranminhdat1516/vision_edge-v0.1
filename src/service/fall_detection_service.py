from fall_detection.simple_fall_detector import SimpleFallDetector

class FallDetectionService:
    def __init__(self, confidence_threshold=0.4):
        self.detector = SimpleFallDetector(confidence_threshold=confidence_threshold)
    def detect_fall(self, frame, person):
        return self.detector.detect_fall(frame, person)
