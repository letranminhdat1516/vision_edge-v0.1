try:
    from seizure_detection.vsvig_detector import VSViGSeizureDetector as ExternalVSViGSeizureDetector
    from seizure_detection.seizure_predictor import SeizurePredictor as ExternalSeizurePredictor
except ImportError:
    class InternalVSViGSeizureDetector:
        def __init__(self, confidence_threshold=0.65):
            self.confidence_threshold = confidence_threshold
        def detect_seizure(self, frame, bbox):
            return {
                'temporal_ready': False,
                'keypoints': None,
                'confidence': 0.0
            }
    class InternalSeizurePredictor:
        def __init__(self, temporal_window=30, alert_threshold=0.65, warning_threshold=0.45):
            pass
        def update_prediction(self, confidence):
            return {
                'smoothed_confidence': 0.0,
                'seizure_detected': False,
                'alert_level': 'normal',
                'ready': False
            }

class SeizureDetectionService:
    def __init__(self, confidence_threshold=0.7, temporal_window=25, alert_threshold=0.7, warning_threshold=0.5):
        # Chọn đúng class dựa trên import
        if 'ExternalVSViGSeizureDetector' in globals():
            self.detector = ExternalVSViGSeizureDetector(confidence_threshold=confidence_threshold)
        else:
            self.detector = InternalVSViGSeizureDetector(confidence_threshold=confidence_threshold)
        if 'ExternalSeizurePredictor' in globals():
            self.predictor = ExternalSeizurePredictor(temporal_window=temporal_window, alert_threshold=alert_threshold, warning_threshold=warning_threshold)
        else:
            self.predictor = InternalSeizurePredictor(temporal_window=temporal_window, alert_threshold=alert_threshold, warning_threshold=warning_threshold)
    def detect_seizure(self, frame, bbox):
        return self.detector.detect_seizure(frame, bbox)
    def update_prediction(self, confidence):
        return self.predictor.update_prediction(confidence)
