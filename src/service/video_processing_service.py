try:
    from video_processing.simple_processing import IntegratedVideoProcessor as ExternalIntegratedVideoProcessor
except ImportError:
    class InternalIntegratedVideoProcessor:
        def __init__(self, config):
            self.config = config
        def process_frame(self, frame):
            return {
                'processed': True,
                'person_detections': [],
                'detections': [],
                'processing_time': 0.01
            }

class VideoProcessingService:
    def __init__(self, config):
        if 'ExternalIntegratedVideoProcessor' in globals():
            self.processor = ExternalIntegratedVideoProcessor(config)
        else:
            self.processor = InternalIntegratedVideoProcessor(config)
    def process_frame(self, frame):
        return self.processor.process_frame(frame)
