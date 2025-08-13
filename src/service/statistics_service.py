import time
class StatisticsService:
    def __init__(self):
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'total_frames': 0,
            'keyframes_detected': 0,
            'motion_frames': 0,
            'start_time': None,
            'last_detection_time': None,
            'fall_detections': 0,
            'last_fall_time': None,
            'fall_confidence_avg': 0.0,
            'fall_false_positives': 0,
            'seizure_detections': 0,
            'last_seizure_time': None,
            'seizure_confidence_avg': 0.0,
            'seizure_warnings': 0,
            'pose_extraction_failures': 0,
            'critical_alerts': 0,
            'total_alerts': 0,
            'last_alert_time': None,
            'alert_type': 'normal'
        }
    def reset(self):
        self.__init__()
    def update(self, detection_result, person_count):
        self.stats['frames_processed'] += 1
        self.stats['persons_detected'] += person_count
        if person_count > 0:
            self.stats['last_detection_time'] = time.time()
        # Update averages
        if detection_result.get('fall_detected', False) and detection_result.get('fall_confidence', 0) > 0:
            current_avg = self.stats['fall_confidence_avg']
            current_count = self.stats['fall_detections']
            new_confidence = detection_result['fall_confidence']
            if current_count == 0:
                self.stats['fall_confidence_avg'] = new_confidence
            else:
                self.stats['fall_confidence_avg'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
        if detection_result.get('seizure_detected', False) and detection_result.get('seizure_confidence', 0) > 0:
            current_avg = self.stats['seizure_confidence_avg']
            current_count = self.stats['seizure_detections']
            new_confidence = detection_result['seizure_confidence']
            if current_count == 0:
                self.stats['seizure_confidence_avg'] = new_confidence
            else:
                self.stats['seizure_confidence_avg'] = (
                    (current_avg * current_count + new_confidence) / (current_count + 1)
                )
