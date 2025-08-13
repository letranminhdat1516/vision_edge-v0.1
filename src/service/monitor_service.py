import cv2
from service.camera_service import CameraService
from service.video_processing_service import VideoProcessingService
from service.fall_detection_service import FallDetectionService
from service.seizure_detection_service import SeizureDetectionService
from service.statistics_service import StatisticsService
from service.image_service import ImageService
from service.display_service import DisplayService

class MonitorService:
    def __init__(self, camera_config, processor_config, alerts_folder):
        self.camera_service = CameraService(camera_config)
        self.video_service = VideoProcessingService(processor_config)
        self.fall_service = FallDetectionService()
        self.seizure_service = SeizureDetectionService()
        self.statistics_service = StatisticsService()
        self.image_service = ImageService(alerts_folder)
        self.display_service = DisplayService()
        self.running = False
    def run(self):
        if not self.camera_service.connect():
            print("❌ Không kết nối được camera!")
            return
        self.running = True
        frame_count = 0
        while self.running:
            frame = self.camera_service.get_frame()
            if frame is None:
                continue
            frame_count += 1
            self.statistics_service.stats['total_frames'] = frame_count
            processing_result = self.video_service.process_frame(frame)
            if processing_result['processed']:
                person_detections = processing_result.get('person_detections', processing_result.get('detections', []))
                # Fall detection
                fall_result = self.fall_service.detect_fall(frame, person_detections[0] if person_detections else None)
                # Seizure detection
                seizure_result = self.seizure_service.detect_seizure(frame, [0,0,0,0])
                # Build detection_result
                detection_result = {
                    'fall_detected': fall_result.get('confidence', 0) >= 0.6,
                    'fall_confidence': fall_result.get('confidence', 0),
                    'seizure_detected': seizure_result.get('confidence', 0) >= 0.7,
                    'seizure_confidence': seizure_result.get('confidence', 0),
                    'alert_level': 'critical' if fall_result.get('confidence', 0) >= 0.6 or seizure_result.get('confidence', 0) >= 0.7 else 'normal',
                    'keypoints': seizure_result.get('keypoints', None),
                    'emergency_type': 'fall' if fall_result.get('confidence', 0) >= 0.6 else ('seizure' if seizure_result.get('confidence', 0) >= 0.7 else None)
                }
                self.statistics_service.update(detection_result, len(person_detections))
                # Lưu ảnh cảnh báo nếu có alert
                if detection_result['alert_level'] != 'normal':
                    self.image_service.save_alert_image(frame, detection_result['emergency_type'], detection_result['fall_confidence'] if detection_result['fall_detected'] else detection_result['seizure_confidence'])
                # Hiển thị 2 màn hình
                frame_vis = self.display_service.visualize_dual_detection(frame, detection_result, person_detections)
                frame_normal = self.display_service.create_normal_camera_window(frame, person_detections)
                cv2.imshow('Healthcare Monitor - Analysis View', frame_vis)
                cv2.imshow('Healthcare Monitor - Normal View', frame_normal)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        self.camera_service.disconnect()
        cv2.destroyAllWindows()
