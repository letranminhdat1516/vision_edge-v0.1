

import cv2
from service.monitor_service import MonitorService
from service.advanced_healthcare_pipeline import AdvancedHealthcarePipeline
from service.healthcare_event_publisher import healthcare_publisher
# ...import các service cần thiết...

if __name__ == "__main__":
    camera_config = {
        'url': 'rtsp://admin:L2C37340@192.168.8.122:554/cam/realmonitor?channel=1&subtype=1',
        'buffer_size': 1,
        'fps': 15,
        'resolution': (640, 480),
        'auto_reconnect': True
    }
    processor_config = 120
    alerts_folder = "examples/data/saved_frames/alerts"
    # Khởi tạo các service thật sự
    from service.camera_service import CameraService
    from service.video_processing_service import VideoProcessingService
    from service.fall_detection_service import FallDetectionService
    from service.seizure_detection_service import SeizureDetectionService

    camera = CameraService(camera_config)
    camera.connect()
    video_processor = VideoProcessingService(processor_config)
    fall_detector = FallDetectionService()
    seizure_detector = SeizureDetectionService()
    
    # Import và init seizure predictor
    from seizure_detection.seizure_predictor import SeizurePredictor
    seizure_predictor = SeizurePredictor(temporal_window=25, alert_threshold=0.7, warning_threshold=0.5)
    
    pipeline = AdvancedHealthcarePipeline(camera, video_processor, fall_detector, seizure_detector, seizure_predictor, alerts_folder)


    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        result = pipeline.process_frame(frame)
        detection_result = result["detection_result"]
        person_detections = result["person_detections"]
        
        # Hiển thị Normal View
        cv2.imshow("Healthcare Monitor - Normal View", result["normal_window"])
        
        # Hiển thị Analysis View với statistics overlay
        analysis_view = pipeline.visualize_dual_detection(frame, detection_result, person_detections)
        analysis_view = pipeline.draw_statistics_overlay(analysis_view, pipeline.stats)
        cv2.imshow("Healthcare Monitor - Analysis View", analysis_view)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # ...các xử lý khác như lưu ảnh, cập nhật thống kê...

    cv2.destroyAllWindows()

    # Xóa duplicate main block cuối file
