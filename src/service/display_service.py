import cv2
from datetime import datetime

class DisplayService:
    def visualize_dual_detection(self, frame, detection_result, person_detections):
        frame_vis = frame.copy()
        for person in person_detections:
            bbox = person['bbox']
            confidence = person['confidence']
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)
            if detection_result['alert_level'] == 'critical':
                color = (0, 0, 255)
            elif detection_result['alert_level'] == 'high':
                color = (0, 165, 255)
            elif detection_result['alert_level'] == 'warning':
                color = (0, 255, 255)
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_vis, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # ...keypoints, alerts, statistics overlay (có thể bổ sung thêm nếu cần)...
        return frame_vis
    def create_normal_camera_window(self, frame, person_detections):
        frame_normal = frame.copy()
        for person in person_detections:
            bbox = person['bbox']
            confidence = person['confidence']
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)
            cv2.rectangle(frame_normal, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_normal, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame_normal, "Normal Camera View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        person_count = len(person_detections)
        status_text = f"Persons Detected: {person_count}"
        cv2.putText(frame_normal, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_normal, timestamp, (10, frame_normal.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame_normal
