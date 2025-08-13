from src.domain.event_detection import EventDetection
import time

class EventService:
	def create_event(self, user_id: str, event_type: str, confidence_score: float, image_url: str, status: str, action: str, timestamp: int):
		"""
		Create and save an event (placeholder, add DB logic as needed)
		"""
		detected_at_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(timestamp))
		event = EventDetection(
			event_id="",  # Can be generated or left blank for DB auto-increment
			snapshot_id="",
			user_id=user_id,
			camera_id="",
			room_id="",
			event_type=event_type,
			event_description=f"{action} detected with status {status}",
			detection_data={"image_url": image_url},
			ai_analysis_result={"confidence_score": confidence_score},
			confidence_score=confidence_score,
			bounding_boxes=None,
			status=status,
			context_data=None,
			detected_at=detected_at_str,
			verified_at=None,
			verified_by=None,
			acknowledged_at=None,
			acknowledged_by=None,
			dismissed_at=None,
			created_at=detected_at_str
		)
		# TODO: Save event to DB (add repository logic)
