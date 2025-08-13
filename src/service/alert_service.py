from src.data.alert_repo import AlertRepository
from src.domain.alert import Alert
import time

class AlertService:
	def __init__(self):
		self.repo = AlertRepository()

	def create_alert(self, event_id: str, user_id: str, status: str, action: str, image_url: str, timestamp: int):
		"""
		Create and save an alert to the DB
		"""
		created_at_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(timestamp))
		alert = Alert(
			alert_id="",  # Can be generated or left blank for DB auto-increment
			event_id=event_id,
			user_id=user_id,
			alert_type=action,
			severity=status,
			alert_message=f"{action} detected with status {status}",
			alert_data={"image_url": image_url},
			status=status,
			acknowledged_by=None,
			acknowledged_at=None,
			resolution_notes=None,
			created_at=created_at_str,
			resolved_at=None
		)
		self.repo.save_alert(alert)
