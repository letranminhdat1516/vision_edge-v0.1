import time
from typing import Dict, Any
from src.domain.alert import Alert
from src.data.alert_repo import AlertRepository

class AIPipeline:
	"""
	Main AI pipeline for healthcare monitoring (fall/seizure detection, statistics, alert saving)
	"""
	def __init__(self):
		self.alert_repo = AlertRepository()
		# TODO: Initialize detectors, processors, etc. (YOLO, VSViG, etc.)

	def process_frame(self, frame: Any, detection_result: Dict[str, Any]) -> Alert:
		"""
		Process a frame, run detection, and save alert to DB if needed.
		Args:
			frame: Camera frame (np.ndarray)
			detection_result: Dict with detection info (from AI models)
		Returns:
			Alert object (if alert generated)
		"""
		# Map detection_result to Alert fields
		now_str = time.strftime('%Y-%m-%dT%H:%M:%S')
		alert = Alert(
			alert_id=detection_result.get("alert_id", ""),
			event_id=detection_result.get("event_id", ""),
			user_id=detection_result.get("user_id", ""),
			alert_type=detection_result.get("alert_type", ""),
			severity=detection_result.get("severity", "normal"),
			alert_message=detection_result.get("alert_message", ""),
			alert_data=detection_result.get("alert_data", {}),
			status=detection_result.get("status", "normal"),
			acknowledged_by=None,
			acknowledged_at=None,
			resolution_notes=None,
			created_at=now_str,
			resolved_at=None
		)
		# Save to DB if status is not normal
		if alert.status != "normal":
			self.alert_repo.save_alert(alert)
		return alert
