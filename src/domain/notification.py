# Dataclass cho Notification
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Notification:
	notification_id: str
	alert_id: str
	user_id: str
	notification_type: str
	message: str
	delivery_data: Optional[Dict]
	status: str
	sent_at: Optional[str]
	delivered_at: Optional[str]
	retry_count: int
	error_message: Optional[str]
# Entity/model for Notification
