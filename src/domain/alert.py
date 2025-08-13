# Dataclass cho Alert
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Alert:
	alert_id: str
	event_id: str
	user_id: str
	alert_type: str
	severity: str
	alert_message: str
	alert_data: Optional[Dict]
	status: str
	acknowledged_by: Optional[str]
	acknowledged_at: Optional[str]
	resolution_notes: Optional[str]
	created_at: str
	resolved_at: Optional[str]
# Entity/model for Alert
