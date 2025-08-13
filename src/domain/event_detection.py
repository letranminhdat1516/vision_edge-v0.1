# Dataclass cho EventDetection
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class EventDetection:
	event_id: str
	snapshot_id: str
	user_id: str
	camera_id: str
	room_id: str
	event_type: str
	event_description: Optional[str]
	detection_data: Optional[Dict]
	ai_analysis_result: Optional[Dict]
	confidence_score: float
	bounding_boxes: Optional[Dict]
	status: str
	context_data: Optional[Dict]
	detected_at: str
	verified_at: Optional[str]
	verified_by: Optional[str]
	acknowledged_at: Optional[str]
	acknowledged_by: Optional[str]
	dismissed_at: Optional[str]
	created_at: str
# Entity/model for EventDetection
