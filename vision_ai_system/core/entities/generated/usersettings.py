# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class UserSettings:
    id: Optional[str] = None
    user_id: str
    category: str
    setting_key: str
    setting_value: str
    is_enabled: bool
    is_overridden: bool
    overridden_at: Optional[datetime] = None

