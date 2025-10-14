# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class SystemSettings:
    setting_id: Optional[str] = None
    setting_key: str
    setting_value: str
    data_type: str
    updated_at: datetime
    updated_by: str
    description: Optional[str] = None
    category: Optional[str] = None
    is_encrypted: Optional[bool] = None

