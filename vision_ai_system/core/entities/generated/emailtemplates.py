# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class EmailTemplates:
    id: Optional[str] = None
    name: str
    type: str
    subject_template: str
    html_template: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    text_template: Optional[str] = None
    variables: Optional[dict[str, Any]] = None

