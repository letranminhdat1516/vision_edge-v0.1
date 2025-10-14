# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class RolePermissions:
    role_id: Optional[str] = None
    permission_id: Optional[str] = None
    assigned_at: datetime

