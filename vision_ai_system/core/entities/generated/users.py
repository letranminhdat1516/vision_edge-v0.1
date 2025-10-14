# Auto-generated from SQLAlchemy models. Do NOT edit by hand.
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass(slots=True)
class Users:
    user_id: Optional[str] = None
    username: str
    email: str
    password_hash: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    date_of_birth: Optional[datetime] = None
    phone_number: Optional[str] = None
    otp_code: Optional[str] = None
    otp_expires_at: Optional[datetime] = None

