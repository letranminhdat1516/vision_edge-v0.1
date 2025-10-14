"""
EmailTemplates Model
Generated from table: email_templates
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class EmailTemplates(Base):
    __tablename__ = 'email_templates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    subject_template = Column(String(255), nullable=False)
    html_template = Column(Text, nullable=False)
    text_template = Column(Text)
    variables = Column(String)
    is_active = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<EmailTemplates(id={self.id})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
