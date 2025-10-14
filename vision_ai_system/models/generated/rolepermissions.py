from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import BigInteger, Boolean, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, Numeric, PrimaryKeyConstraint, SmallInteger, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB, TIME, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class RolePermissions(Base):
    __tablename__ = 'role_permissions'
    __table_args__ = (
        ForeignKeyConstraint(['permission_id'], ['permissions.id'], ondelete='CASCADE', onupdate='CASCADE', name='role_permissions_permission_id_fkey'),
        ForeignKeyConstraint(['role_id'], ['roles.id'], ondelete='CASCADE', onupdate='CASCADE', name='role_permissions_role_id_fkey'),
        PrimaryKeyConstraint('role_id', 'permission_id', name='role_permissions_pkey'),
        Index('idx_role_permissions_permission', 'permission_id'),
        Index('idx_role_permissions_role', 'role_id')
    )

    role_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    permission_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    assigned_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(True, 6), nullable=False, server_default=text('CURRENT_TIMESTAMP'))

    permission: Mapped['Permissions'] = relationship('Permissions', back_populates='role_permissions')
    role: Mapped['Roles'] = relationship('Roles', back_populates='role_permissions')
