"""User and authentication database models."""

import uuid

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from .base import Base, MetadataMixin, TimestampMixin


class User(Base, TimestampMixin, MetadataMixin):
    """User model for authentication and account management."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)

    # Authentication tracking
    last_login_at = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))  # Account lockout timestamp

    # Permissions
    scopes = Column(JSON, default=lambda: ["read"])  # JSON array of permission scopes

    # Indexes
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_active", "is_active"),
    )

    def __repr__(self):
        return f"<User {self.username}>"

    @property
    def full_name(self) -> str:
        """Get full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return str(self.username)

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return bool(self.is_active) and bool(self.is_verified)
