"""User and authentication database models."""

import uuid
from decimal import Decimal

from sqlalchemy import DECIMAL, JSON, Boolean, CheckConstraint, Column, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    # Trading and financial fields
    allocated_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0, nullable=False)
    max_daily_loss: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0, nullable=False)
    risk_tolerance: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=0.5, nullable=False)

    # Relationships
    balance_snapshots = relationship("BalanceSnapshot", back_populates="user")
    backtest_runs = relationship("BacktestRun", back_populates="user", cascade="all, delete-orphan")

    # Alert relationships - fix missing back_populates for system models
    alert_rules_created = relationship(
        "AlertRule",
        foreign_keys="AlertRule.created_by",
        back_populates="creator",
        overlaps="creator"
    )
    escalation_policies_created = relationship(
        "EscalationPolicy",
        foreign_keys="EscalationPolicy.created_by",
        back_populates="creator",
        overlaps="creator"
    )
    audit_logs = relationship("AuditLog", back_populates="user", overlaps="user")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_active", "is_active"),
        Index("idx_users_verified", "is_verified"),  # User verification status
        Index("idx_users_login", "last_login_at"),  # Recent activity tracking
        Index("idx_users_locked", "locked_until"),  # Account lockout queries
        Index("idx_users_allocated_capital", "allocated_capital"),  # Capital allocation queries
        Index("idx_users_active_verified", "is_active", "is_verified"),  # Active user queries
        Index("idx_users_risk_tolerance", "risk_tolerance"),  # Risk-based user queries
        CheckConstraint(
            "failed_login_attempts >= 0", name="check_failed_login_attempts_non_negative"
        ),
        CheckConstraint("LENGTH(username) >= 3", name="check_username_min_length"),
        CheckConstraint("LENGTH(email) >= 5", name="check_email_min_length"),
        CheckConstraint("email LIKE '%@%'", name="check_email_format"),
        CheckConstraint(
            "failed_login_attempts <= 10", name="check_failed_login_attempts_max"
        ),  # Reasonable limit
        CheckConstraint("allocated_capital >= 0", name="check_allocated_capital_non_negative"),
        CheckConstraint("max_daily_loss >= 0", name="check_max_daily_loss_non_negative"),
        CheckConstraint(
            "risk_tolerance >= 0 AND risk_tolerance <= 1", name="check_risk_tolerance_range"
        ),
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
