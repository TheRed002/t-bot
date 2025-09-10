"""Bot instance database model."""

import uuid
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, MetadataMixin, TimestampMixin


class BotInstance(Base, TimestampMixin, MetadataMixin):
    """Bot instance model for managing individual trading bots."""

    __tablename__ = "bot_instances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)

    # Link to main bot record
    bot_id = Column(
        UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # Bot configuration
    strategy_type = Column(String(50), nullable=False)
    exchange = Column(String(50), nullable=False)
    status = Column(
        String(20), default="stopped", nullable=False
    )  # stopped, running, paused, error

    # Configuration and settings
    config = Column(JSONB, nullable=False, default=dict)

    # Activity tracking
    last_active = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=False)

    # Capital tracking
    allocated_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    current_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Relationships
    bot = relationship("Bot", back_populates="instances")
    capital_audit_logs = relationship(
        "CapitalAuditLog", foreign_keys="CapitalAuditLog.bot_id", back_populates="bot_instance"
    )
    execution_audit_logs = relationship(
        "ExecutionAuditLog", foreign_keys="ExecutionAuditLog.bot_id", back_populates="bot_instance"
    )
    risk_audit_logs = relationship(
        "RiskAuditLog", foreign_keys="RiskAuditLog.bot_id", back_populates="bot_instance"
    )
    performance_audit_logs = relationship(
        "PerformanceAuditLog",
        foreign_keys="PerformanceAuditLog.bot_id",
        back_populates="bot_instance",
    )

    # Risk management relationships - fixed foreign keys
    risk_configurations = relationship(
        "RiskConfiguration",
        foreign_keys="RiskConfiguration.bot_id",
        back_populates="bot_instance",
        cascade="all, delete-orphan",
    )
    risk_violations = relationship(
        "RiskViolation",
        foreign_keys="RiskViolation.bot_id",
        back_populates="bot_instance",
        cascade="all, delete-orphan",
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_bot_instances_name", "name"),
        Index("idx_bot_instances_bot_id", "bot_id"),
        Index("idx_bot_instances_strategy_type", "strategy_type"),
        Index("idx_bot_instances_exchange", "exchange"),
        Index("idx_bot_instances_status", "status"),
        Index("idx_bot_instances_active", "is_active"),
        Index("idx_bot_instances_last_active", "last_active"),  # Activity tracking
        Index(
            "idx_bot_instances_strategy_exchange", "strategy_type", "exchange"
        ),  # Composite query optimization
        CheckConstraint(
            "status IN ('stopped', 'running', 'paused', 'error')", name="check_bot_instance_status"
        ),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')", name="check_bot_instance_exchange"
        ),
        CheckConstraint(
            "allocated_capital >= 0", name="check_bot_instance_allocated_capital_non_negative"
        ),
        CheckConstraint(
            "current_balance >= 0", name="check_bot_instance_current_balance_non_negative"
        ),
    )

    def __repr__(self):
        return f"<BotInstance {self.name}: {self.strategy_type} on {self.exchange}>"

    def is_running(self) -> bool:
        """Check if bot instance is running."""
        return bool(self.status == "running")

    def is_stopped(self) -> bool:
        """Check if bot instance is stopped."""
        return bool(self.status == "stopped")
