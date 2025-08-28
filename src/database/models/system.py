"""System models for alerts, performance metrics, and other system-wide data."""

import uuid
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped

from .base import Base, TimestampMixin


class Alert(Base, TimestampMixin):
    """Alert model for system notifications."""

    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL, INFO
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    # Alert status
    status = Column(String(20), default="ACTIVE")  # ACTIVE, ACKNOWLEDGED, RESOLVED, SUPPRESSED
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))

    # Context data
    context = Column(JSONB, default={})

    # Indexes and constraints
    __table_args__ = (
        Index("idx_alerts_type", "alert_type"),
        Index("idx_alerts_severity", "severity"),
        Index("idx_alerts_status", "status"),
        Index("idx_alerts_created", "created_at"),
        CheckConstraint(
            "severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'INFO')", name="check_alert_severity"
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', 'SUPPRESSED')",
            name="check_alert_status",
        ),
    )

    def __repr__(self):
        return f"<Alert {self.alert_type}: {self.title}>"


class AlertRule(Base, TimestampMixin):
    """Alert rule configuration model."""

    __tablename__ = "alert_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL, INFO

    # Rule configuration
    query = Column(Text, nullable=False)  # PromQL or other query language
    threshold = Column(DECIMAL(20, 8), nullable=False)
    operator = Column(String(10), nullable=False)  # >, <, >=, <=, ==, !=
    duration = Column(String(20), nullable=False)  # e.g., "5m", "1h"

    # Notification settings
    notification_channels = Column(JSONB, default=[])  # List of channel types
    escalation_delay = Column(String(20))  # e.g., "15m"

    # Rule metadata
    labels = Column(JSONB, default={})
    annotations = Column(JSONB, default={})

    # Rule status
    enabled = Column(Boolean, default=True)  # Boolean for enabled state

    # Indexes and constraints
    __table_args__ = (
        Index("idx_alert_rules_name", "name"),
        Index("idx_alert_rules_severity", "severity"),
        Index("idx_alert_rules_enabled", "enabled"),
        CheckConstraint(
            "severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'INFO')",
            name="check_alert_rule_severity",
        ),
        CheckConstraint(
            "operator IN ('>', '<', '>=', '<=', '==', '!=')", name="check_alert_rule_operator"
        ),
        CheckConstraint("threshold >= 0", name="check_alert_rule_threshold_non_negative"),
    )

    def __repr__(self):
        return f"<AlertRule {self.name}: {self.severity}>"


class EscalationPolicy(Base, TimestampMixin):
    """Escalation policy for alert management."""

    __tablename__ = "escalation_policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

    # Escalation configuration
    severity_levels = Column(JSONB, default=[])  # List of severity levels this policy applies to
    escalation_steps = Column(
        JSONB, default=[]
    )  # List of escalation steps with delays and channels

    # Policy settings
    repeat_interval = Column(String(20))  # e.g., "30m" - how often to repeat notifications
    max_escalations = Column(Integer, default=3)  # Maximum number of escalations

    # Policy status
    enabled = Column(Boolean, default=True)  # Boolean for enabled state

    # Indexes and constraints
    __table_args__ = (
        Index("idx_escalation_policies_name", "name"),
        Index("idx_escalation_policies_enabled", "enabled"),
        CheckConstraint("max_escalations >= 1", name="check_escalation_max_escalations_positive"),
    )

    def __repr__(self):
        return f"<EscalationPolicy {self.name}>"


class AuditLog(Base, TimestampMixin):
    """General audit log model."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100))

    # User context
    user_id = Column(UUID(as_uuid=True))
    session_id = Column(String(100))

    # Action details
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    changes = Column(JSONB)

    # Request context
    ip_address = Column(String(45))
    user_agent = Column(Text)

    # Indexes
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_created", "created_at"),
    )

    def __repr__(self):
        return f"<AuditLog {self.action} on {self.entity_type}>"


class PerformanceMetrics(Base, TimestampMixin):
    """Performance metrics model."""

    __tablename__ = "performance_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100), nullable=False)

    # Metric values
    value = Column(DECIMAL(20, 8), nullable=False)
    previous_value = Column(DECIMAL(20, 8))
    change_percentage = Column(DECIMAL(8, 4))

    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_performance_type", "metric_type"),
        Index("idx_performance_entity", "entity_type", "entity_id"),
        Index("idx_performance_period", "period_start", "period_end"),
    )

    def __repr__(self):
        return f"<PerformanceMetrics {self.metric_type}: {self.value}>"


class BalanceSnapshot(Base, TimestampMixin):
    """Balance snapshot model for tracking account balances over time."""

    __tablename__ = "balance_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    account_type = Column(String(20), nullable=False)  # spot, futures, margin

    # Balance details
    currency: Mapped[str] = Column(String(10), nullable=False)
    total_balance: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    available_balance: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    locked_balance: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)

    # USD equivalent
    usd_value = Column(DECIMAL(20, 8))
    exchange_rate = Column(DECIMAL(20, 8))

    # Snapshot metadata
    snapshot_reason = Column(String(50))  # scheduled, triggered, manual

    # Indexes and constraints
    __table_args__ = (
        Index("idx_balance_exchange", "exchange"),
        Index("idx_balance_currency", "currency"),
        Index("idx_balance_created", "created_at"),
        Index("idx_balance_composite", "exchange", "currency", "created_at"),
        CheckConstraint("total_balance >= 0", name="check_balance_total_non_negative"),
        CheckConstraint("available_balance >= 0", name="check_balance_available_non_negative"),
        CheckConstraint("locked_balance >= 0", name="check_balance_locked_non_negative"),
        CheckConstraint(
            "total_balance = available_balance + locked_balance", name="check_balance_consistency"
        ),
        CheckConstraint("account_type IN ('spot', 'futures', 'margin')", name="check_account_type"),
    )

    def __repr__(self):
        return f"<BalanceSnapshot {self.exchange} {self.currency}: {self.total_balance}>"
