"""System models for alerts, performance metrics, and other system-wide data."""

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
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    # Foreign keys
    alert_rule_id = Column(UUID(as_uuid=True), ForeignKey("alert_rules.id", ondelete="SET NULL"))
    escalation_policy_id = Column(UUID(as_uuid=True), ForeignKey("escalation_policies.id", ondelete="SET NULL"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="SET NULL"))

    # Context data
    context = Column(JSONB, default={})

    # Relationships
    alert_rule = relationship("AlertRule", back_populates="alerts")
    escalation_policy = relationship("EscalationPolicy", back_populates="alerts")
    user = relationship("User")
    bot = relationship("Bot")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_alerts_type", "alert_type"),
        Index("idx_alerts_severity", "severity"),
        Index("idx_alerts_status", "status"),
        Index("idx_alerts_created", "created_at"),
        Index("idx_alerts_rule_id", "alert_rule_id"),
        Index("idx_alerts_user_id", "user_id"),
        Index("idx_alerts_bot_id", "bot_id"),
        Index("idx_alerts_status_severity", "status", "severity"),
        CheckConstraint("severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'INFO')", name="check_alert_severity"),
        CheckConstraint(
            "status IN ('ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', 'SUPPRESSED')",
            name="check_alert_status",
        ),
        CheckConstraint(
            "acknowledged_at IS NULL OR acknowledged_by IS NOT NULL",
            name="check_alert_acknowledge_consistency",
        ),
        CheckConstraint("resolved_at IS NULL OR status = 'RESOLVED'", name="check_alert_resolve_consistency"),
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
    threshold: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    operator = Column(String(10), nullable=False)  # >, <, >=, <=, ==, !=
    duration = Column(String(20), nullable=False)  # e.g., "5m", "1h"

    # Notification settings
    notification_channels = Column(JSONB, default=[])  # List of channel types
    escalation_delay = Column(String(20))  # e.g., "15m"

    # Foreign keys
    escalation_policy_id = Column(UUID(as_uuid=True), ForeignKey("escalation_policies.id", ondelete="SET NULL"))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    # Rule metadata
    labels = Column(JSONB, default={})
    annotations = Column(JSONB, default={})

    # Rule status
    enabled = Column(Boolean, default=True)  # Boolean for enabled state

    # Relationships
    escalation_policy = relationship("EscalationPolicy", back_populates="alert_rules")
    alerts = relationship("Alert", back_populates="alert_rule", cascade="all, delete-orphan")
    creator = relationship("User")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_alert_rules_name", "name"),
        Index("idx_alert_rules_severity", "severity"),
        Index("idx_alert_rules_enabled", "enabled"),
        Index("idx_alert_rules_escalation_policy", "escalation_policy_id"),
        Index("idx_alert_rules_created_by", "created_by"),
        Index("idx_alert_rules_enabled_severity", "enabled", "severity"),
        CheckConstraint(
            "severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'INFO')",
            name="check_alert_rule_severity",
        ),
        CheckConstraint("operator IN ('>', '<', '>=', '<=', '==', '!=')", name="check_alert_rule_operator"),
        CheckConstraint("threshold >= 0", name="check_alert_rule_threshold_non_negative"),
        CheckConstraint("LENGTH(name) >= 1", name="check_alert_rule_name_not_empty"),
        CheckConstraint("LENGTH(query) >= 1", name="check_alert_rule_query_not_empty"),
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
    escalation_steps = Column(JSONB, default=[])  # List of escalation steps with delays and channels

    # Policy settings
    repeat_interval = Column(String(20))  # e.g., "30m" - how often to repeat notifications
    max_escalations = Column(Integer, default=3)  # Maximum number of escalations

    # Foreign keys
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    # Policy status
    enabled = Column(Boolean, default=True)  # Boolean for enabled state

    # Relationships
    alert_rules = relationship("AlertRule", back_populates="escalation_policy")
    alerts = relationship("Alert", back_populates="escalation_policy")
    creator = relationship("User")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_escalation_policies_name", "name"),
        Index("idx_escalation_policies_enabled", "enabled"),
        Index("idx_escalation_policies_created_by", "created_by"),
        CheckConstraint("max_escalations >= 1", name="check_escalation_max_escalations_positive"),
        CheckConstraint("LENGTH(name) >= 1", name="check_escalation_policy_name_not_empty"),
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
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    session_id = Column(String(100))

    # Action details
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    changes = Column(JSONB)

    # Request context
    ip_address = Column(String(45))
    user_agent = Column(Text)

    # Relationships
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_created", "created_at"),
        Index("idx_audit_action_entity", "action", "entity_type"),
        Index("idx_audit_user_action", "user_id", "action"),
        CheckConstraint("LENGTH(action) >= 1", name="check_audit_action_not_empty"),
        CheckConstraint("LENGTH(entity_type) >= 1", name="check_audit_entity_type_not_empty"),
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
    value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    previous_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    change_percentage: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 4))

    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Foreign keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"))

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Relationships
    bot = relationship("Bot")
    strategy = relationship("Strategy")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_performance_type", "metric_type"),
        Index("idx_performance_entity", "entity_type", "entity_id"),
        Index("idx_performance_period", "period_start", "period_end"),
        Index("idx_performance_value", "value"),  # Value-based queries
        Index("idx_performance_created", "created_at"),  # Time-series optimization
        Index("idx_performance_bot_id", "bot_id"),
        Index("idx_performance_strategy_id", "strategy_id"),
        Index("idx_performance_type_period", "metric_type", "period_start", "period_end"),
        Index("idx_performance_bot_type", "bot_id", "metric_type"),
        CheckConstraint("value >= 0", name="check_performance_value_non_negative"),
        CheckConstraint("period_start < period_end", name="check_performance_period_order"),
        CheckConstraint("LENGTH(metric_type) >= 1", name="check_performance_metric_type_not_empty"),
        CheckConstraint("LENGTH(entity_type) >= 1", name="check_performance_entity_type_not_empty"),
        CheckConstraint("change_percentage >= -100", name="check_performance_change_percentage_min"),
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
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    total_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    available_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    locked_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # USD equivalent
    usd_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    exchange_rate: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    # Foreign keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))

    # Snapshot metadata
    snapshot_reason = Column(String(50))  # scheduled, triggered, manual

    # Relationships
    bot = relationship("Bot", back_populates="balance_snapshots")
    user = relationship("User", back_populates="balance_snapshots")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_balance_exchange", "exchange"),
        Index("idx_balance_currency", "currency"),
        Index("idx_balance_created", "created_at"),
        Index("idx_balance_composite", "exchange", "currency", "created_at"),
        Index("idx_balance_bot_id", "bot_id"),
        Index("idx_balance_user_id", "user_id"),
        Index("idx_balance_bot_currency", "bot_id", "currency"),
        Index("idx_balance_exchange_account", "exchange", "account_type"),
        CheckConstraint("total_balance >= 0", name="check_balance_total_non_negative"),
        CheckConstraint("available_balance >= 0", name="check_balance_available_non_negative"),
        CheckConstraint("locked_balance >= 0", name="check_balance_locked_non_negative"),
        CheckConstraint("total_balance = available_balance + locked_balance", name="check_balance_consistency"),
        CheckConstraint("account_type IN ('spot', 'futures', 'margin')", name="check_account_type"),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_balance_supported_exchange",
        ),
        CheckConstraint("usd_value IS NULL OR usd_value >= 0", name="check_balance_usd_value_non_negative"),
        CheckConstraint(
            "exchange_rate IS NULL OR exchange_rate > 0",
            name="check_balance_exchange_rate_positive",
        ),
        CheckConstraint(
            "snapshot_reason IN ('scheduled', 'triggered', 'manual')",
            name="check_balance_snapshot_reason",
        ),
    )

    def __repr__(self):
        return f"<BalanceSnapshot {self.exchange} {self.currency}: {self.total_balance}>"
