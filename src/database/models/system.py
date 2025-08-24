"""System models for alerts, performance metrics, and other system-wide data."""

import uuid

from sqlalchemy import DECIMAL, Column, DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .base import Base, TimestampMixin


class Alert(Base, TimestampMixin):
    """Alert model for system notifications."""

    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    # Alert status
    status = Column(String(20), default="ACTIVE")  # ACTIVE, ACKNOWLEDGED, RESOLVED
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))

    # Context data
    context = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_alerts_type", "alert_type"),
        Index("idx_alerts_severity", "severity"),
        Index("idx_alerts_status", "status"),
        Index("idx_alerts_created", "created_at"),
    )

    def __repr__(self):
        return f"<Alert {self.alert_type}: {self.title}>"


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
    value = Column(Float, nullable=False)
    previous_value = Column(Float)
    change_percentage = Column(Float)

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
    currency = Column(String(10), nullable=False)
    total_balance = Column(DECIMAL(20, 8), nullable=False)
    available_balance = Column(DECIMAL(20, 8), nullable=False)
    locked_balance = Column(DECIMAL(20, 8), default=0)

    # USD equivalent
    usd_value = Column(Float)
    exchange_rate = Column(Float)

    # Snapshot metadata
    snapshot_reason = Column(String(50))  # scheduled, triggered, manual

    # Indexes
    __table_args__ = (
        Index("idx_balance_exchange", "exchange"),
        Index("idx_balance_currency", "currency"),
        Index("idx_balance_created", "created_at"),
        Index("idx_balance_composite", "exchange", "currency", "created_at"),
    )

    def __repr__(self):
        return f"<BalanceSnapshot {self.exchange} {self.currency}: {self.total_balance}>"
