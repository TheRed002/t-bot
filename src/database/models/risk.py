"""Risk management database models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.types import (
    AlertSeverity,
    CircuitBreakerStatus,
    CircuitBreakerType,
    PositionSizeMethod,
)

from .base import Base, TimestampMixin


class RiskConfiguration(Base, TimestampMixin):
    """Risk configuration storage for strategies and bots."""

    __tablename__ = "risk_configurations"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Configuration identification
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Position limits
    max_position_size: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    max_position_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    max_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    max_leverage: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 4), nullable=False, default=Decimal("1.0")
    )
    min_position_size: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), nullable=False, default=Decimal("0")
    )

    # Risk limits
    max_portfolio_risk: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    max_correlation: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, default=Decimal("0.8")
    )
    max_drawdown: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, default=Decimal("0.2")
    )
    max_daily_loss: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    max_weekly_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    max_monthly_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # VaR limits
    max_var_95: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    max_var_99: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    max_expected_shortfall: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Concentration limits
    max_concentration: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, default=Decimal("0.25")
    )
    min_liquidity_ratio: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 6), nullable=False, default=Decimal("0.1")
    )

    # Performance limits
    min_sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    min_sortino_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)

    # Trading limits
    max_trades_per_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_trades_per_hour: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_time_between_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)  # seconds

    # Configuration flags
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    strict_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    emergency_stop_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Additional configuration
    position_sizing_method: Mapped[PositionSizeMethod] = mapped_column(
        SQLEnum(PositionSizeMethod), nullable=False, default=PositionSizeMethod.FIXED_PERCENTAGE
    )
    risk_assessment_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="standard"
    )
    custom_parameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Timestamps handled by TimestampMixin

    # Relationships
    bot_instance = relationship("BotInstance", back_populates="risk_configurations")
    circuit_breaker_configs = relationship(
        "CircuitBreakerConfig", back_populates="risk_config", cascade="all, delete-orphan"
    )
    risk_violations = relationship(
        "RiskViolation", back_populates="risk_config", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        Index("idx_risk_config_bot_id", "bot_id"),
        Index("idx_risk_config_strategy", "strategy_name"),
        Index("idx_risk_config_symbol", "symbol"),
        Index("idx_risk_config_enabled", "enabled"),
        Index("idx_risk_config_bot_strategy", "bot_id", "strategy_name"),
        Index("idx_risk_config_exchange", "exchange"),
        Index("idx_risk_config_bot_exchange", "bot_id", "exchange"),
        UniqueConstraint(
            "bot_id",
            "strategy_name",
            "symbol",
            "exchange",
            name="uq_risk_config_bot_strategy_symbol_exchange",
        ),
        CheckConstraint(
            "max_position_size > 0", name="check_risk_config_max_position_size_positive"
        ),
        CheckConstraint(
            "max_position_value IS NULL OR max_position_value > 0",
            name="check_risk_config_max_position_value_positive",
        ),
        CheckConstraint("max_positions > 0", name="check_risk_config_max_positions_positive"),
        CheckConstraint("max_leverage > 0", name="check_risk_config_max_leverage_positive"),
        CheckConstraint(
            "min_position_size >= 0", name="check_risk_config_min_position_size_non_negative"
        ),
        CheckConstraint(
            "max_portfolio_risk > 0 AND max_portfolio_risk <= 1",
            name="check_risk_config_max_portfolio_risk_range",
        ),
        CheckConstraint(
            "max_correlation >= -1 AND max_correlation <= 1",
            name="check_risk_config_max_correlation_range",
        ),
        CheckConstraint(
            "max_drawdown >= 0 AND max_drawdown <= 1", name="check_risk_config_max_drawdown_range"
        ),
        CheckConstraint("max_daily_loss > 0", name="check_risk_config_max_daily_loss_positive"),
        CheckConstraint(
            "max_weekly_loss IS NULL OR max_weekly_loss > 0",
            name="check_risk_config_max_weekly_loss_positive",
        ),
        CheckConstraint(
            "max_monthly_loss IS NULL OR max_monthly_loss > 0",
            name="check_risk_config_max_monthly_loss_positive",
        ),
        CheckConstraint(
            "max_concentration >= 0 AND max_concentration <= 1",
            name="check_risk_config_max_concentration_range",
        ),
        CheckConstraint(
            "min_liquidity_ratio >= 0 AND min_liquidity_ratio <= 1",
            name="check_risk_config_min_liquidity_ratio_range",
        ),
        CheckConstraint(
            "max_trades_per_day IS NULL OR max_trades_per_day > 0",
            name="check_risk_config_max_trades_per_day_positive",
        ),
        CheckConstraint(
            "max_trades_per_hour IS NULL OR max_trades_per_hour > 0",
            name="check_risk_config_max_trades_per_hour_positive",
        ),
        CheckConstraint(
            "min_time_between_trades IS NULL OR min_time_between_trades >= 0",
            name="check_risk_config_min_time_between_trades_non_negative",
        ),
        CheckConstraint(
            "risk_assessment_method IN ('standard', 'monte_carlo', 'historical', 'parametric', 'custom')",
            name="check_risk_config_risk_assessment_method",
        ),
    )


class CircuitBreakerConfig(Base, TimestampMixin):
    """Circuit breaker configuration storage."""

    __tablename__ = "circuit_breaker_configs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Configuration identification
    risk_config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("risk_configurations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    breaker_type: Mapped[CircuitBreakerType] = mapped_column(SQLEnum(CircuitBreakerType), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Trigger configuration
    threshold_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    threshold_percentage: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    lookback_period: Mapped[int] = mapped_column(Integer, nullable=False, default=86400)  # seconds
    cooldown_period: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)  # seconds

    # Status and control
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[CircuitBreakerStatus] = mapped_column(
        SQLEnum(CircuitBreakerStatus), nullable=False, default=CircuitBreakerStatus.ACTIVE
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Action configuration
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action_parameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    auto_reset: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Trigger history
    trigger_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Additional configuration
    extra_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Timestamps handled by TimestampMixin

    # Relationships
    risk_config = relationship("RiskConfiguration", back_populates="circuit_breaker_configs")
    trigger_events = relationship(
        "CircuitBreakerEvent", back_populates="config", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        Index("idx_circuit_breaker_config_risk_config", "risk_config_id"),
        Index("idx_circuit_breaker_config_type", "breaker_type"),
        Index("idx_circuit_breaker_config_status", "status"),
        Index("idx_circuit_breaker_config_enabled", "enabled"),
        Index("idx_circuit_breaker_config_priority", "priority"),
        UniqueConstraint(
            "risk_config_id", "breaker_type", "name", name="uq_circuit_breaker_config_unique"
        ),
        CheckConstraint(
            "threshold_value > 0", name="check_circuit_breaker_config_threshold_positive"
        ),
        CheckConstraint(
            "threshold_percentage IS NULL OR (threshold_percentage >= 0 AND threshold_percentage <= 1)",
            name="check_circuit_breaker_config_threshold_pct_range",
        ),
        CheckConstraint(
            "lookback_period > 0", name="check_circuit_breaker_config_lookback_positive"
        ),
        CheckConstraint(
            "cooldown_period >= 0", name="check_circuit_breaker_config_cooldown_non_negative"
        ),
        CheckConstraint("priority > 0", name="check_circuit_breaker_config_priority_positive"),
        CheckConstraint(
            "trigger_count >= 0", name="check_circuit_breaker_config_trigger_count_non_negative"
        ),
        CheckConstraint(
            "action_type IN ('close_all_positions', 'block_new_orders', 'cancel_pending_orders', 'reduce_position_sizes', 'switch_to_safe_mode', 'manual_override')",
            name="check_circuit_breaker_config_action_type",
        ),
    )


class CircuitBreakerEvent(Base, TimestampMixin):
    """Circuit breaker trigger event storage."""

    __tablename__ = "circuit_breaker_events"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Event identification
    config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("circuit_breaker_configs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    breaker_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Trigger details
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    trigger_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    threshold_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(
        SQLEnum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM
    )

    # Status and resolution
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="triggered"
    )  # triggered, active, resolved, cancelled
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_method: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Impact assessment
    affected_symbols: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    affected_strategies: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    positions_affected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    orders_cancelled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Action taken
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action_success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    action_details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Recovery
    recovery_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    recovery_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Context and analysis
    market_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    risk_analysis: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Timestamps handled by TimestampMixin

    # Relationships
    config = relationship("CircuitBreakerConfig", back_populates="trigger_events")

    # Constraints
    __table_args__ = (
        Index("idx_circuit_breaker_event_config", "config_id"),
        Index("idx_circuit_breaker_event_type", "breaker_type"),
        Index("idx_circuit_breaker_event_triggered", "triggered_at"),
        Index("idx_circuit_breaker_event_status", "status"),
        Index("idx_circuit_breaker_event_severity", "severity"),
        Index("idx_circuit_breaker_event_resolved", "resolved_at"),
        CheckConstraint(
            "trigger_value >= 0", name="check_circuit_breaker_event_trigger_value_non_negative"
        ),
        CheckConstraint(
            "threshold_value > 0", name="check_circuit_breaker_event_threshold_positive"
        ),
        CheckConstraint(
            "positions_affected >= 0",
            name="check_circuit_breaker_event_positions_affected_non_negative",
        ),
        CheckConstraint(
            "orders_cancelled >= 0",
            name="check_circuit_breaker_event_orders_cancelled_non_negative",
        ),
        CheckConstraint(
            "status IN ('triggered', 'active', 'resolved', 'cancelled')",
            name="check_circuit_breaker_event_status",
        ),
    )


class RiskViolation(Base, TimestampMixin):
    """Risk violation event storage."""

    __tablename__ = "risk_violations"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Violation identification
    violation_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    bot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    risk_config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("risk_configurations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Violation details
    violation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    rule_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(
        SQLEnum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM
    )

    # Context
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    strategy_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    order_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    position_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Violation metrics
    current_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    limit_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    violation_amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    violation_percentage: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    resolution_action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Additional data
    violation_data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    market_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    extra_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Event timestamp (separate from created_at)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # created_at/updated_at handled by TimestampMixin

    # Relationships
    bot_instance = relationship("BotInstance", back_populates="risk_violations")
    risk_config = relationship("RiskConfiguration", back_populates="risk_violations")

    # Constraints
    __table_args__ = (
        Index("idx_risk_violation_bot_id", "bot_id"),
        Index("idx_risk_violation_type", "violation_type"),
        Index("idx_risk_violation_severity", "severity"),
        Index("idx_risk_violation_resolved", "resolved"),
        Index("idx_risk_violation_detected", "detected_at"),
        Index("idx_risk_violation_symbol", "symbol"),
        Index("idx_risk_violation_strategy", "strategy_name"),
        Index("idx_risk_violation_bot_type", "bot_id", "violation_type"),
        CheckConstraint(
            "violation_type IN ('position_limit', 'exposure_limit', 'drawdown_limit', 'correlation_limit', 'var_limit', 'concentration_limit', 'leverage_limit', 'liquidity_limit', 'trading_limit', 'custom')",
            name="check_risk_violation_type",
        ),
        CheckConstraint(
            "current_value IS NULL OR current_value >= 0",
            name="check_risk_violation_current_value_non_negative",
        ),
        CheckConstraint(
            "limit_value IS NULL OR limit_value >= 0",
            name="check_risk_violation_limit_value_non_negative",
        ),
        CheckConstraint(
            "violation_amount IS NULL OR violation_amount >= 0",
            name="check_risk_violation_amount_non_negative",
        ),
        CheckConstraint(
            "violation_percentage IS NULL OR violation_percentage >= 0",
            name="check_risk_violation_percentage_non_negative",
        ),
    )
