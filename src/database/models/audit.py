
"""
Audit trail models for capital management and execution operations.

This module provides comprehensive audit logging for all capital allocation
and trade execution operations, ensuring full traceability and compliance.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class CapitalAuditLog(Base):
    """Audit log model for capital management operations."""

    __tablename__ = "capital_audit_logs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Operation identification
    operation_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    operation_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # allocate, release, rebalance, update_utilization

    # Entity information
    strategy_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    bot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Operation details
    operation_description: Mapped[str] = mapped_column(Text, nullable=False)

    # Financial amounts
    amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    previous_amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    new_amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Allocation details
    previous_allocation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_allocation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    allocation_changes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Operation context
    operation_context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    risk_assessment: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Performance impact
    utilization_before: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimals for percentages
    utilization_after: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimals for percentages
    efficiency_impact: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimals for efficiency scores

    # Operation result
    operation_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="completed"
    )  # pending, completed, failed, cancelled
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Approval and authorization
    authorized_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    approval_level: Mapped[str] = mapped_column(
        String(20), nullable=False, default="automatic"
    )  # automatic, manual, override

    # Timing
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Audit metadata
    source_component: Mapped[str] = mapped_column(String(100), nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    audit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Compliance
    compliance_flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    regulatory_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    bot_instance = relationship(
        "BotInstance", foreign_keys=[bot_id], back_populates="capital_audit_logs"
    )

    # Constraints
    __table_args__ = (
        Index("idx_capital_audit_operation_type", "operation_type"),
        Index("idx_capital_audit_strategy", "strategy_id"),
        Index("idx_capital_audit_exchange", "exchange"),
        Index("idx_capital_audit_status", "operation_status"),
        Index("idx_capital_audit_timestamp", "created_at"),
        Index("idx_capital_audit_correlation", "correlation_id"),
        Index("idx_capital_audit_requested", "requested_at"),
        CheckConstraint("amount >= 0", name="check_capital_audit_amount_non_negative"),
        CheckConstraint(
            "previous_amount >= 0", name="check_capital_audit_previous_amount_non_negative"
        ),
        CheckConstraint("new_amount >= 0", name="check_capital_audit_new_amount_non_negative"),
        CheckConstraint(
            "operation_status IN ('pending', 'completed', 'failed', 'cancelled')",
            name="check_capital_audit_operation_status",
        ),
        CheckConstraint(
            "approval_level IN ('automatic', 'manual', 'override')",
            name="check_capital_audit_approval_level",
        ),
    )


class ExecutionAuditLog(Base):
    """Audit log model for execution operations."""

    __tablename__ = "execution_audit_logs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Execution identification
    execution_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    operation_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # order_placement, order_modification, order_cancellation, fill, settlement

    # Trade identification
    trade_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("trades.id", ondelete="SET NULL"), nullable=True, index=True
    )
    order_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    exchange_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Bot and strategy context
    bot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    strategy_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Market information
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Order details
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # buy, sell, long, short
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # market, limit, etc.

    # Quantities and pricing
    requested_quantity: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    executed_quantity: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    remaining_quantity: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    requested_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    executed_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    market_price_at_time: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Execution quality metrics
    slippage_bps: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # BPS with 2 decimal precision
    market_impact_bps: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # BPS with 2 decimal precision
    timing_cost_bps: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # BPS with 2 decimal precision
    total_cost_bps: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # BPS with 2 decimal precision

    # Algorithm and routing
    execution_algorithm: Mapped[str | None] = mapped_column(String(50), nullable=True)
    algorithm_parameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    routing_decision: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Risk and validation
    pre_trade_checks: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    post_trade_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    risk_violations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Operation result
    operation_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, completed, failed, cancelled, rejected
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Fees and costs
    total_fees: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), nullable=False, default=Decimal("0")
    )
    fee_currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    exchange_fees: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), nullable=False, default=Decimal("0")
    )
    network_fees: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), nullable=False, default=Decimal("0")
    )

    # Market conditions
    market_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    volatility_at_time: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # High precision for volatility
    volume_at_time: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    spread_bps: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # BPS with 2 decimal precision

    # Timing details
    signal_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    order_submission_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    first_fill_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    final_fill_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Latency metrics
    decision_to_submission_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submission_to_fill_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Audit metadata
    source_component: Mapped[str] = mapped_column(String(100), nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    audit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Compliance and regulatory
    compliance_flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    mifid_flags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # MiFID II compliance
    best_execution_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    bot_instance = relationship(
        "BotInstance", foreign_keys=[bot_id], back_populates="execution_audit_logs"
    )
    trade = relationship("Trade", foreign_keys=[trade_id], back_populates="execution_audit_logs")
    order = relationship("Order", foreign_keys=[order_id], back_populates="execution_audit_logs")

    # Constraints
    __table_args__ = (
        Index("idx_execution_audit_execution_id", "execution_id"),
        Index("idx_execution_audit_operation_type", "operation_type"),
        Index("idx_execution_audit_trade_id", "trade_id"),
        Index("idx_execution_audit_order_id", "order_id"),
        Index("idx_execution_audit_symbol", "symbol"),
        Index("idx_execution_audit_exchange", "exchange"),
        Index("idx_execution_audit_status", "operation_status"),
        Index("idx_execution_audit_timestamp", "created_at"),
        Index("idx_execution_audit_correlation", "correlation_id"),
        Index("idx_execution_audit_signal_time", "signal_timestamp"),
        Index("idx_execution_audit_bot_strategy", "bot_id", "strategy_name"),
        Index(
            "idx_execution_audit_timing", "signal_timestamp", "order_submission_timestamp"
        ),  # Execution latency analysis
        Index("idx_execution_audit_slippage", "slippage_bps"),  # Slippage analysis
        Index(
            "idx_execution_audit_algorithm", "execution_algorithm", "operation_status"
        ),  # Algorithm performance
        CheckConstraint(
            "requested_quantity > 0", name="check_execution_audit_requested_quantity_positive"
        ),
        CheckConstraint(
            "executed_quantity >= 0", name="check_execution_audit_executed_quantity_non_negative"
        ),
        CheckConstraint(
            "remaining_quantity >= 0", name="check_execution_audit_remaining_quantity_non_negative"
        ),
        CheckConstraint("total_fees >= 0", name="check_execution_audit_total_fees_non_negative"),
        CheckConstraint(
            "exchange_fees >= 0", name="check_execution_audit_exchange_fees_non_negative"
        ),
        CheckConstraint(
            "network_fees >= 0", name="check_execution_audit_network_fees_non_negative"
        ),
        CheckConstraint(
            "operation_status IN ('pending', 'completed', 'failed', 'cancelled', 'rejected')",
            name="check_execution_audit_operation_status",
        ),
        CheckConstraint(
            "side IN ('buy', 'sell', 'long', 'short')", name="check_execution_audit_side"
        ),
    )


class RiskAuditLog(Base):
    """Audit log model for risk management decisions and violations."""

    __tablename__ = "risk_audit_logs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Risk event identification
    risk_event_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # position_limit, drawdown, exposure, correlation, var

    # Associated entities
    bot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    strategy_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)

    # Risk assessment
    risk_level: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # low, medium, high, critical
    risk_score: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 4), nullable=False, default=Decimal("0.0")
    )  # Risk score with 4 decimal precision
    threshold_breached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Risk metrics
    current_value: Mapped[Decimal | None] = mapped_column(
        DECIMAL(20, 8), nullable=True
    )  # Financial values with 8 decimal precision
    threshold_value: Mapped[Decimal | None] = mapped_column(
        DECIMAL(20, 8), nullable=True
    )  # Financial values with 8 decimal precision
    threshold_percentage: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # Percentages with 4 decimal precision

    # Risk details
    risk_description: Mapped[str] = mapped_column(Text, nullable=False)
    risk_calculation: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contributing_factors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Actions taken
    action_taken: Mapped[str] = mapped_column(String(50), nullable=False, default="none")
    action_details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    preventive_measures: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Impact assessment
    potential_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    actual_impact: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    positions_affected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    resolution_method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolution_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Market context
    market_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    volatility_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Timing
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Audit metadata
    source_component: Mapped[str] = mapped_column(String(100), nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    audit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    bot_instance = relationship(
        "BotInstance", foreign_keys=[bot_id], back_populates="risk_audit_logs"
    )

    # Constraints
    __table_args__ = (
        Index("idx_risk_audit_event_type", "event_type"),
        Index("idx_risk_audit_risk_level", "risk_level"),
        Index("idx_risk_audit_threshold_breached", "threshold_breached"),
        Index("idx_risk_audit_resolved", "resolved"),
        Index("idx_risk_audit_detected", "detected_at"),
        Index("idx_risk_audit_bot_strategy", "bot_id", "strategy_name"),
        Index("idx_risk_audit_correlation", "correlation_id"),
        CheckConstraint("risk_score >= 0", name="check_risk_audit_risk_score_non_negative"),
        CheckConstraint(
            "positions_affected >= 0", name="check_risk_audit_positions_affected_non_negative"
        ),
        CheckConstraint("potential_loss >= 0", name="check_risk_audit_potential_loss_non_negative"),
        CheckConstraint("actual_impact >= 0", name="check_risk_audit_actual_impact_non_negative"),
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="check_risk_audit_risk_level",
        ),
    )


class PerformanceAuditLog(Base):
    """Audit log model for performance tracking and analysis."""

    __tablename__ = "performance_audit_logs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Performance event identification
    performance_event_id: Mapped[str] = mapped_column(
        String(100), nullable=False, unique=True, index=True
    )
    metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # pnl, sharpe, drawdown, win_rate, execution_quality

    # Time period
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)  # daily, weekly, monthly

    # Associated entities
    bot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("bot_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    strategy_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Performance metrics
    metric_value: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), nullable=False
    )  # High precision for financial metrics
    previous_value: Mapped[Decimal | None] = mapped_column(
        DECIMAL(20, 8), nullable=True
    )  # High precision for financial metrics
    change_percentage: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # Percentages with 4 decimal precision
    benchmark_value: Mapped[Decimal | None] = mapped_column(
        DECIMAL(20, 8), nullable=True
    )  # High precision for financial metrics

    # Detailed metrics
    performance_details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    contributing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Performance analysis
    performance_grade: Mapped[str] = mapped_column(
        String(10), nullable=False, default="C"
    )  # A+, A, B+, B, C+, C, D, F
    improvement_opportunities: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Attribution analysis
    alpha_contribution: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # High precision for alpha/beta
    beta_contribution: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # High precision for alpha/beta
    strategy_attribution: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Risk-adjusted metrics
    volatility: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # High precision for volatility
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimal precision for ratios
    sortino_ratio: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimal precision for ratios
    max_drawdown: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 4), nullable=True
    )  # 4 decimal precision for drawdown

    # Market context
    market_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    market_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Recommendations
    recommendations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    action_items: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Audit metadata
    calculation_method: Mapped[str] = mapped_column(String(100), nullable=False, default="standard")
    data_quality_score: Mapped[Decimal] = mapped_column(
        DECIMAL(6, 4), nullable=False, default=Decimal("1.0")
    )  # Quality score 0-1 with 4 decimals
    confidence_level: Mapped[Decimal] = mapped_column(
        DECIMAL(6, 4), nullable=False, default=Decimal("0.95")
    )  # Confidence level 0-1 with 4 decimals

    # Timestamps
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    bot_instance = relationship(
        "BotInstance", foreign_keys=[bot_id], back_populates="performance_audit_logs"
    )

    # Constraints
    __table_args__ = (
        Index("idx_performance_audit_metric_type", "metric_type"),
        Index("idx_performance_audit_period", "period_start", "period_end"),
        Index("idx_performance_audit_bot_strategy", "bot_id", "strategy_name"),
        Index("idx_performance_audit_grade", "performance_grade"),
        Index("idx_performance_audit_calculated", "calculated_at"),
        UniqueConstraint(
            "bot_id",
            "strategy_name",
            "metric_type",
            "period_start",
            "period_end",
            name="unique_performance_audit_record",
        ),
        CheckConstraint(
            "contributing_trades >= 0",
            name="check_performance_audit_contributing_trades_non_negative",
        ),
        CheckConstraint(
            "data_quality_score >= 0 AND data_quality_score <= 1",
            name="check_performance_audit_data_quality_score_range",
        ),
        CheckConstraint(
            "confidence_level >= 0 AND confidence_level <= 1",
            name="check_performance_audit_confidence_level_range",
        ),
        CheckConstraint("period_start < period_end", name="check_performance_audit_period_order"),
        CheckConstraint(
            "performance_grade IN ('A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F')",
            name="check_performance_audit_grade",
        ),
        CheckConstraint(
            "period_type IN ('daily', 'weekly', 'monthly')",
            name="check_performance_audit_period_type",
        ),
    )
