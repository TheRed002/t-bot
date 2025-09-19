"""Bot and strategy database models."""

import uuid
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import (
    AuditMixin,
    Base,
    MetadataMixin,
    SoftDeleteMixin,
    TimestampMixin,
)


class Bot(Base, AuditMixin, MetadataMixin, SoftDeleteMixin):
    """Bot model."""

    __tablename__ = "bots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(String(1000))
    status = Column(
        String(20), nullable=False, default="stopped"
    )  # initializing, ready, running, paused, stopping, stopped, error, maintenance

    exchange = Column(String(50), nullable=False)
    test_mode = Column(Boolean, default=False)
    paper_trading = Column(Boolean, default=False)

    # Capital allocation
    allocated_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    current_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Configuration
    config = Column(JSON, default={})

    # Relationships
    instances = relationship("BotInstance", back_populates="bot", cascade="all, delete-orphan")
    strategies = relationship("Strategy", back_populates="bot", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="bot")
    positions = relationship("Position", back_populates="bot")
    trades = relationship("Trade", back_populates="bot")
    logs = relationship("BotLog", back_populates="bot", cascade="all, delete-orphan")
    data_pipeline_records = relationship(
        "DataPipelineRecord", back_populates="bot", cascade="all, delete-orphan"
    )
    balance_snapshots = relationship("BalanceSnapshot", back_populates="bot")

    # Analytics relationships
    portfolio_metrics = relationship(
        "AnalyticsPortfolioMetrics", back_populates="bot", cascade="all, delete-orphan"
    )
    position_metrics = relationship(
        "AnalyticsPositionMetrics", back_populates="bot", cascade="all, delete-orphan"
    )
    risk_metrics = relationship(
        "AnalyticsRiskMetrics", back_populates="bot", cascade="all, delete-orphan"
    )
    strategy_metrics = relationship(
        "AnalyticsStrategyMetrics", back_populates="bot", cascade="all, delete-orphan"
    )
    operational_metrics = relationship(
        "AnalyticsOperationalMetrics", back_populates="bot", cascade="all, delete-orphan"
    )

    # Capital management relationships
    fund_flows = relationship("FundFlowDB", back_populates="bot", cascade="all, delete-orphan")

    # State management relationships
    state_snapshots = relationship("StateSnapshot", back_populates="bot")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_bots_status", "status"),
        Index("idx_bots_exchange", "exchange"),
        Index("idx_bots_created_at", "created_at"),
        CheckConstraint("allocated_capital >= 0", name="check_allocated_capital_non_negative"),
        CheckConstraint("current_balance >= 0", name="check_current_balance_non_negative"),
        CheckConstraint("total_trades >= 0", name="check_total_trades_non_negative"),
        CheckConstraint("winning_trades >= 0", name="check_winning_trades_non_negative"),
        CheckConstraint("losing_trades >= 0", name="check_losing_trades_non_negative"),
        CheckConstraint(
            "winning_trades + losing_trades <= total_trades", name="check_trades_consistency"
        ),
        CheckConstraint(
            "status IN ('initializing', 'ready', 'running', 'paused', 'stopping', 'stopped', 'error', 'maintenance')",
            name="check_bot_status",
        ),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_bot_supported_exchange",
        ),
    )

    def __repr__(self):
        return f"<Bot {self.id}: {self.name} ({self.status})>"

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return bool(self.status == "running")

    def win_rate(self) -> Decimal:
        """Calculate win rate."""
        if int(self.total_trades) == 0:
            return Decimal("0.0")
        return (Decimal(str(self.winning_trades)) / Decimal(str(self.total_trades))) * Decimal(
            "100"
        )

    def average_pnl(self) -> Decimal:
        """Calculate average P&L per trade."""
        if int(self.total_trades) == 0:
            return Decimal("0.0")
        return self.total_pnl / Decimal(str(self.total_trades))


class Strategy(Base, AuditMixin, MetadataMixin):
    """Strategy model."""

    __tablename__ = "strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # market_making, arbitrage, trend_following, etc.
    status = Column(
        String(20), nullable=False, default="inactive"
    )  # inactive, active, paused, starting, stopping, stopped, error

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)

    # Strategy parameters
    params = Column(JSON, default={})

    # Risk parameters
    max_position_size: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))
    risk_per_trade: Mapped[Decimal] = mapped_column(DECIMAL(6, 4))
    stop_loss_percentage: Mapped[Decimal | None] = mapped_column(DECIMAL(6, 4))
    take_profit_percentage: Mapped[Decimal | None] = mapped_column(DECIMAL(6, 4))

    # Performance tracking
    total_signals = Column(Integer, default=0)
    executed_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)

    # Relationships
    bot = relationship("Bot", back_populates="strategies")
    orders = relationship("Order", back_populates="strategy")
    positions = relationship("Position", back_populates="strategy")
    trades = relationship("Trade", back_populates="strategy")
    signals = relationship("Signal", back_populates="strategy", cascade="all, delete-orphan")
    capital_allocations = relationship(
        "CapitalAllocationDB", back_populates="strategy", cascade="all, delete-orphan"
    )
    fund_flows = relationship("FundFlowDB", back_populates="strategy", cascade="all, delete-orphan")
    optimization_runs = relationship("OptimizationRun", back_populates="strategy", cascade="all, delete-orphan")

    # Analytics relationships
    strategy_metrics = relationship(
        "AnalyticsStrategyMetrics", back_populates="strategy", cascade="all, delete-orphan"
    )

    # Backtesting relationships
    backtest_runs = relationship(
        "BacktestRun", back_populates="strategy", cascade="all, delete-orphan"
    )

    # State management relationships
    state_snapshots = relationship("StateSnapshot", back_populates="strategy")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_strategies_bot_id", "bot_id"),
        Index("idx_strategies_status", "status"),
        Index("idx_strategies_type", "type"),
        CheckConstraint(
            "risk_per_trade >= 0 AND risk_per_trade <= 1", name="check_risk_per_trade_range"
        ),
        CheckConstraint("max_position_size > 0", name="check_max_position_size_positive"),
        CheckConstraint(
            "stop_loss_percentage >= 0 AND stop_loss_percentage <= 1", name="check_stop_loss_range"
        ),
        CheckConstraint("take_profit_percentage >= 0", name="check_take_profit_non_negative"),
        CheckConstraint("total_signals >= 0", name="check_total_signals_non_negative"),
        CheckConstraint("executed_signals >= 0", name="check_executed_signals_non_negative"),
        CheckConstraint("successful_signals >= 0", name="check_successful_signals_non_negative"),
        CheckConstraint(
            "executed_signals <= total_signals", name="check_signals_execution_consistency"
        ),
        CheckConstraint(
            "successful_signals <= executed_signals", name="check_signals_success_consistency"
        ),
        CheckConstraint(
            "status IN ('inactive', 'starting', 'active', 'paused', 'stopping', 'stopped', 'error')",
            name="check_strategy_status",
        ),
        CheckConstraint(
            "stop_loss_percentage IS NULL OR take_profit_percentage IS NULL OR stop_loss_percentage < take_profit_percentage",
            name="check_stop_loss_less_than_take_profit",
        ),
        CheckConstraint(
            "max_position_size <= 10000000",  # Reasonable upper limit
            name="check_max_position_size_reasonable",
        ),
        CheckConstraint(
            "type IN ('mean_reversion', 'momentum', 'arbitrage', 'market_making', 'trend_following', 'pairs_trading', 'statistical_arbitrage', 'breakout', 'custom')",
            name="check_strategy_type",
        ),
    )

    def __repr__(self):
        return f"<Strategy {self.id}: {self.name} ({self.type})>"

    def is_active(self) -> bool:
        """Check if strategy is active."""
        return bool(self.status == "active")

    def signal_success_rate(self) -> Decimal:
        """Calculate signal success rate."""
        if int(self.executed_signals) == 0:
            return Decimal("0.0")
        return (
            Decimal(str(self.successful_signals)) / Decimal(str(self.executed_signals))
        ) * Decimal("100")


class Signal(Base, TimestampMixin, MetadataMixin):
    """Trading signal model."""

    __tablename__ = "signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )

    symbol = Column(String(50), nullable=False)
    direction = Column(String(20), nullable=False)  # BUY, SELL, HOLD (matches SignalDirection enum)
    strength: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Signal strength 0-1
    source = Column(String(100), nullable=False)  # Signal source (matches core types)

    # Signal details
    price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    quantity: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    reason = Column(Text)

    # ML prediction link (if signal is generated by ML model)
    ml_prediction_id = Column(
        BigInteger, ForeignKey("ml_predictions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Execution status
    executed = Column(Boolean, default=False)
    execution_time: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 3)
    )  # Time to execute in seconds
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"))

    # Outcome
    outcome = Column(String(20))  # SUCCESS, FAILURE, PARTIAL, EXPIRED
    pnl: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    order = relationship("Order", back_populates="signal")
    ml_prediction = relationship("MLPrediction", back_populates="signals")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_signals_strategy_id", "strategy_id"),
        Index("idx_signals_symbol", "symbol"),
        Index("idx_signals_direction", "direction"),  # New index for direction field
        Index("idx_signals_source", "source"),  # New index for source field
        Index("idx_signals_created_at", "created_at"),
        Index("idx_signals_executed", "executed"),
        Index("idx_signals_outcome", "outcome"),
        Index("idx_signals_strength", "strength"),
        Index("idx_signals_ml_prediction", "ml_prediction_id"),  # ML prediction performance
        CheckConstraint("strength >= 0 AND strength <= 1", name="check_signal_strength_range"),
        CheckConstraint("price IS NULL OR price > 0", name="check_signal_price_positive"),
        CheckConstraint("quantity IS NULL OR quantity > 0", name="check_signal_quantity_positive"),
        CheckConstraint(
            "execution_time IS NULL OR execution_time >= 0",
            name="check_execution_time_non_negative",
        ),
        CheckConstraint("direction IN ('BUY', 'SELL', 'HOLD', 'CLOSE')", name="check_signal_direction"),
        CheckConstraint(
            "outcome IN ('SUCCESS', 'FAILURE', 'PARTIAL', 'EXPIRED')", name="check_signal_outcome"
        ),
    )

    def __repr__(self):
        return f"<Signal {self.id}: {self.direction} {self.symbol}>"

    def is_executed(self) -> bool:
        """Check if signal was executed."""
        return bool(self.executed)

    def is_successful(self) -> bool:
        """Check if signal was successful."""
        return bool(self.outcome == "SUCCESS")


class BotLog(Base, TimestampMixin):
    """Bot activity log model for error handling and audit trail."""

    __tablename__ = "bot_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)

    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category = Column(String(50))  # strategy, execution, risk, system, etc.
    message = Column(Text, nullable=False)

    # Error handling specific fields
    error_code = Column(String(50))
    error_type = Column(String(100))
    stack_trace = Column(Text)

    # Additional context
    context = Column(JSON, default={})

    # Correlation for tracking related events
    correlation_id = Column(String(100), index=True)
    request_id = Column(String(100), index=True)

    # Source information
    component = Column(String(100))
    function_name = Column(String(100))

    # Severity and priority
    severity_score = Column(Integer, default=0)  # 0-100 scale
    requires_attention = Column(Boolean, default=False)

    # Resolution tracking
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)

    # Relationships
    bot = relationship("Bot", back_populates="logs")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_bot_logs_bot_id", "bot_id"),
        Index("idx_bot_logs_level", "level"),
        Index("idx_bot_logs_created_at", "created_at"),
        Index("idx_bot_logs_category", "category"),
        Index("idx_bot_logs_error_code", "error_code"),
        Index("idx_bot_logs_correlation", "correlation_id"),
        Index("idx_bot_logs_resolved", "resolved"),
        Index("idx_bot_logs_attention", "requires_attention"),
        Index("idx_bot_logs_severity", "severity_score"),
        Index("idx_bot_logs_composite", "bot_id", "level", "created_at"),
        CheckConstraint(
            "level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", name="check_log_level"
        ),
        CheckConstraint(
            "severity_score >= 0 AND severity_score <= 100", name="check_severity_score_range"
        ),
        CheckConstraint(
            "resolved_at IS NULL OR resolved = true", name="check_resolved_consistency"
        ),
        UniqueConstraint(
            "bot_id", "correlation_id", "created_at", name="unique_bot_log_correlation"
        ),
    )

    def __repr__(self):
        return f"<BotLog {self.id}: [{self.level}] {self.message[:50]}>"
