"""Backtesting database models."""

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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import Base, TimestampMixin


class BacktestRun(Base, TimestampMixin):
    """Backtest run configuration and metadata storage."""

    __tablename__ = "backtest_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic run information
    name = Column(String(200), nullable=False, index=True)
    status = Column(
        String(20), nullable=False, index=True
    )  # 'pending', 'running', 'completed', 'failed', 'cancelled'

    # Foreign Keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=True
    )

    # Configuration
    symbols = Column(JSONB, nullable=False)  # List of trading symbols
    exchange = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, index=True)

    # Date range
    start_date = Column(DateTime(timezone=True), nullable=False, index=True)
    end_date = Column(DateTime(timezone=True), nullable=False, index=True)

    # Capital and trading parameters
    initial_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    commission_rate: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False, default=0)
    slippage_rate: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False, default=0)
    enable_shorting = Column(Boolean, nullable=False, default=False)
    max_open_positions = Column(Integer, nullable=False, default=5)

    # Strategy and risk configuration (JSON)
    strategy_config = Column(JSONB, nullable=False)
    risk_config = Column(JSONB, nullable=True)

    # Execution tracking
    execution_start_time = Column(DateTime(timezone=True), nullable=True)
    execution_end_time = Column(DateTime(timezone=True), nullable=True)
    execution_duration_seconds = Column(Integer, nullable=True)

    # Progress tracking
    progress_percentage = Column(Integer, nullable=False, default=0)
    current_stage = Column(String(50), nullable=True)

    # Results summary (calculated fields)
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    total_return_pct: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)

    # Metadata
    error_message = Column(Text, nullable=True)
    additional_metadata = Column(JSONB, nullable=True)

    # Relationships
    user = relationship("User", back_populates="backtest_runs")
    strategy = relationship("Strategy", back_populates="backtest_runs")
    optimization_runs = relationship("OptimizationRun", back_populates="backtest_run", cascade="all, delete-orphan")
    backtest_results = relationship(
        "BacktestResult", back_populates="backtest_run", cascade="all, delete-orphan"
    )
    backtest_trades = relationship(
        "BacktestTrade", back_populates="backtest_run", cascade="all, delete-orphan"
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_backtest_runs_user_id", "user_id"),
        Index("idx_backtest_runs_strategy_id", "strategy_id"),
        Index("idx_backtest_runs_status", "status"),
        Index("idx_backtest_runs_exchange", "exchange"),
        Index("idx_backtest_runs_timeframe", "timeframe"),
        Index("idx_backtest_runs_start_date", "start_date"),
        Index("idx_backtest_runs_end_date", "end_date"),
        Index("idx_backtest_runs_user_status", "user_id", "status"),
        Index("idx_backtest_runs_execution_times", "execution_start_time", "execution_end_time"),
        # Business constraints
        CheckConstraint("end_date > start_date", name="check_backtest_date_range"),
        CheckConstraint("initial_capital > 0", name="check_backtest_initial_capital_positive"),
        CheckConstraint(
            "commission_rate >= 0 AND commission_rate <= 1",
            name="check_backtest_commission_rate_range",
        ),
        CheckConstraint(
            "slippage_rate >= 0 AND slippage_rate <= 1", name="check_backtest_slippage_rate_range"
        ),
        CheckConstraint(
            "max_open_positions > 0 AND max_open_positions <= 100",
            name="check_backtest_max_positions_range",
        ),
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name="check_backtest_progress_range",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="check_backtest_status",
        ),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_backtest_exchange",
        ),
        CheckConstraint(
            "total_trades IS NULL OR total_trades >= 0",
            name="check_backtest_total_trades_non_negative",
        ),
        CheckConstraint(
            "winning_trades IS NULL OR winning_trades >= 0",
            name="check_backtest_winning_trades_non_negative",
        ),
        CheckConstraint(
            "losing_trades IS NULL OR losing_trades >= 0",
            name="check_backtest_losing_trades_non_negative",
        ),
        CheckConstraint(
            "(winning_trades IS NULL AND losing_trades IS NULL AND total_trades IS NULL) OR "
            "(winning_trades + losing_trades <= total_trades)",
            name="check_backtest_trades_consistency",
        ),
        UniqueConstraint("user_id", "name", name="uq_backtest_runs_user_name"),
    )


class BacktestResult(Base, TimestampMixin):
    """Comprehensive backtest results and performance metrics."""

    __tablename__ = "backtest_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    backtest_run_id = Column(
        UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )

    # Performance metrics
    total_return_pct: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    annual_return_pct: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    sortino_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    max_drawdown_pct: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), nullable=False)
    win_rate_pct: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False)

    # Trade statistics
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    avg_win_amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    avg_loss_amount: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    profit_factor: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)

    # Risk metrics
    volatility_pct: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    value_at_risk_95_pct: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    conditional_var_95_pct: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)

    # Capital metrics
    initial_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    final_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    peak_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    lowest_capital: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # Time-based metrics
    total_time_in_market_hours: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )
    avg_trade_duration_hours: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 2), nullable=True)
    longest_winning_streak = Column(Integer, nullable=True)
    longest_losing_streak = Column(Integer, nullable=True)

    # Advanced analytics results (stored as JSON)
    equity_curve = Column(JSONB, nullable=True)  # Array of {timestamp, equity} points
    daily_returns = Column(JSONB, nullable=True)  # Array of daily return values
    monte_carlo_results = Column(JSONB, nullable=True)  # Monte Carlo analysis results
    walk_forward_results = Column(JSONB, nullable=True)  # Walk-forward analysis results
    performance_attribution = Column(JSONB, nullable=True)  # Performance attribution data

    # Metadata
    analysis_metadata = Column(JSONB, nullable=True)

    # Relationships
    backtest_run = relationship("BacktestRun", back_populates="backtest_results")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_backtest_results_run_id", "backtest_run_id"),
        Index("idx_backtest_results_total_return", "total_return_pct"),
        Index("idx_backtest_results_sharpe_ratio", "sharpe_ratio"),
        Index("idx_backtest_results_max_drawdown", "max_drawdown_pct"),
        Index("idx_backtest_results_win_rate", "win_rate_pct"),
        Index("idx_backtest_results_profit_factor", "profit_factor"),
        # Business constraints
        CheckConstraint(
            "total_trades >= 0", name="check_backtest_result_total_trades_non_negative"
        ),
        CheckConstraint(
            "winning_trades >= 0", name="check_backtest_result_winning_trades_non_negative"
        ),
        CheckConstraint(
            "losing_trades >= 0", name="check_backtest_result_losing_trades_non_negative"
        ),
        CheckConstraint(
            "winning_trades + losing_trades <= total_trades",
            name="check_backtest_result_trades_consistency",
        ),
        CheckConstraint(
            "win_rate_pct >= 0 AND win_rate_pct <= 1", name="check_backtest_result_win_rate_range"
        ),
        CheckConstraint(
            "max_drawdown_pct >= 0 AND max_drawdown_pct <= 1",
            name="check_backtest_result_drawdown_range",
        ),
        CheckConstraint(
            "initial_capital > 0", name="check_backtest_result_initial_capital_positive"
        ),
        CheckConstraint(
            "final_capital >= 0", name="check_backtest_result_final_capital_non_negative"
        ),
        CheckConstraint(
            "peak_capital >= initial_capital", name="check_backtest_result_peak_capital_valid"
        ),
        CheckConstraint(
            "lowest_capital >= 0", name="check_backtest_result_lowest_capital_non_negative"
        ),
        CheckConstraint(
            "lowest_capital <= initial_capital", name="check_backtest_result_lowest_capital_valid"
        ),
        CheckConstraint(
            "profit_factor IS NULL OR profit_factor >= 0",
            name="check_backtest_result_profit_factor_non_negative",
        ),
        CheckConstraint(
            "avg_win_amount IS NULL OR avg_win_amount > 0",
            name="check_backtest_result_avg_win_positive",
        ),
        CheckConstraint(
            "avg_loss_amount IS NULL OR avg_loss_amount < 0",
            name="check_backtest_result_avg_loss_negative",
        ),
        CheckConstraint(
            "total_time_in_market_hours IS NULL OR total_time_in_market_hours >= 0",
            name="check_backtest_result_market_time_non_negative",
        ),
        CheckConstraint(
            "avg_trade_duration_hours IS NULL OR avg_trade_duration_hours > 0",
            name="check_backtest_result_avg_duration_positive",
        ),
        CheckConstraint(
            "longest_winning_streak IS NULL OR longest_winning_streak >= 0",
            name="check_backtest_result_win_streak_non_negative",
        ),
        CheckConstraint(
            "longest_losing_streak IS NULL OR longest_losing_streak >= 0",
            name="check_backtest_result_lose_streak_non_negative",
        ),
        UniqueConstraint("backtest_run_id", name="uq_backtest_results_run_id"),
    )


class BacktestTrade(Base, TimestampMixin):
    """Individual trade records from backtest simulations."""

    __tablename__ = "backtest_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    backtest_run_id = Column(
        UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )

    # Trade identification
    trade_sequence = Column(Integer, nullable=False)  # Sequential trade number within backtest
    symbol = Column(String(20), nullable=False, index=True)

    # Entry details
    entry_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    entry_signal_strength: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 4), nullable=True)

    # Exit details
    exit_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    exit_reason = Column(
        String(50), nullable=False
    )  # 'signal', 'stop_loss', 'take_profit', 'max_duration', 'backtest_end'

    # Trade characteristics
    side = Column(String(10), nullable=False, index=True)  # 'LONG', 'SHORT'
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    position_size_usd: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # Performance metrics
    pnl_usd: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    pnl_percentage: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    commission_paid: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    slippage_cost: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)

    # Duration metrics
    duration_hours: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    bars_held = Column(Integer, nullable=True)  # Number of candles/bars held

    # Risk metrics
    max_adverse_excursion_pct: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # Max loss during trade
    max_favorable_excursion_pct: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 6), nullable=True
    )  # Max profit during trade

    # Strategy context
    strategy_signal_data = Column(JSONB, nullable=True)  # Original signal data
    risk_metrics = Column(JSONB, nullable=True)  # Risk calculations at time of trade

    # Execution details
    execution_algorithm = Column(String(50), nullable=True)
    execution_metadata = Column(JSONB, nullable=True)

    # Relationships
    backtest_run = relationship("BacktestRun", back_populates="backtest_trades")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_backtest_trades_run_id", "backtest_run_id"),
        Index("idx_backtest_trades_symbol", "symbol"),
        Index("idx_backtest_trades_side", "side"),
        Index("idx_backtest_trades_entry_time", "entry_timestamp"),
        Index("idx_backtest_trades_exit_time", "exit_timestamp"),
        Index("idx_backtest_trades_pnl", "pnl_usd"),
        Index("idx_backtest_trades_duration", "duration_hours"),
        Index("idx_backtest_trades_run_sequence", "backtest_run_id", "trade_sequence"),
        Index("idx_backtest_trades_run_symbol", "backtest_run_id", "symbol"),
        Index("idx_backtest_trades_performance", "backtest_run_id", "pnl_usd"),
        # Business constraints
        CheckConstraint("exit_timestamp > entry_timestamp", name="check_backtest_trade_time_order"),
        CheckConstraint("entry_price > 0", name="check_backtest_trade_entry_price_positive"),
        CheckConstraint("exit_price > 0", name="check_backtest_trade_exit_price_positive"),
        CheckConstraint("quantity > 0", name="check_backtest_trade_quantity_positive"),
        CheckConstraint(
            "position_size_usd > 0", name="check_backtest_trade_position_size_positive"
        ),
        CheckConstraint(
            "commission_paid >= 0", name="check_backtest_trade_commission_non_negative"
        ),
        CheckConstraint("slippage_cost >= 0", name="check_backtest_trade_slippage_non_negative"),
        CheckConstraint("duration_hours > 0", name="check_backtest_trade_duration_positive"),
        CheckConstraint("trade_sequence > 0", name="check_backtest_trade_sequence_positive"),
        CheckConstraint(
            "side IN ('LONG', 'SHORT')",
            name="check_backtest_trade_side",
        ),
        CheckConstraint(
            "exit_reason IN ('signal', 'stop_loss', 'take_profit', 'max_duration', 'backtest_end', 'risk_limit')",
            name="check_backtest_trade_exit_reason",
        ),
        CheckConstraint(
            "entry_signal_strength IS NULL OR (entry_signal_strength >= 0 AND entry_signal_strength <= 1)",
            name="check_backtest_trade_signal_strength_range",
        ),
        CheckConstraint(
            "bars_held IS NULL OR bars_held > 0",
            name="check_backtest_trade_bars_held_positive",
        ),
        CheckConstraint(
            "max_adverse_excursion_pct IS NULL OR max_adverse_excursion_pct <= 0",
            name="check_backtest_trade_mae_non_positive",
        ),
        CheckConstraint(
            "max_favorable_excursion_pct IS NULL OR max_favorable_excursion_pct >= 0",
            name="check_backtest_trade_mfe_non_negative",
        ),
        UniqueConstraint(
            "backtest_run_id", "trade_sequence", name="uq_backtest_trades_run_sequence"
        ),
    )
