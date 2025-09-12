"""Analytics database models."""

import uuid
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import Base, TimestampMixin


class AnalyticsPortfolioMetrics(Base, TimestampMixin):
    """Portfolio metrics storage."""

    __tablename__ = "analytics_portfolio_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Foreign Keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)

    # Portfolio metrics
    total_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    daily_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    number_of_positions = Column(Integer, nullable=False, default=0)
    leverage_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    margin_usage: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    cash_balance: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Relationships
    bot = relationship("Bot", back_populates="portfolio_metrics")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_analytics_portfolio_bot_id", "bot_id"),
        Index("idx_analytics_portfolio_timestamp", "timestamp"),
        Index("idx_analytics_portfolio_bot_timestamp", "bot_id", "timestamp"),
        CheckConstraint("total_value >= 0", name="check_portfolio_total_value_non_negative"),
        CheckConstraint("number_of_positions >= 0", name="check_portfolio_positions_non_negative"),
        CheckConstraint(
            "leverage_ratio IS NULL OR leverage_ratio >= 0",
            name="check_portfolio_leverage_non_negative",
        ),
        CheckConstraint(
            "margin_usage IS NULL OR (margin_usage >= 0 AND margin_usage <= 1)",
            name="check_portfolio_margin_range",
        ),
        CheckConstraint(
            "cash_balance IS NULL OR cash_balance >= 0", name="check_portfolio_cash_non_negative"
        ),
        UniqueConstraint("bot_id", "timestamp", name="uq_portfolio_metrics_bot_timestamp"),
        {"extend_existing": True},
    )


class AnalyticsPositionMetrics(Base, TimestampMixin):
    """Position metrics storage."""

    __tablename__ = "analytics_position_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Foreign Keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    position_id = Column(
        UUID(as_uuid=True), ForeignKey("positions.id", ondelete="CASCADE"), nullable=True
    )

    # Position data
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    market_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    average_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    position_side = Column(String(10), nullable=False)  # 'LONG' or 'SHORT'

    # Relationships
    bot = relationship("Bot", back_populates="position_metrics")
    position = relationship("Position", back_populates="position_metrics")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_analytics_position_bot_id", "bot_id"),
        Index("idx_analytics_position_timestamp", "timestamp"),
        Index("idx_analytics_position_symbol", "symbol"),
        Index("idx_analytics_position_exchange", "exchange"),
        Index("idx_analytics_position_bot_symbol", "bot_id", "symbol"),
        Index("idx_analytics_position_bot_timestamp", "bot_id", "timestamp"),
        CheckConstraint("quantity != 0", name="check_position_quantity_not_zero"),
        CheckConstraint("market_value >= 0", name="check_position_market_value_non_negative"),
        CheckConstraint("average_price > 0", name="check_position_avg_price_positive"),
        CheckConstraint("current_price > 0", name="check_position_current_price_positive"),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')", name="check_position_exchange"
        ),
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="check_position_side"),
        {"extend_existing": True},
    )


class AnalyticsRiskMetrics(Base, TimestampMixin):
    """Risk metrics storage."""

    __tablename__ = "analytics_risk_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Foreign Keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)

    # Risk metrics
    portfolio_var_95: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    portfolio_var_99: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    expected_shortfall_95: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    maximum_drawdown: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    volatility: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    sortino_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    correlation_risk: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)
    concentration_risk: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Relationships
    bot = relationship("Bot", back_populates="risk_metrics")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_analytics_risk_bot_id", "bot_id"),
        Index("idx_analytics_risk_timestamp", "timestamp"),
        Index("idx_analytics_risk_bot_timestamp", "bot_id", "timestamp"),
        CheckConstraint(
            "portfolio_var_95 IS NULL OR portfolio_var_95 >= 0",
            name="check_risk_var_95_non_negative",
        ),
        CheckConstraint(
            "portfolio_var_99 IS NULL OR portfolio_var_99 >= 0",
            name="check_risk_var_99_non_negative",
        ),
        CheckConstraint(
            "expected_shortfall_95 IS NULL OR expected_shortfall_95 >= 0",
            name="check_risk_es_95_non_negative",
        ),
        CheckConstraint(
            "maximum_drawdown IS NULL OR (maximum_drawdown >= 0 AND maximum_drawdown <= 1)",
            name="check_risk_max_dd_range",
        ),
        CheckConstraint(
            "volatility IS NULL OR volatility >= 0", name="check_risk_volatility_non_negative"
        ),
        CheckConstraint(
            "correlation_risk IS NULL OR (correlation_risk >= -1 AND correlation_risk <= 1)",
            name="check_risk_correlation_range",
        ),
        CheckConstraint(
            "concentration_risk IS NULL OR (concentration_risk >= 0 AND concentration_risk <= 1)",
            name="check_risk_concentration_range",
        ),
        UniqueConstraint("bot_id", "timestamp", name="uq_risk_metrics_bot_timestamp"),
        {"extend_existing": True},
    )


class AnalyticsStrategyMetrics(Base, TimestampMixin):
    """Strategy performance metrics storage."""

    __tablename__ = "analytics_strategy_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Foreign Keys
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )

    # Strategy identification
    strategy_name = Column(String(100), nullable=False, index=True)

    # Performance metrics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    total_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=0)
    average_win: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    average_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(
        DECIMAL(5, 4), nullable=True
    )  # Percentage as decimal
    profit_factor: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 4), nullable=True)
    maximum_drawdown: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Relationships
    bot = relationship("Bot", back_populates="strategy_metrics")
    strategy = relationship("Strategy", back_populates="strategy_metrics")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_analytics_strategy_bot_id", "bot_id"),
        Index("idx_analytics_strategy_strategy_id", "strategy_id"),
        Index("idx_analytics_strategy_timestamp", "timestamp"),
        Index("idx_analytics_strategy_name", "strategy_name"),
        Index("idx_analytics_strategy_bot_timestamp", "bot_id", "timestamp"),
        Index("idx_analytics_strategy_strategy_timestamp", "strategy_id", "timestamp"),
        Index("idx_analytics_strategy_performance", "total_pnl", "total_trades", "timestamp"),
        CheckConstraint("total_trades >= 0", name="check_strategy_total_trades_non_negative"),
        CheckConstraint("winning_trades >= 0", name="check_strategy_winning_trades_non_negative"),
        CheckConstraint("losing_trades >= 0", name="check_strategy_losing_trades_non_negative"),
        CheckConstraint(
            "winning_trades + losing_trades <= total_trades",
            name="check_strategy_trades_consistency",
        ),
        CheckConstraint(
            "average_win IS NULL OR average_win > 0", name="check_strategy_avg_win_positive"
        ),
        CheckConstraint(
            "average_loss IS NULL OR average_loss < 0", name="check_strategy_avg_loss_negative"
        ),
        CheckConstraint(
            "win_rate IS NULL OR (win_rate >= 0 AND win_rate <= 1)",
            name="check_strategy_win_rate_range",
        ),
        CheckConstraint(
            "profit_factor IS NULL OR profit_factor >= 0",
            name="check_strategy_profit_factor_non_negative",
        ),
        CheckConstraint(
            "maximum_drawdown IS NULL OR (maximum_drawdown >= 0 AND maximum_drawdown <= 1)",
            name="check_strategy_max_dd_range",
        ),
        UniqueConstraint(
            "bot_id", "strategy_id", "timestamp", name="uq_strategy_metrics_bot_strategy_timestamp"
        ),
        {"extend_existing": True},
    )


class AnalyticsOperationalMetrics(Base, TimestampMixin):
    """Operational metrics storage."""

    __tablename__ = "analytics_operational_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Foreign Keys
    bot_id = Column(
        UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=True
    )  # Nullable for system-wide metrics

    # Operational metrics
    orders_per_minute: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False, default=0)
    trades_per_minute: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False, default=0)
    api_latency_avg: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 3), nullable=True
    )  # milliseconds
    api_latency_p95: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 3), nullable=True)
    websocket_latency_avg: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 3), nullable=True)
    error_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False, default=0)
    success_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), nullable=False, default=1)
    active_connections = Column(Integer, nullable=False, default=0)
    memory_usage_mb: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 2), nullable=True)
    cpu_usage_percent: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 2), nullable=True)
    database_connections_active = Column(Integer, nullable=False, default=0)
    database_query_avg_time: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 3), nullable=True
    )  # milliseconds

    # Relationships
    bot = relationship("Bot", back_populates="operational_metrics")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_analytics_operational_bot_id", "bot_id"),
        Index("idx_analytics_operational_timestamp", "timestamp"),
        Index("idx_analytics_operational_bot_timestamp", "bot_id", "timestamp"),
        CheckConstraint(
            "orders_per_minute >= 0", name="check_operational_orders_per_min_non_negative"
        ),
        CheckConstraint(
            "trades_per_minute >= 0", name="check_operational_trades_per_min_non_negative"
        ),
        CheckConstraint(
            "api_latency_avg IS NULL OR api_latency_avg >= 0",
            name="check_operational_api_latency_avg_non_negative",
        ),
        CheckConstraint(
            "api_latency_p95 IS NULL OR api_latency_p95 >= 0",
            name="check_operational_api_latency_p95_non_negative",
        ),
        CheckConstraint(
            "websocket_latency_avg IS NULL OR websocket_latency_avg >= 0",
            name="check_operational_ws_latency_non_negative",
        ),
        CheckConstraint(
            "error_rate >= 0 AND error_rate <= 1", name="check_operational_error_rate_range"
        ),
        CheckConstraint(
            "success_rate >= 0 AND success_rate <= 1", name="check_operational_success_rate_range"
        ),
        CheckConstraint(
            "active_connections >= 0", name="check_operational_connections_non_negative"
        ),
        CheckConstraint(
            "memory_usage_mb IS NULL OR memory_usage_mb >= 0",
            name="check_operational_memory_non_negative",
        ),
        CheckConstraint(
            "cpu_usage_percent IS NULL OR (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100)",
            name="check_operational_cpu_range",
        ),
        CheckConstraint(
            "database_connections_active >= 0", name="check_operational_db_connections_non_negative"
        ),
        CheckConstraint(
            "database_query_avg_time IS NULL OR database_query_avg_time >= 0",
            name="check_operational_db_query_time_non_negative",
        ),
        CheckConstraint(
            "error_rate + success_rate <= 1.01", name="check_operational_rate_consistency"
        ),  # Allow small rounding errors
        {"extend_existing": True},
    )
