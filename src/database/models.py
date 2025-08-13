"""
Database models for the trading bot framework.

This module defines all SQLAlchemy models for persistent storage of trading data,
user information, bot instances, and performance metrics.

CRITICAL: These models integrate with P-001 core types and will be used by
all subsequent prompts for data persistence.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Column,
    Numeric,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship
from sqlalchemy.sql import func

# Import core types from P-001 - only the ones actually used
from src.core.types import OrderSide, OrderStatus, OrderType, StrategyStatus, StrategyType

Base = declarative_base()


class User(Base):
    """User model for authentication and account management."""

    __tablename__ = "users"

    # Primary key using UUID
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # User credentials
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # User status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    bot_instances: Mapped[list["BotInstance"]] = relationship(
        "BotInstance", back_populates="user", cascade="all, delete-orphan"
    )
    balance_snapshots: Mapped[list["BalanceSnapshot"]] = relationship(
        "BalanceSnapshot", back_populates="user", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="user", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("length(username) >= 3", name="username_min_length"),
        CheckConstraint("length(email) >= 5", name="email_min_length"),
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_active", "is_active"),
    )


class BotInstance(Base):
    """Bot instance model for managing individual trading bots."""

    __tablename__ = "bot_instances"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Bot identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True
    )

    # Bot configuration
    strategy_type: Mapped[StrategyType] = mapped_column(
        String(50), nullable=False, index=True
    )  # Use core StrategyType enum
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[StrategyStatus] = mapped_column(
        String(20), default="stopped", nullable=False, index=True
    )  # Use core StrategyStatus enum

    # Configuration and settings
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_active: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="bot_instances")
    trades: Mapped[list["Trade"]] = relationship(
        "Trade", back_populates="bot_instance", cascade="all, delete-orphan"
    )
    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="bot_instance", cascade="all, delete-orphan"
    )
    performance_metrics: Mapped[list["PerformanceMetrics"]] = relationship(
        "PerformanceMetrics", back_populates="bot_instance", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('stopped', 'running', 'paused', 'error')", name="valid_bot_status"
        ),
        Index("idx_bot_instances_user_id", "user_id"),
        Index("idx_bot_instances_strategy_type", "strategy_type"),
        Index("idx_bot_instances_exchange", "exchange"),
        Index("idx_bot_instances_status", "status"),
        UniqueConstraint("user_id", "name", name="unique_user_bot_name"),
    )


class Trade(Base):
    """Trade model for tracking executed trades."""

    __tablename__ = "trades"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Trade identification
    bot_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("bot_instances.id"), nullable=False, index=True
    )
    exchange_order_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Trade details
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[OrderSide] = mapped_column(String(10), nullable=False)  # Use core OrderSide enum
    order_type: Mapped[OrderType] = mapped_column(
        String(20), nullable=False
    )  # Use core OrderType enum

    # Quantities and prices
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    executed_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # Fees and costs
    fee: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=Decimal("0"), nullable=False)
    fee_currency: Mapped[str] = mapped_column(String(10), default="USDT", nullable=False)

    # Status and P&L
    status: Mapped[OrderStatus] = mapped_column(
        String(20), nullable=False, index=True
    )  # Use core OrderStatus enum
    pnl: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    bot_instance: Mapped["BotInstance"] = relationship("BotInstance", back_populates="trades")

    # Constraints
    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="valid_trade_side"),
        CheckConstraint(
            "order_type IN ('market', 'limit', 'stop_loss', 'take_profit')", name="valid_order_type"
        ),
        CheckConstraint(
            "status IN ('pending', 'filled', 'cancelled', 'rejected')", name="valid_trade_status"
        ),
        CheckConstraint("quantity > 0", name="positive_quantity"),
        CheckConstraint("price > 0", name="positive_price"),
        Index("idx_trades_bot_id", "bot_id"),
        Index("idx_trades_exchange", "exchange"),
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_timestamp", "timestamp"),
        Index("idx_trades_status", "status"),
    )


class Position(Base):
    """Position model for tracking open positions."""

    __tablename__ = "positions"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Position identification
    bot_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("bot_instances.id"), nullable=False, index=True
    )
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Position details
    side: Mapped[OrderSide] = mapped_column(String(10), nullable=False)  # Use core OrderSide enum
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # P&L tracking
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), default=Decimal("0"), nullable=False
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), default=Decimal("0"), nullable=False
    )

    # Risk management
    stop_loss_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    bot_instance: Mapped["BotInstance"] = relationship("BotInstance", back_populates="positions")

    # Constraints
    __table_args__ = (
        CheckConstraint("side IN ('long', 'short')", name="valid_position_side"),
        CheckConstraint("quantity > 0", name="positive_position_quantity"),
        CheckConstraint("entry_price > 0", name="positive_entry_price"),
        CheckConstraint("current_price > 0", name="positive_current_price"),
        Index("idx_positions_bot_id", "bot_id"),
        Index("idx_positions_exchange", "exchange"),
        Index("idx_positions_symbol", "symbol"),
        Index("idx_positions_side", "side"),
        UniqueConstraint("bot_id", "exchange", "symbol", "side", name="unique_position"),
    )


class BalanceSnapshot(Base):
    """Balance snapshot model for tracking account balances."""

    __tablename__ = "balance_snapshots"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Balance identification
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True
    )
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Balance amounts
    free_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    locked_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    total_balance: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # Value conversions
    btc_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)
    usd_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8), nullable=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="balance_snapshots")

    # Constraints
    __table_args__ = (
        CheckConstraint("free_balance >= 0", name="non_negative_free_balance"),
        CheckConstraint("locked_balance >= 0", name="non_negative_locked_balance"),
        CheckConstraint("total_balance >= 0", name="non_negative_total_balance"),
        Index("idx_balance_snapshots_user_id", "user_id"),
        Index("idx_balance_snapshots_exchange", "exchange"),
        Index("idx_balance_snapshots_currency", "currency"),
        Index("idx_balance_snapshots_timestamp", "timestamp"),
    )


class StrategyConfig(Base):
    """Strategy configuration model for storing strategy parameters."""

    __tablename__ = "strategy_configs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Configuration identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Configuration data
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    risk_parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), default="1.0.0", nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_strategy_configs_name", "name"),
        Index("idx_strategy_configs_type", "strategy_type"),
        Index("idx_strategy_configs_active", "is_active"),
        UniqueConstraint("name", "version", name="unique_strategy_config"),
    )


class MLModel(Base):
    """Machine learning model model for tracking ML models."""

    __tablename__ = "ml_models"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Model identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False)

    # Model data
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Training information
    training_data_range: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # e.g., "2023-01-01 to 2023-12-31"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_ml_models_name", "name"),
        Index("idx_ml_models_type", "model_type"),
        Index("idx_ml_models_active", "is_active"),
        UniqueConstraint("name", "version", name="unique_ml_model"),
    )


class PerformanceMetrics(Base):
    """Performance metrics model for tracking bot performance."""

    __tablename__ = "performance_metrics"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Metrics identification
    bot_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("bot_instances.id"), nullable=False, index=True
    )
    metric_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Trade counts
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # P&L values
    total_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), default=Decimal("0"), nullable=False
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), default=Decimal("0"), nullable=False
    )

    # Performance ratios
    win_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    bot_instance: Mapped["BotInstance"] = relationship(
        "BotInstance", back_populates="performance_metrics"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("total_trades >= 0", name="non_negative_total_trades"),
        CheckConstraint("winning_trades >= 0", name="non_negative_winning_trades"),
        CheckConstraint("losing_trades >= 0", name="non_negative_losing_trades"),
        CheckConstraint("win_rate >= 0 AND win_rate <= 1", name="valid_win_rate"),
        CheckConstraint("profit_factor >= 0", name="non_negative_profit_factor"),
        Index("idx_performance_metrics_bot_id", "bot_id"),
        Index("idx_performance_metrics_date", "metric_date"),
        UniqueConstraint("bot_id", "metric_date", name="unique_daily_metrics"),
    )


class Alert(Base):
    """Alert model for system notifications and alerts."""

    __tablename__ = "alerts"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Alert identification
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True
    )
    bot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("bot_instances.id"), nullable=True, index=True
    )

    # Alert details
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    alert_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="alerts")
    bot_instance: Mapped[Optional["BotInstance"]] = relationship("BotInstance")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "severity IN ('low', 'medium', 'high', 'critical')", name="valid_alert_severity"
        ),
        Index("idx_alerts_user_id", "user_id"),
        Index("idx_alerts_bot_id", "bot_id"),
        Index("idx_alerts_type", "alert_type"),
        Index("idx_alerts_severity", "severity"),
        Index("idx_alerts_read", "is_read"),
        Index("idx_alerts_timestamp", "timestamp"),
    )


class AuditLog(Base):
    """Audit log model for tracking system changes and user actions."""

    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Audit identification
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id"), nullable=True, index=True
    )

    # Action details
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Change tracking
    old_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Request details
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)  # IPv6 compatible
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")

    # Constraints
    __table_args__ = (
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_resource_type", "resource_type"),
        Index("idx_audit_logs_resource_id", "resource_id"),
        Index("idx_audit_logs_timestamp", "timestamp"),
    )


class CapitalAllocationDB(Base):
    """Database model for capital allocations."""
    __tablename__ = "capital_allocations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String, nullable=False, index=True)
    exchange = Column(String, nullable=False, index=True)
    allocated_amount = Column(Numeric(precision=20, scale=8), nullable=False)
    utilized_amount = Column(Numeric(precision=20, scale=8), nullable=False, default=0)
    available_amount = Column(Numeric(precision=20, scale=8), nullable=False, default=0)
    allocation_percentage = Column(Float, nullable=False)
    last_rebalance = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_strategy_exchange', 'strategy_id', 'exchange'),
        Index('idx_created_at', 'created_at'),
    )


class FundFlowDB(Base):
    """Database model for fund flows."""
    __tablename__ = "fund_flows"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    from_strategy = Column(String, nullable=True, index=True)
    to_strategy = Column(String, nullable=True, index=True)
    from_exchange = Column(String, nullable=True, index=True)
    to_exchange = Column(String, nullable=True, index=True)
    amount = Column(Numeric(precision=20, scale=8), nullable=False)
    currency = Column(String, nullable=False, index=True)
    reason = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    converted_amount = Column(Numeric(precision=20, scale=8), nullable=True)
    exchange_rate = Column(Numeric(precision=20, scale=8), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_reason', 'reason'),
        Index('idx_currency', 'currency'),
    )


class CurrencyExposureDB(Base):
    """Database model for currency exposures."""
    __tablename__ = "currency_exposures"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    currency = Column(String, nullable=False, index=True)
    base_currency_equivalent = Column(Numeric(precision=20, scale=8), nullable=False)
    hedging_required = Column(Boolean, nullable=False, default=False)
    hedge_amount = Column(Numeric(precision=20, scale=8), nullable=True)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_currency', 'currency'),
        Index('idx_last_updated', 'last_updated'),
    )


class ExchangeAllocationDB(Base):
    """Database model for exchange allocations."""
    __tablename__ = "exchange_allocations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    exchange = Column(String, nullable=False, index=True)
    allocated_amount = Column(Numeric(precision=20, scale=8), nullable=False)
    utilized_amount = Column(Numeric(precision=20, scale=8), nullable=False, default=0)
    available_amount = Column(Numeric(precision=20, scale=8), nullable=False, default=0)
    allocation_percentage = Column(Float, nullable=False)
    last_rebalance = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_exchange', 'exchange'),
        Index('idx_last_rebalance', 'last_rebalance'),
    )


class MarketDataRecord(Base):
    """Database model for market data records."""
    __tablename__ = "market_data_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Market data fields
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    # Price data
    open_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Volume and trade data
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quote_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trades_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Bid/Ask data
    bid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bid_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, default="exchange")
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(20), nullable=False, default="valid")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_market_data_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_market_data_exchange_timestamp", "exchange", "timestamp"),
        Index("idx_market_data_quality", "quality_score"),
        Index("idx_market_data_validation", "validation_status"),
        UniqueConstraint("symbol", "exchange", "timestamp", name="uq_market_data_unique"),
    )


class FeatureRecord(Base):
    """Database model for calculated features."""
    __tablename__ = "feature_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Feature identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    feature_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    calculation_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True)

    # Feature values
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Calculation metadata
    lookback_period: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    parameters: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    calculation_method: Mapped[str] = mapped_column(String(100), nullable=False, default="standard")

    # Data source
    source_data_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)
    source_data_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_feature_symbol_type", "symbol", "feature_type"),
        Index("idx_feature_timestamp", "calculation_timestamp"),
        Index("idx_feature_name", "feature_name"),
        UniqueConstraint("symbol", "feature_type", "feature_name",
                         "calculation_timestamp", name="uq_feature_unique"),
    )


class DataQualityRecord(Base):
    """Database model for data quality metrics."""
    __tablename__ = "data_quality_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Quality identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    quality_check_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True)

    # Quality metrics
    completeness_score: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy_score: Mapped[float] = mapped_column(Float, nullable=False)
    consistency_score: Mapped[float] = mapped_column(Float, nullable=False)
    timeliness_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Quality issues
    missing_data_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outlier_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_errors: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Metadata
    check_type: Mapped[str] = mapped_column(String(50), nullable=False, default="comprehensive")
    data_period_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)
    data_period_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_quality_symbol_timestamp", "symbol", "quality_check_timestamp"),
        Index("idx_quality_source_timestamp", "data_source", "quality_check_timestamp"),
        Index("idx_quality_overall_score", "overall_score"),
        Index("idx_quality_check_type", "check_type"),
    )


class DataPipelineRecord(Base):
    """Database model for data pipeline execution tracking."""
    __tablename__ = "data_pipeline_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Pipeline identification
    pipeline_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    execution_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    execution_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True)

    # Execution status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    stage: Mapped[str] = mapped_column(String(50), nullable=False, default="started")

    # Performance metrics
    records_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_successful: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_messages: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    configuration: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    dependencies: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        Index("idx_pipeline_name_timestamp", "pipeline_name", "execution_timestamp"),
        Index("idx_pipeline_status", "status"),
        Index("idx_pipeline_stage", "stage"),
        Index("idx_pipeline_execution_id", "execution_id"),
    )
