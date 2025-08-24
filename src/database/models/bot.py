"""Bot and strategy database models."""

import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

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
        String(20), nullable=False, default="STOPPED"
    )  # INITIALIZING, RUNNING, PAUSED, STOPPING, STOPPED, ERROR

    exchange = Column(String(50), nullable=False)
    test_mode = Column(Boolean, default=False)
    paper_trading = Column(Boolean, default=False)

    # Capital allocation
    allocated_capital = Column(Float, default=0)
    current_balance = Column(Float, default=0)

    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)

    # Configuration
    config = Column(JSON, default={})

    # Relationships
    strategies = relationship("Strategy", back_populates="bot", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="bot")
    positions = relationship("Position", back_populates="bot")
    trades = relationship("Trade", back_populates="bot")
    logs = relationship("BotLog", back_populates="bot", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_bots_status", "status"),
        Index("idx_bots_exchange", "exchange"),
        Index("idx_bots_created_at", "created_at"),
        CheckConstraint("allocated_capital >= 0", name="check_allocated_capital_non_negative"),
    )

    def __repr__(self):
        return f"<Bot {self.id}: {self.name} ({self.status})>"

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self.status == "RUNNING"

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def average_pnl(self) -> float:
        """Calculate average P&L per trade."""
        if self.total_trades == 0:
            return 0
        return self.total_pnl / self.total_trades


class Strategy(Base, AuditMixin, MetadataMixin):
    """Strategy model."""

    __tablename__ = "strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # market_making, arbitrage, trend_following, etc.
    status = Column(
        String(20), nullable=False, default="INACTIVE"
    )  # ACTIVE, INACTIVE, PAUSED, ERROR

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)

    # Strategy parameters
    params = Column(JSON, default={})

    # Risk parameters
    max_position_size = Column(Float)
    risk_per_trade = Column(Float)
    stop_loss_percentage = Column(Float)
    take_profit_percentage = Column(Float)

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

    # Indexes
    __table_args__ = (
        Index("idx_strategies_bot_id", "bot_id"),
        Index("idx_strategies_status", "status"),
        Index("idx_strategies_type", "type"),
        CheckConstraint(
            "risk_per_trade >= 0 AND risk_per_trade <= 1", name="check_risk_per_trade_range"
        ),
    )

    def __repr__(self):
        return f"<Strategy {self.id}: {self.name} ({self.type})>"

    @property
    def is_active(self) -> bool:
        """Check if strategy is active."""
        return self.status == "ACTIVE"

    @property
    def signal_success_rate(self) -> float:
        """Calculate signal success rate."""
        if self.executed_signals == 0:
            return 0
        return (self.successful_signals / self.executed_signals) * 100


class Signal(Base, TimestampMixin, MetadataMixin):
    """Trading signal model."""

    __tablename__ = "signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)

    symbol = Column(String(50), nullable=False)
    action = Column(String(20), nullable=False)  # BUY, SELL, HOLD, CLOSE
    strength = Column(Float)  # Signal strength 0-1

    # Signal details
    price = Column(Float)
    quantity = Column(Float)
    reason = Column(String(500))

    # Execution status
    executed = Column(Boolean, default=False)
    execution_time = Column(Float)  # Time to execute in seconds
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"))

    # Outcome
    outcome = Column(String(20))  # SUCCESS, FAILURE, PARTIAL, EXPIRED
    pnl = Column(Float)

    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    order = relationship("Order")

    # Indexes
    __table_args__ = (
        Index("idx_signals_strategy_id", "strategy_id"),
        Index("idx_signals_symbol", "symbol"),
        Index("idx_signals_created_at", "created_at"),
        Index("idx_signals_executed", "executed"),
    )

    def __repr__(self):
        return f"<Signal {self.id}: {self.action} {self.symbol}>"

    @property
    def is_executed(self) -> bool:
        """Check if signal was executed."""
        return self.executed

    @property
    def is_successful(self) -> bool:
        """Check if signal was successful."""
        return self.outcome == "SUCCESS"


class BotLog(Base, TimestampMixin):
    """Bot activity log model."""

    __tablename__ = "bot_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)

    level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category = Column(String(50))  # strategy, execution, risk, system, etc.
    message = Column(String(2000), nullable=False)

    # Additional context
    context = Column(JSON)

    # Relationships
    bot = relationship("Bot", back_populates="logs")

    # Indexes
    __table_args__ = (
        Index("idx_bot_logs_bot_id", "bot_id"),
        Index("idx_bot_logs_level", "level"),
        Index("idx_bot_logs_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<BotLog {self.id}: [{self.level}] {self.message[:50]}>"
