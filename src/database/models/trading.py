"""Trading-related database models."""

import uuid
from decimal import Decimal

from sqlalchemy import DECIMAL, CheckConstraint, Column, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import AuditMixin, Base, MetadataMixin, TimestampMixin


class Order(Base, AuditMixin, MetadataMixin):
    """Order model."""

    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    client_order_id = Column(String(255))
    exchange_order_id = Column(String(255))
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # BUY, SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP, etc.
    status = Column(String(20), nullable=False)  # PENDING, OPEN, FILLED, CANCELLED, etc.

    price: Mapped[Decimal | None] = mapped_column(
        DECIMAL(20, 8)
    )  # 8 decimal places for crypto precision
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    average_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position_id = Column(
        UUID(as_uuid=True), ForeignKey("positions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Relationships
    bot = relationship("Bot", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    position = relationship("Position", back_populates="orders")
    fills = relationship("OrderFill", back_populates="order", cascade="all, delete-orphan")
    signal = relationship("Signal", back_populates="order", uselist=False)
    execution_audit_logs = relationship(
        "ExecutionAuditLog", foreign_keys="ExecutionAuditLog.order_id", back_populates="order"
    )

    # Indexes - Performance Optimized
    __table_args__ = (
        Index("idx_orders_exchange_symbol", "exchange", "symbol"),
        Index("idx_orders_status", "status"),
        Index("idx_orders_bot_id", "bot_id"),
        Index("idx_orders_created_at", "created_at"),
        # High-performance composite indexes for order execution pipeline
        Index("idx_orders_status_created_at", "status", "created_at"),  # Critical for order latency
        Index("idx_orders_bot_id_status", "bot_id", "status"),  # Bot-specific order queries
        Index("idx_orders_symbol_status", "symbol", "status"),  # Symbol-specific active orders
        Index(
            "idx_orders_exchange_status_created", "exchange", "status", "created_at"
        ),  # Exchange order tracking
        Index(
            "idx_orders_exchange_symbol_status", "exchange", "symbol", "status"
        ),  # Exchange trading pair monitoring
        Index(
            "idx_orders_strategy_status", "strategy_id", "status"
        ),  # Strategy execution monitoring
        Index("idx_orders_position_id", "position_id"),  # Position-related order tracking
        Index(
            "idx_orders_execution_priority", "status", "price", "created_at"
        ),  # Execution priority queue
        Index("idx_orders_client_order_id", "client_order_id"),  # Client order tracking
        Index("idx_orders_symbol_created", "symbol", "created_at"),  # Symbol time-series queries
        UniqueConstraint("exchange", "exchange_order_id", name="uq_exchange_order_id"),
        CheckConstraint("quantity > 0", name="check_quantity_positive"),
        CheckConstraint("filled_quantity >= 0", name="check_filled_quantity_non_negative"),
        CheckConstraint("filled_quantity <= quantity", name="check_filled_quantity_max"),
        CheckConstraint("price IS NULL OR price > 0", name="check_price_positive_when_set"),
        CheckConstraint(
            "average_price IS NULL OR average_price > 0", name="check_avg_price_positive"
        ),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')", name="check_supported_exchange"
        ),
        CheckConstraint("side IN ('buy', 'sell')", name="check_order_side"),
        CheckConstraint(
            "order_type IN ('market', 'limit', 'stop_loss', 'take_profit')",
            name="check_order_type",
        ),
        CheckConstraint(
            "status IN ('new', 'pending', 'open', 'partially_filled', 'filled', 'cancelled', 'expired', 'rejected', 'unknown')",
            name="check_order_status",
        ),
        CheckConstraint(
            "average_price IS NULL OR status != 'filled' OR average_price IS NOT NULL",
            name="check_filled_orders_have_avg_price",
        ),
    )

    def __repr__(self):
        return f"<Order {self.id}: {self.side} {self.quantity} {self.symbol} @ {self.price}>"

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        if self.filled_quantity is None or self.quantity is None:
            return False
        return bool(self.status == "filled" and self.filled_quantity >= self.quantity)

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in ("pending", "open", "partially_filled")

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to fill."""
        if self.quantity is None:
            return Decimal("0")
        filled = self.filled_quantity if self.filled_quantity else Decimal("0")
        return self.quantity - filled


class Position(Base, AuditMixin, MetadataMixin):
    """Position model."""

    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # LONG, SHORT
    status = Column(String(20), nullable=False)  # OPEN, CLOSED, LIQUIDATED

    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    current_price: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    unrealized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Risk management
    stop_loss: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    take_profit: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    max_position_size: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    # Relationships
    bot = relationship("Bot", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    orders = relationship("Order", back_populates="position")
    trades = relationship("Trade", back_populates="position")
    position_metrics = relationship(
        "AnalyticsPositionMetrics", back_populates="position", cascade="all, delete-orphan"
    )
    
    # State management relationships  
    state_snapshots = relationship("StateSnapshot", back_populates="position")

    # Indexes
    __table_args__ = (
        Index("idx_positions_exchange_symbol", "exchange", "symbol"),
        Index("idx_positions_status", "status"),
        Index("idx_positions_bot_id", "bot_id"),
        Index(
            "idx_positions_exchange_status", "exchange", "status"
        ),  # Exchange position monitoringw
        Index(
            "idx_positions_strategy_exchange", "strategy_id", "exchange"
        ),  # Strategy-exchange performance
        Index("idx_positions_realized_pnl", "realized_pnl"),  # P&L analysis
        Index("idx_positions_unrealized_pnl", "unrealized_pnl"),  # P&L analysis
        Index("idx_positions_created_at", "created_at"),  # Time-series analysis
        CheckConstraint("quantity > 0", name="check_position_quantity_positive"),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')", name="check_position_exchange"
        ),
        CheckConstraint("side IN ('LONG', 'SHORT')", name="check_position_side"),
        CheckConstraint("status IN ('OPEN', 'CLOSED', 'LIQUIDATED')", name="check_position_status"),
        CheckConstraint("entry_price > 0", name="check_entry_price_positive"),
        CheckConstraint("exit_price IS NULL OR exit_price > 0", name="check_exit_price_positive"),
        CheckConstraint(
            "current_price IS NULL OR current_price > 0", name="check_current_price_positive"
        ),
        CheckConstraint("stop_loss IS NULL OR stop_loss > 0", name="check_stop_loss_positive"),
        CheckConstraint(
            "take_profit IS NULL OR take_profit > 0", name="check_take_profit_positive"
        ),
        CheckConstraint(
            "max_position_size IS NULL OR max_position_size > 0",
            name="check_max_position_size_positive",
        ),
        CheckConstraint(
            "max_position_size IS NULL OR quantity <= max_position_size",
            name="check_quantity_within_max",
        ),
    )

    def __repr__(self):
        return f"<Position {self.id}: {self.side} {self.quantity} {self.symbol}>"

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return bool(self.status == "OPEN")

    @property
    def value(self) -> Decimal:
        """Get current position value."""
        if self.current_price is None or self.quantity is None:
            return Decimal("0")
        return self.quantity * self.current_price

    def calculate_pnl(self, current_price: Decimal | None = None) -> Decimal:
        """Calculate P&L."""
        if current_price is not None:
            price = Decimal(str(current_price))
        else:
            price = self.current_price or self.entry_price
            if price is None:
                return Decimal("0")

        entry_price = self.entry_price
        quantity = self.quantity

        # Handle zero entry price edge case
        if entry_price is None or entry_price == 0:
            return Decimal("0")

        # Handle both string and enum values for side
        side_value = self.side.value if hasattr(self.side, "value") else self.side

        if side_value == "LONG":
            return (price - entry_price) * quantity
        else:  # SHORT
            return (entry_price - price) * quantity


class OrderFill(Base, TimestampMixin):
    """Order fill/execution model."""

    __tablename__ = "order_fills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(
        UUID(as_uuid=True), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    exchange_fill_id = Column(String(255))

    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    fee_currency = Column(String(20))

    # Relationships
    order = relationship("Order", back_populates="fills")

    # Indexes
    __table_args__ = (
        Index("idx_fills_order_id", "order_id"),
        Index("idx_fills_created_at", "created_at"),
        Index(
            "idx_fills_order_created", "order_id", "created_at"
        ),  # Order fill history optimization
        Index("idx_fills_exchange_fill_id", "exchange_fill_id"),  # Exchange fill tracking
        Index("idx_fills_price_quantity", "price", "quantity"),  # Fill analysis queries
        CheckConstraint("quantity > 0", name="check_fill_quantity_positive"),
        CheckConstraint("price > 0", name="check_fill_price_positive"),
        CheckConstraint("fee >= 0", name="check_fill_fee_non_negative"),
    )

    def __repr__(self):
        return f"<OrderFill {self.id}: {self.quantity} @ {self.price}>"

    @property
    def value(self) -> Decimal:
        """Get fill value."""
        if self.price is None or self.quantity is None:
            return Decimal("0")
        return self.price * self.quantity

    @property
    def net_value(self) -> Decimal:
        """Get fill value after fees."""
        fee = self.fee if self.fee else Decimal("0")
        return self.value - fee


class Trade(Base, TimestampMixin, MetadataMixin):
    """Completed trade model."""

    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)

    entry_order_id = Column(
        UUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    exit_order_id = Column(
        UUID(as_uuid=True), ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    position_id = Column(
        UUID(as_uuid=True), ForeignKey("positions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    pnl_percentage: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 4)
    )  # Percentage (4 decimals)
    fees: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    net_pnl: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationships
    bot = relationship("Bot", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")
    entry_order = relationship("Order", foreign_keys=[entry_order_id], overlaps="orders")
    exit_order = relationship("Order", foreign_keys=[exit_order_id], overlaps="orders")
    position = relationship("Position", back_populates="trades")
    execution_audit_logs = relationship(
        "ExecutionAuditLog", foreign_keys="ExecutionAuditLog.trade_id", back_populates="trade"
    )

    # Indexes - Performance Optimized for Analytics
    __table_args__ = (
        Index("idx_trades_exchange_symbol", "exchange", "symbol"),
        Index("idx_trades_bot_id", "bot_id"),
        Index("idx_trades_created_at", "created_at"),
        Index("idx_trades_pnl", "pnl"),
        # Critical composite index for performance analytics
        Index("idx_trades_bot_id_timestamp", "bot_id", "created_at"),  # Bot performance over time
        Index("idx_trades_symbol_timestamp", "symbol", "created_at"),  # Symbol-specific analytics
        Index(
            "idx_trades_strategy_performance", "strategy_id", "pnl", "created_at"
        ),  # Strategy performance
        Index(
            "idx_trades_exchange_performance", "exchange", "pnl", "created_at"
        ),  # Exchange comparison
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')", name="check_trade_exchange"
        ),
        CheckConstraint("side IN ('buy', 'sell')", name="check_trade_side"),
        CheckConstraint("quantity > 0", name="check_trade_quantity_positive"),
        CheckConstraint("entry_price > 0", name="check_trade_entry_price_positive"),
        CheckConstraint("exit_price > 0", name="check_trade_exit_price_positive"),
        CheckConstraint("fees >= 0", name="check_trade_fees_non_negative"),
    )

    def __repr__(self):
        return f"<Trade {self.id}: {self.symbol} P&L={self.pnl}>"

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        if self.pnl is None:
            return False
        return self.pnl > Decimal("0")

    @property
    def return_percentage(self) -> Decimal:
        """Calculate return percentage."""
        if self.entry_price is None or self.entry_price == 0:
            return Decimal("0")
        entry = self.entry_price
        exit_price = self.exit_price if self.exit_price else Decimal("0")
        return ((exit_price - entry) / entry) * Decimal("100")
