"""Trading-related database models."""

import uuid

from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database.models.base import AuditMixin, Base, MetadataMixin, TimestampMixin


class Order(Base, AuditMixin, MetadataMixin):
    """Order model."""

    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    exchange_order_id = Column(String(255))
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # BUY, SELL
    type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP, etc.
    status = Column(String(20), nullable=False)  # PENDING, OPEN, FILLED, CANCELLED, etc.

    price = Column(Float)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0)
    average_fill_price = Column(Float)

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"))

    # Relationships
    bot = relationship("Bot", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    position = relationship("Position", back_populates="orders")
    fills = relationship("OrderFill", back_populates="order", cascade="all, delete-orphan")

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
        UniqueConstraint("exchange", "exchange_order_id", name="uq_exchange_order_id"),
        CheckConstraint("quantity > 0", name="check_quantity_positive"),
        CheckConstraint("filled_quantity >= 0", name="check_filled_quantity_non_negative"),
        CheckConstraint("filled_quantity <= quantity", name="check_filled_quantity_max"),
    )

    def __repr__(self):
        return f"<Order {self.id}: {self.side} {self.quantity} {self.symbol} @ {self.price}>"

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == "FILLED" and self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in ("PENDING", "OPEN", "PARTIALLY_FILLED")

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity


class Position(Base, AuditMixin, MetadataMixin):
    """Position model."""

    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # LONG, SHORT
    status = Column(String(20), nullable=False)  # OPEN, CLOSED, LIQUIDATED

    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    current_price = Column(Float)

    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))

    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    max_position_size = Column(Float)

    # Relationships
    bot = relationship("Bot", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    orders = relationship("Order", back_populates="position")

    # Indexes
    __table_args__ = (
        Index("idx_positions_exchange_symbol", "exchange", "symbol"),
        Index("idx_positions_status", "status"),
        Index("idx_positions_bot_id", "bot_id"),
        CheckConstraint("quantity > 0", name="check_position_quantity_positive"),
    )

    def __repr__(self):
        return f"<Position {self.id}: {self.side} {self.quantity} {self.symbol}>"

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status == "OPEN"

    @property
    def value(self) -> float:
        """Get current position value."""
        price = self.current_price or self.entry_price
        return self.quantity * price

    def calculate_pnl(self, current_price: float | None = None) -> float:
        """Calculate P&L."""
        price = current_price or self.current_price or self.entry_price

        if self.side == "LONG":
            return (price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - price) * self.quantity


class OrderFill(Base, TimestampMixin):
    """Order fill/execution model."""

    __tablename__ = "order_fills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
    exchange_fill_id = Column(String(255))

    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    fee = Column(Float, default=0)
    fee_currency = Column(String(20))

    # Relationships
    order = relationship("Order", back_populates="fills")

    # Indexes
    __table_args__ = (
        Index("idx_fills_order_id", "order_id"),
        Index("idx_fills_created_at", "created_at"),
        CheckConstraint("quantity > 0", name="check_fill_quantity_positive"),
        CheckConstraint("price > 0", name="check_fill_price_positive"),
    )

    def __repr__(self):
        return f"<OrderFill {self.id}: {self.quantity} @ {self.price}>"

    @property
    def value(self) -> float:
        """Get fill value."""
        return self.price * self.quantity

    @property
    def net_value(self) -> float:
        """Get fill value after fees."""
        return self.value - (self.fee or 0)


class Trade(Base, TimestampMixin, MetadataMixin):
    """Completed trade model."""

    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)

    entry_order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"))
    exit_order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"))
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"))

    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)

    pnl = Column(Float, nullable=False)
    pnl_percentage = Column(Float)
    fees = Column(Float, default=0)
    net_pnl = Column(Float)

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))

    # Relationships
    bot = relationship("Bot", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")
    entry_order = relationship("Order", foreign_keys=[entry_order_id])
    exit_order = relationship("Order", foreign_keys=[exit_order_id])
    position = relationship("Position")

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
    )

    def __repr__(self):
        return f"<Trade {self.id}: {self.symbol} P&L={self.pnl}>"

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def return_percentage(self) -> float:
        """Calculate return percentage."""
        if self.entry_price == 0:
            return 0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
