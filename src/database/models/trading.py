"""Trading-related database models."""

import uuid
from decimal import Decimal

from sqlalchemy import DECIMAL, CheckConstraint, Column, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship

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

    price: Mapped[Decimal | None] = Column(DECIMAL(20, 8))  # 8 decimal places for crypto precision
    quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)
    average_fill_price: Mapped[Decimal | None] = Column(DECIMAL(20, 8))

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
        if self.filled_quantity is None or self.quantity is None:
            return False
        return bool(self.status == "FILLED" and self.filled_quantity >= self.quantity)

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in ("PENDING", "OPEN", "PARTIALLY_FILLED")

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

    quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = Column(DECIMAL(20, 8))
    current_price: Mapped[Decimal | None] = Column(DECIMAL(20, 8))

    realized_pnl: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)
    unrealized_pnl: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)

    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))

    # Risk management
    stop_loss: Mapped[Decimal | None] = Column(DECIMAL(20, 8))
    take_profit: Mapped[Decimal | None] = Column(DECIMAL(20, 8))
    max_position_size: Mapped[Decimal | None] = Column(DECIMAL(20, 8))

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
        return bool(self.status == "OPEN")

    @property
    def value(self) -> Decimal:
        """Get current position value."""
        price = self.current_price or self.entry_price
        if price is None or self.quantity is None:
            return Decimal("0")
        return self.quantity * price

    def calculate_pnl(self, current_price: Decimal | float | None = None) -> Decimal:
        """Calculate P&L."""
        if current_price is not None:
            price = Decimal(str(current_price))
        else:
            price = self.current_price or self.entry_price
            if price is None:
                return Decimal("0")

        entry_price = self.entry_price
        quantity = self.quantity

        if self.side == "LONG":
            return (price - entry_price) * quantity
        else:  # SHORT
            return (entry_price - price) * quantity


class OrderFill(Base, TimestampMixin):
    """Order fill/execution model."""

    __tablename__ = "order_fills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
    exchange_fill_id = Column(String(255))

    price: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    fee: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)
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

    entry_order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"))
    exit_order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"))
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"))

    quantity: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    exit_price: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)

    pnl: Mapped[Decimal] = Column(DECIMAL(20, 8), nullable=False)
    pnl_percentage: Mapped[Decimal | None] = Column(DECIMAL(10, 4))  # Percentage (4 decimals)
    fees: Mapped[Decimal] = Column(DECIMAL(20, 8), default=0)
    net_pnl: Mapped[Decimal | None] = Column(DECIMAL(20, 8))

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
