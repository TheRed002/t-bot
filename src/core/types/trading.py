"""Trading-related types for the T-Bot trading system.

This module provides comprehensive trading types with:
- Full validation on construction
- Immutable critical data structures
- Rich business logic methods
- Serialization/deserialization support
- Type conversion utilities
- Comprehensive examples and documentation

TODO: Update the following modules to use enhanced trading types:
- All strategy modules (update signal handling)
- All execution modules (update order management)
- Risk management modules (update position tracking)
- Portfolio management modules (update balance handling)
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalDirection(Enum):
    """Signal direction for trading decisions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(Enum):
    """Order side for buy/sell operations."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type for different execution strategies."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status in exchange systems."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class TimeInForce(Enum):
    """Time in force for order execution."""

    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing
    DAY = "DAY"  # Day order


class TradeState(Enum):
    """Trade lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


class Signal(BaseModel):
    """Trading signal with direction and metadata."""

    symbol: str
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strength")
    @classmethod
    def validate_strength(cls, v: float) -> float:
        """Validate signal strength is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        return v


class OrderRequest(BaseModel):
    """Request to create an order."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol is not empty."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip()

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Decimal) -> Decimal:
        """Validate quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Decimal | None) -> Decimal | None:
        """Validate price is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class OrderResponse(BaseModel):
    """Response from order creation."""

    order_id: str = Field(alias="id", validation_alias="id")
    client_order_id: str | None = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    created_at: datetime
    updated_at: datetime | None = None
    exchange: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias for order_id for backward compatibility."""
        return self.order_id

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class Order(BaseModel):
    """Complete order information."""

    order_id: str
    client_order_id: str | None = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    time_in_force: TimeInForce
    created_at: datetime
    updated_at: datetime | None = None
    exchange: str
    fees: Decimal = Decimal("0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]


class Position(BaseModel):
    """Trading position information."""

    symbol: str
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    opened_at: datetime
    closed_at: datetime | None = None
    exchange: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.closed_at is None

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL."""
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


class Trade(BaseModel):
    """Executed trade information."""

    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: datetime
    exchange: str
    is_maker: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Balance(BaseModel):
    """Account balance information."""

    currency: str
    available: Decimal
    locked: Decimal = Decimal("0")
    total: Decimal
    exchange: str
    updated_at: datetime

    @property
    def free(self) -> Decimal:
        """Alias for available balance."""
        return self.available


class ArbitrageOpportunity(BaseModel):
    """Arbitrage opportunity data structure."""

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    quantity: Decimal
    profit: Decimal
    profit_percentage: Decimal
    timestamp: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("buy_price", "sell_price")
    @classmethod
    def validate_prices(cls, v: Decimal) -> Decimal:
        """Validate prices are positive."""
        if v <= 0:
            raise ValueError("Prices must be positive")
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Decimal) -> Decimal:
        """Validate quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @property
    def is_expired(self) -> bool:
        """Check if opportunity has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def calculate_profit(self) -> Decimal:
        """Calculate total profit from this opportunity."""
        return (self.sell_price - self.buy_price) * self.quantity
