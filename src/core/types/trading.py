"""Trading-related types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Decimal) -> Decimal:
        """Validate quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate price is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class OrderResponse(BaseModel):
    """Response from order creation."""

    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    exchange: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    """Complete order information."""

    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    time_in_force: TimeInForce
    created_at: datetime
    updated_at: Optional[datetime] = None
    exchange: str
    fees: Decimal = Decimal("0")
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal("0")
    opened_at: datetime
    closed_at: Optional[datetime] = None
    exchange: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    metadata: Dict[str, Any] = Field(default_factory=dict)


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