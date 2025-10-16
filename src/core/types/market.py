"""Market data types for the T-Bot trading system."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExchangeStatus(Enum):
    """Exchange operational status."""

    ONLINE = "online"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class MarketData(BaseModel):
    """Market data snapshot."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    trades_count: int | None = None
    vwap: Decimal | None = None
    exchange: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional bid/ask for ticker-like data
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None

    # Legacy attribute compatibility properties
    @property
    def price(self) -> Decimal:
        """Alias for close price for backward compatibility."""
        return self.close

    @property
    def high_price(self) -> Decimal:
        """Alias for high for backward compatibility."""
        return self.high

    @property
    def low_price(self) -> Decimal:
        """Alias for low for backward compatibility."""
        return self.low

    @property
    def open_price(self) -> Decimal:
        """Alias for open for backward compatibility."""
        return self.open

    @property
    def close_price(self) -> Decimal:
        """Alias for close for backward compatibility."""
        return self.close

    @property
    def bid(self) -> Decimal | None:
        """Alias for bid_price for backward compatibility."""
        return self.bid_price

    @property
    def ask(self) -> Decimal | None:
        """Alias for ask_price for backward compatibility."""
        return self.ask_price


class Ticker(BaseModel):
    """Market ticker information."""

    symbol: str
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    last_price: Decimal
    last_quantity: Decimal | None = None
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    timestamp: datetime
    exchange: str
    price_change: Decimal | None = None
    price_change_percent: Decimal | None = None

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_percent(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        mid_price = (self.bid_price + self.ask_price) / Decimal("2")
        if mid_price == 0:
            return Decimal("0")
        return ((self.spread / mid_price) * Decimal("100")).quantize(Decimal("0.0001"))


class OrderBookLevel(BaseModel):
    """Single level in order book."""

    price: Decimal
    quantity: Decimal
    order_count: int | None = None


class OrderBook(BaseModel):
    """Order book snapshot."""

    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime
    exchange: str
    sequence: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def best_bid(self) -> OrderBookLevel | None:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        """Calculate spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    def get_depth(self, side: str, levels: int = 5) -> Decimal:
        """Calculate cumulative depth for given number of levels."""
        book_side = self.bids if side.lower() == "bid" else self.asks
        total = sum(level.quantity for level in book_side[:levels])
        return total if total else Decimal("0")


class Trade(BaseModel):
    """Represents a trade executed on an exchange."""

    id: str
    symbol: str
    exchange: str
    side: str  # "BUY" or "SELL"
    price: Decimal
    quantity: Decimal
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    maker: bool = False
    fee: Decimal | None = None


class ExchangeGeneralInfo(BaseModel):
    """General exchange information and capabilities."""

    name: str
    supported_symbols: list[str] = Field(default_factory=list)
    rate_limits: dict[str, int] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    api_version: str | None = None
    status: ExchangeStatus = ExchangeStatus.ONLINE
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExchangeInfo(BaseModel):
    """Exchange trading rules and information for a specific symbol."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal
    min_notional: Decimal | None = None
    exchange: str
    is_trading: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to valid tick size."""
        return (price // self.tick_size) * self.tick_size

    def round_quantity(self, quantity: Decimal) -> Decimal:
        """Round quantity to valid step size."""
        return (quantity // self.step_size) * self.step_size

    def validate_order(self, price: Decimal, quantity: Decimal) -> bool:
        """Validate if order parameters are within exchange limits."""
        if price < self.min_price or price > self.max_price:
            return False
        if quantity < self.min_quantity or quantity > self.max_quantity:
            return False
        if self.min_notional and (price * quantity) < self.min_notional:
            return False
        return True
