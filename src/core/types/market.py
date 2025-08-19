"""Market data types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

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
    quote_volume: Optional[Decimal] = None
    trades_count: Optional[int] = None
    vwap: Optional[Decimal] = None
    exchange: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Ticker(BaseModel):
    """Market ticker information."""

    symbol: str
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    last_price: Decimal
    last_quantity: Optional[Decimal] = None
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None
    timestamp: datetime
    exchange: str
    price_change: Optional[Decimal] = None
    price_change_percent: Optional[float] = None

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid_price = (self.bid_price + self.ask_price) / 2
        if mid_price == 0:
            return 0.0
        return float((self.spread / mid_price) * 100)


class OrderBookLevel(BaseModel):
    """Single level in order book."""

    price: Decimal
    quantity: Decimal
    order_count: Optional[int] = None


class OrderBook(BaseModel):
    """Order book snapshot."""

    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    exchange: str
    sequence: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    def get_depth(self, side: str, levels: int = 5) -> Decimal:
        """Calculate cumulative depth for given number of levels."""
        book_side = self.bids if side.lower() == "bid" else self.asks
        return sum(level.quantity for level in book_side[:levels])


class ExchangeInfo(BaseModel):
    """Exchange trading rules and information."""

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
    min_notional: Optional[Decimal] = None
    exchange: str
    is_trading: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

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