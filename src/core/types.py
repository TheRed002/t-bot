"""
Core type definitions for the trading bot framework.

This module contains all fundamental data structures used throughout the system.
These types will be extended by subsequent prompts as new functionality is added.

CRITICAL: This file will be updated by subsequent prompts. Use exact types from @COMMON_PATTERNS.md.
"""

from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class TradingMode(Enum):
    """Trading mode enumeration for different execution environments."""
    LIVE = "live"
    PAPER = "paper" 
    BACKTEST = "backtest"


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


class ExchangeType(Enum):
    """Exchange types for rate limiting and coordination."""
    BINANCE = "binance"
    OKX = "okx"
    COINBASE = "coinbase"


class RequestType(Enum):
    """Request types for global coordination and rate limiting."""
    MARKET_DATA = "market_data"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    BALANCE_QUERY = "balance_query"
    POSITION_QUERY = "position_query"
    HISTORICAL_DATA = "historical_data"
    WEBSOCKET_CONNECTION = "websocket_connection"


class ConnectionType(Enum):
    """Connection types for different stream types."""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    USER_DATA = "user_data"
    MARKET_DATA = "market_data"


class Signal(BaseModel):
    """Trading signal with direction, confidence, and metadata."""
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence between 0 and 1")
    timestamp: datetime
    symbol: str
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal metadata")


class MarketData(BaseModel):
    """Market data structure for price and volume information."""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None


class OrderRequest(BaseModel):
    """Order request structure for placing trades."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    client_order_id: Optional[str] = None


class OrderResponse(BaseModel):
    """Order response structure from exchange."""
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    filled_quantity: Decimal = Decimal("0")
    status: str
    timestamp: datetime


class Position(BaseModel):
    """Position structure for tracking open positions."""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    side: OrderSide
    timestamp: datetime


# TODO:REVERSE INTEGRATION POINTS: Future prompts will add:
# - P-008: Risk types (RiskMetrics, PositionLimits)  
# - P-011: Strategy types (StrategyConfig, StrategyStatus)
# - P-017: ML types (ModelPrediction, ModelMetadata) 

class ExchangeInfo(BaseModel):
    """Exchange information including supported symbols and features."""
    name: str
    supported_symbols: List[str]
    rate_limits: Dict[str, int]
    features: List[str]
    api_version: str

class Ticker(BaseModel):
    """Real-time ticker information for a symbol."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume_24h: Decimal
    price_change_24h: Decimal
    timestamp: datetime

class OrderBook(BaseModel):
    """Order book with bid and ask levels."""
    symbol: str
    bids: List[List[Decimal]]  # [[price, quantity], ...]
    asks: List[List[Decimal]]  # [[price, quantity], ...]
    timestamp: datetime

class ExchangeStatus(Enum):
    """Exchange connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

class Trade(BaseModel):
    """Trade execution record."""
    id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal = Decimal("0")
    fee_currency: str = "USDT" 