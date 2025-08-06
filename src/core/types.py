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


# Risk Management Types (P-008)
class RiskLevel(Enum):
    """Risk level enumeration for portfolio risk assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizeMethod(Enum):
    """Position sizing method enumeration."""
    FIXED_PCT = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class RiskMetrics(BaseModel):
    """Risk metrics for portfolio risk assessment."""
    var_1d: Decimal = Field(description="1-day Value at Risk")
    var_5d: Decimal = Field(description="5-day Value at Risk")
    expected_shortfall: Decimal = Field(description="Expected shortfall (Conditional VaR)")
    max_drawdown: Decimal = Field(description="Maximum historical drawdown")
    sharpe_ratio: Optional[Decimal] = Field(default=None, description="Sharpe ratio")
    current_drawdown: Decimal = Field(default=Decimal("0"), description="Current drawdown")
    risk_level: RiskLevel = Field(description="Current risk level")
    timestamp: datetime = Field(description="Timestamp of risk calculation")


class PositionLimits(BaseModel):
    """Position limits for risk management."""
    max_position_size: Decimal = Field(description="Maximum position size")
    max_positions_per_symbol: int = Field(default=1, description="Maximum positions per symbol")
    max_total_positions: int = Field(default=10, description="Maximum total positions")
    max_portfolio_exposure: Decimal = Field(default=Decimal("0.95"), description="Maximum portfolio exposure (95%)")
    max_sector_exposure: Decimal = Field(default=Decimal("0.25"), description="Maximum sector exposure (25%)")
    max_correlation_exposure: Decimal = Field(default=Decimal("0.5"), description="Maximum correlated exposure (50%)")
    max_leverage: Decimal = Field(default=Decimal("1.0"), description="Maximum leverage (no leverage by default)")


# Circuit Breaker Types (P-009)
class CircuitBreakerStatus(Enum):
    """Circuit breaker status enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerType(Enum):
    """Circuit breaker type enumeration."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    MODEL_CONFIDENCE = "model_confidence"
    SYSTEM_ERROR_RATE = "system_error_rate"
    MANUAL_TRIGGER = "manual_trigger"


class CircuitBreakerEvent(BaseModel):
    """Circuit breaker event record."""
    trigger_type: CircuitBreakerType
    threshold: Decimal
    actual_value: Decimal
    timestamp: datetime
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)