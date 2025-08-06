"""
Core type definitions for the trading bot framework.

This module contains all fundamental data structures used throughout the system.
These types will be extended by subsequent prompts as new functionality is added.

CRITICAL: This file will be updated by subsequent prompts. Use exact types from @COMMON_PATTERNS.md.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, field_validator


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


# Market Regime Types (P-010)
class MarketRegime(Enum):
    """Market regime enumeration for classification."""
    LOW_VOLATILITY = "low_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_CORRELATION = "high_correlation"
    LOW_CORRELATION = "low_correlation"
    CRISIS = "crisis"


class RegimeChangeEvent(BaseModel):
    """Regime change event record."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    trigger_metrics: Dict[str, Any] = Field(default_factory=dict)
    description: str


class AllocationStrategy(Enum):
    """Capital allocation strategy enumeration."""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    RISK_PARITY = "risk_parity"
    DYNAMIC = "dynamic"


class CapitalAllocation(BaseModel):
    """Capital allocation record for strategies and exchanges."""
    strategy_id: str = Field(description="Strategy identifier")
    exchange: str = Field(description="Exchange name")
    allocated_amount: Decimal = Field(description="Total allocated capital")
    utilized_amount: Decimal = Field(default=Decimal("0"), description="Currently utilized capital")
    available_amount: Decimal = Field(description="Available capital for trading")
    allocation_percentage: float = Field(description="Allocation percentage of total capital")
    last_rebalance: datetime = Field(description="Last rebalancing timestamp")
    
    @field_validator('allocation_percentage')
    @classmethod
    def validate_allocation_percentage(cls, v):
        """Validate allocation percentage is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Allocation percentage must be between 0 and 1, got {v}')
        return v


class FundFlow(BaseModel):
    """Fund flow record for capital movements."""
    from_strategy: Optional[str] = Field(default=None, description="Source strategy")
    to_strategy: Optional[str] = Field(default=None, description="Target strategy")
    from_exchange: Optional[str] = Field(default=None, description="Source exchange")
    to_exchange: Optional[str] = Field(default=None, description="Target exchange")
    amount: Decimal = Field(description="Flow amount")
    currency: str = Field(default="USDT", description="Currency of flow")
    reason: str = Field(description="Flow reason")
    timestamp: datetime = Field(description="Flow timestamp")
    converted_amount: Optional[Decimal] = Field(default=None, description="Converted amount for currency flows")
    exchange_rate: Optional[Decimal] = Field(default=None, description="Exchange rate for currency conversion")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate flow amount is positive."""
        if v <= 0:
            raise ValueError(f'Fund flow amount must be positive, got {v}')
        return v


class CapitalMetrics(BaseModel):
    """Capital management metrics."""
    total_capital: Decimal = Field(description="Total available capital")
    allocated_capital: Decimal = Field(description="Total allocated capital")
    available_capital: Decimal = Field(description="Unallocated capital")
    utilization_rate: float = Field(description="Capital utilization rate")
    allocation_efficiency: float = Field(
        ge=0.0, 
        le=3.0,  # Allow up to 300% efficiency for exceptional performance
        description="Real allocation efficiency based on utilization, performance, and market conditions"
    )
    rebalance_frequency_hours: int = Field(description="Rebalancing frequency in hours")
    emergency_reserve: Decimal = Field(description="Emergency reserve amount")
    last_updated: datetime = Field(description="Last metrics update timestamp")
    allocation_count: int = Field(description="Number of active allocations")
    
    @field_validator('utilization_rate')
    @classmethod
    def validate_utilization_rate(cls, v):
        """Validate utilization rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Utilization rate must be between 0 and 1, got {v}')
        return v
    
    @field_validator('allocation_efficiency')
    @classmethod
    def validate_allocation_efficiency(cls, v):
        """Validate allocation efficiency is between 0 and 3 (300% efficiency allowed)."""
        if not 0.0 <= v <= 3.0:
            raise ValueError(f'Allocation efficiency must be between 0 and 3, got {v}')
        return v


class CurrencyExposure(BaseModel):
    """Currency exposure tracking."""
    currency: str = Field(description="Currency code")
    total_exposure: Decimal = Field(description="Total exposure in this currency")
    base_currency_equivalent: Decimal = Field(description="Equivalent in base currency")
    exposure_percentage: float = Field(description="Percentage of total portfolio")
    hedging_required: bool = Field(description="Whether hedging is required")
    hedge_amount: Decimal = Field(default=Decimal("0"), description="Amount to hedge")
    timestamp: datetime = Field(description="Exposure calculation timestamp")
    
    @field_validator('exposure_percentage')
    @classmethod
    def validate_exposure_percentage(cls, v):
        """Validate exposure percentage is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Exposure percentage must be between 0 and 1, got {v}')
        return v


class ExchangeAllocation(BaseModel):
    """Exchange-specific capital allocation."""
    exchange: str = Field(description="Exchange name")
    allocated_amount: Decimal = Field(description="Allocated capital")
    available_amount: Decimal = Field(description="Available capital")
    utilization_rate: float = Field(description="Utilization rate")
    liquidity_score: float = Field(description="Liquidity score (0-1)")
    fee_efficiency: float = Field(description="Fee efficiency score (0-1)")
    reliability_score: float = Field(description="API reliability score (0-1)")
    last_rebalance: datetime = Field(description="Last rebalancing timestamp")
    
    @field_validator('utilization_rate', 'liquidity_score', 'fee_efficiency', 'reliability_score')
    @classmethod
    def validate_score_fields(cls, v):
        """Validate score fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Score must be between 0 and 1, got {v}')
        return v


class WithdrawalRule(BaseModel):
    """Withdrawal rule configuration."""
    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    threshold: Optional[float] = Field(default=None, description="Performance threshold")
    min_amount: Optional[Decimal] = Field(default=None, description="Minimum withdrawal amount")
    max_percentage: Optional[float] = Field(default=None, description="Maximum withdrawal percentage")
    cooldown_hours: Optional[int] = Field(default=None, description="Cooldown period in hours")
    
    @field_validator('threshold', 'max_percentage')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v


class CapitalProtection(BaseModel):
    """Capital protection configuration."""
    emergency_reserve_pct: float = Field(default=0.1, description="Emergency reserve percentage")
    max_daily_loss_pct: float = Field(default=0.05, description="Maximum daily loss percentage")
    max_weekly_loss_pct: float = Field(default=0.10, description="Maximum weekly loss percentage")
    max_monthly_loss_pct: float = Field(default=0.15, description="Maximum monthly loss percentage")
    profit_lock_pct: float = Field(default=0.5, description="Profit lock percentage")
    auto_compound_enabled: bool = Field(default=True, description="Enable auto-compounding")
    auto_compound_frequency: str = Field(default="weekly", description="Auto-compound frequency")
    profit_threshold: Decimal = Field(default=Decimal("100"), description="Minimum profit for compounding")
    
    @field_validator('emergency_reserve_pct', 'max_daily_loss_pct', 'max_weekly_loss_pct', 
                    'max_monthly_loss_pct', 'profit_lock_pct')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v

class StrategyType(Enum):
    """Strategy type enumeration for different trading strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic" 
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"
    AI_ML = "ai_ml"

class StrategyStatus(Enum):
    """Strategy status enumeration for tracking strategy state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class StrategyConfig(BaseModel):
    """Strategy configuration model for storing strategy parameters."""
    name: str = Field(description="Strategy name")
    strategy_type: StrategyType = Field(description="Strategy type")
    enabled: bool = Field(default=True, description="Whether strategy is enabled")
    symbols: List[str] = Field(description="Trading symbols")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum signal confidence")
    max_positions: int = Field(default=5, ge=1, description="Maximum positions")
    position_size_pct: float = Field(default=0.02, ge=0.001, le=0.1, description="Position size percentage")
    stop_loss_pct: float = Field(default=0.02, ge=0.001, le=0.1, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.04, ge=0.001, le=0.2, description="Take profit percentage")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    
    @field_validator('min_confidence', 'position_size_pct', 'stop_loss_pct', 'take_profit_pct')
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Percentage must be between 0 and 1, got {v}')
        return v
    
    @field_validator('max_positions')
    @classmethod
    def validate_positive_integer(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f'Value must be positive, got {v}')
        return v

class StrategyMetrics(BaseModel):
    """Strategy performance metrics model."""
    total_trades: int = Field(default=0, description="Total number of trades")
    winning_trades: int = Field(default=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    win_rate: float = Field(default=0.0, description="Win rate percentage")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last metrics update timestamp")
    
    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v):
        """Validate win rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f'Win rate must be between 0 and 1, got {v}')
        return v