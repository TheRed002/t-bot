"""
Core type definitions for the trading bot framework.

This module contains all fundamental data structures used throughout the system.
These types will be extended by subsequent prompts as new functionality is added.

CRITICAL: This file will be updated by subsequent prompts. Use exact types from @COMMON_PATTERNS.md.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TradingMode(Enum):
    """Trading mode enumeration for different execution environments.

    Attributes:
        LIVE: Live trading with real money
        PAPER: Paper trading for testing
        BACKTEST: Historical backtesting mode
    """

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


# Data validation and quality enums
class ValidationLevel(Enum):
    """Data validation severity levels used across quality and pipeline validation"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationResult(Enum):
    """Data validation result enumeration"""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class QualityLevel(Enum):
    """Data quality level enumeration for quality monitoring"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class DriftType(Enum):
    """Data drift type enumeration for quality monitoring"""

    CONCEPT_DRIFT = "concept_drift"
    COVARIATE_DRIFT = "covariate_drift"
    LABEL_DRIFT = "label_drift"
    DISTRIBUTION_DRIFT = "distribution_drift"


# Data pipeline enums
class IngestionMode(Enum):
    """Data ingestion mode enumeration"""

    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"


class PipelineStatus(Enum):
    """Pipeline status enumeration"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


class ProcessingStep(Enum):
    """Data processing step enumeration"""

    NORMALIZE = "normalize"
    ENRICH = "enrich"
    AGGREGATE = "aggregate"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    FILTER = "filter"


class StorageMode(Enum):
    """Storage operation mode enumeration"""

    REAL_TIME = "real_time"
    BATCH = "batch"
    BUFFER = "buffer"


class Signal(BaseModel):
    """Trading signal with direction, confidence, and metadata.

    Attributes:
        direction: Signal direction (buy/sell/hold)
        confidence: Signal confidence level (0.0 to 1.0)
        timestamp: When the signal was generated
        symbol: Trading symbol for the signal
        strategy_name: Name of strategy that generated signal
        metadata: Additional signal metadata
    """

    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence between 0 and 1")
    timestamp: datetime
    symbol: str
    strategy_name: str
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional signal metadata")


class MarketData(BaseModel):
    """Market data structure for price and volume information.

    Attributes:
        symbol: Trading symbol
        price: Current market price
        volume: Trading volume
        timestamp: Data timestamp
        bid: Bid price (optional)
        ask: Ask price (optional)
        open_price: Opening price (optional)
        high_price: High price (optional)
        low_price: Low price (optional)
    """

    # Allow potentially invalid raw values; downstream validators (DataValidator)
    # enforce correctness so tests can construct edge cases.
    symbol: str = Field(description="Trading symbol")
    price: Decimal = Field(description="Current price (raw value; may be validated elsewhere)")
    volume: Decimal = Field(description="Volume (raw value; may be validated elsewhere)")
    timestamp: datetime
    bid: Decimal | None = Field(default=None, description="Bid price (raw value)")
    ask: Decimal | None = Field(default=None, description="Ask price (raw value)")
    open_price: Decimal | None = Field(default=None, description="Open price (raw value)")
    high_price: Decimal | None = Field(default=None, description="High price (raw value)")
    low_price: Decimal | None = Field(default=None, description="Low price (raw value)")


class OrderRequest(BaseModel):
    """Order request structure for placing trades."""

    # Allow raw values to enable downstream validation in exchange/risk layers
    symbol: str = Field(description="Trading symbol (validated downstream)")
    side: OrderSide
    order_type: OrderType
    # Allow raw values for testing; risk layer enforces positivity
    quantity: Decimal = Field(description="Order quantity (validated in risk layer)")
    price: Decimal | None = Field(
        default=None, gt=0, description="Order price (must be positive if set)"
    )
    stop_price: Decimal | None = Field(
        default=None, gt=0, description="Stop price (must be positive if set)"
    )
    time_in_force: str = Field(
        default="GTC", pattern=r"^(GTC|IOC|FOK)$", description="Time in force"
    )
    client_order_id: str | None = None


class OrderResponse(BaseModel):
    """Order response structure from exchange."""

    id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None
    filled_quantity: Decimal = Decimal("0")
    status: str
    timestamp: datetime


class Position(BaseModel):
    """Position structure for tracking open positions.

    Attributes:
        symbol: Trading symbol
        quantity: Position size (must be positive)
        entry_price: Entry price (must be positive)
        current_price: Current market price (must be positive)
        unrealized_pnl: Unrealized profit/loss
        side: Position side (buy/sell)
        timestamp: Position timestamp
        metadata: Additional position metadata
    """

    symbol: str = Field(min_length=3, description="Trading symbol")
    quantity: Decimal = Field(gt=0, description="Position quantity (must be positive)")
    entry_price: Decimal = Field(gt=0, description="Entry price (must be positive)")
    current_price: Decimal = Field(gt=0, description="Current price (must be positive)")
    unrealized_pnl: Decimal
    side: OrderSide
    timestamp: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional position metadata"
    )


# TODO:REVERSE INTEGRATION POINTS: Future prompts will add:
# - P-011: Strategy types (StrategyConfig, StrategyStatus)
# - P-017: ML types (ModelPrediction, ModelMetadata)


class ExchangeInfo(BaseModel):
    """Exchange information including supported symbols and features."""

    name: str
    supported_symbols: list[str]
    rate_limits: dict[str, int]
    features: list[str]
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
    bids: list[list[Decimal]]  # [[price, quantity], ...]
    asks: list[list[Decimal]]  # [[price, quantity], ...]
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


class TradeState(Enum):
    """Trade lifecycle state enumeration."""
    
    SIGNAL_GENERATED = "signal_generated"
    PRE_TRADE_VALIDATION = "pre_trade_validation"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    PARTIALLY_FILLED = "partially_filled"
    FULLY_FILLED = "fully_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    SETTLED = "settled"
    ATTRIBUTED = "attributed"


class Trade(BaseModel):
    """Trade execution record."""

    id: str = Field(min_length=1, description="Trade ID")
    symbol: str = Field(min_length=3, description="Trading symbol")
    side: OrderSide
    quantity: Decimal = Field(gt=0, description="Trade quantity (must be positive)")
    price: Decimal = Field(gt=0, description="Trade price (must be positive)")
    timestamp: datetime
    fee: Decimal = Field(ge=0, default=Decimal("0"), description="Trade fee (must be non-negative)")
    fee_currency: str = Field(
        default="USDT", pattern=r"^[A-Z]{3,10}$", description="Fee currency code"
    )


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
    sharpe_ratio: Decimal | None = Field(default=None, description="Sharpe ratio")
    current_drawdown: Decimal = Field(default=Decimal("0"), description="Current drawdown")
    risk_level: RiskLevel = Field(description="Current risk level")
    timestamp: datetime = Field(description="Timestamp of risk calculation")


class PositionLimits(BaseModel):
    """Position limits for risk management."""

    max_position_size: Decimal = Field(description="Maximum position size")
    max_positions_per_symbol: int = Field(default=1, description="Maximum positions per symbol")
    max_total_positions: int = Field(default=10, description="Maximum total positions")
    max_portfolio_exposure: Decimal = Field(
        default=Decimal("0.95"), description="Maximum portfolio exposure (95%)"
    )
    max_sector_exposure: Decimal = Field(
        default=Decimal("0.25"), description="Maximum sector exposure (25%)"
    )
    max_correlation_exposure: Decimal = Field(
        default=Decimal("0.5"), description="Maximum correlated exposure (50%)"
    )
    max_leverage: Decimal = Field(
        default=Decimal("1.0"), description="Maximum leverage (no leverage by default)"
    )


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
    CORRELATION_SPIKE = "correlation_spike"
    MANUAL_TRIGGER = "manual_trigger"


class CircuitBreakerEvent(BaseModel):
    """Circuit breaker event record."""

    trigger_type: CircuitBreakerType
    threshold: Decimal
    actual_value: Decimal
    timestamp: datetime
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    trigger_metrics: dict[str, Any] = Field(default_factory=dict)
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

    strategy_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9_-]+$", description="Strategy identifier"
    )
    exchange: str = Field(min_length=3, pattern=r"^[a-z]+$", description="Exchange name")
    allocated_amount: Decimal = Field(gt=0, description="Total allocated capital")
    utilized_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Currently utilized capital"
    )
    available_amount: Decimal = Field(ge=0, description="Available capital for trading")
    allocation_percentage: float = Field(description="Allocation percentage of total capital")
    last_rebalance: datetime = Field(description="Last rebalancing timestamp")

    @field_validator("allocation_percentage")
    @classmethod
    def validate_allocation_percentage(cls, v: float) -> float:
        """Validate allocation percentage is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Allocation percentage must be between 0 and 1, got {v}")
        return v


class FundFlow(BaseModel):
    """Fund flow record for capital movements."""

    from_strategy: str | None = Field(default=None, description="Source strategy")
    to_strategy: str | None = Field(default=None, description="Target strategy")
    from_exchange: str | None = Field(default=None, description="Source exchange")
    to_exchange: str | None = Field(default=None, description="Target exchange")
    amount: Decimal = Field(description="Flow amount")
    currency: str = Field(default="USDT", pattern=r"^[A-Z]{3,10}$", description="Currency code")
    reason: str = Field(min_length=1, description="Flow reason")
    timestamp: datetime = Field(description="Flow timestamp")
    converted_amount: Decimal | None = Field(
        default=None, gt=0, description="Converted amount for currency flows"
    )
    exchange_rate: Decimal | None = Field(
        default=None, gt=0, description="Exchange rate for currency conversion"
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate flow amount is positive."""
        if v <= 0:
            raise ValueError(f"Fund flow amount must be positive, got {v}")
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
        description="Real allocation efficiency based on utilization, performance, and market conditions",
    )
    rebalance_frequency_hours: int = Field(description="Rebalancing frequency in hours")
    emergency_reserve: Decimal = Field(description="Emergency reserve amount")
    last_updated: datetime = Field(description="Last metrics update timestamp")
    allocation_count: int = Field(description="Number of active allocations")

    @field_validator("utilization_rate")
    @classmethod
    def validate_utilization_rate(cls, v):
        """Validate utilization rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Utilization rate must be between 0 and 1, got {v}")
        return v

    @field_validator("allocation_efficiency")
    @classmethod
    def validate_allocation_efficiency(cls, v):
        """Validate allocation efficiency is between 0 and 3 (300% efficiency allowed)."""
        if not 0.0 <= v <= 3.0:
            raise ValueError(f"Allocation efficiency must be between 0 and 3, got {v}")
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

    @field_validator("exposure_percentage")
    @classmethod
    def validate_exposure_percentage(cls, v):
        """Validate exposure percentage is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Exposure percentage must be between 0 and 1, got {v}")
        return v


class ExchangeAllocation(BaseModel):
    """Exchange-specific capital allocation."""

    exchange: str = Field(min_length=3, pattern=r"^[a-z]+$", description="Exchange name")
    allocated_amount: Decimal = Field(gt=0, description="Allocated capital")
    available_amount: Decimal = Field(ge=0, description="Available capital")
    utilization_rate: float = Field(description="Utilization rate")
    liquidity_score: float = Field(description="Liquidity score (0-1)")
    fee_efficiency: float = Field(description="Fee efficiency score (0-1)")
    reliability_score: float = Field(description="API reliability score (0-1)")
    last_rebalance: datetime = Field(description="Last rebalancing timestamp")

    @field_validator("utilization_rate", "liquidity_score", "fee_efficiency", "reliability_score")
    @classmethod
    def validate_score_fields(cls, v):
        """Validate score fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v


class WithdrawalRule(BaseModel):
    """Withdrawal rule configuration."""

    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    threshold: float | None = Field(default=None, description="Performance threshold")
    min_amount: Decimal | None = Field(default=None, description="Minimum withdrawal amount")
    max_percentage: float | None = Field(default=None, description="Maximum withdrawal percentage")
    cooldown_hours: int | None = Field(default=None, description="Cooldown period in hours")

    @field_validator("threshold", "max_percentage")
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
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
    profit_threshold: Decimal = Field(
        default=Decimal("100"), description="Minimum profit for compounding"
    )

    @field_validator(
        "emergency_reserve_pct",
        "max_daily_loss_pct",
        "max_weekly_loss_pct",
        "max_monthly_loss_pct",
        "profit_lock_pct",
    )
    @classmethod
    def validate_percentage_fields(cls, v):
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
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

    name: str = Field(min_length=1, description="Strategy name")
    strategy_type: StrategyType = Field(description="Strategy type")
    enabled: bool = Field(default=True, description="Whether strategy is enabled")
    symbols: list[str] = Field(min_length=1, description="Trading symbols")
    timeframe: str = Field(
        default="1h", pattern=r"^(1m|5m|15m|30m|1h|4h|1d)$", description="Trading timeframe"
    )
    min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum signal confidence"
    )
    max_positions: int = Field(default=5, ge=1, description="Maximum positions")
    position_size_pct: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Position size percentage"
    )
    stop_loss_pct: float = Field(default=0.02, ge=0.001, le=0.1, description="Stop loss percentage")
    take_profit_pct: float = Field(
        default=0.04, ge=0.001, le=0.2, description="Take profit percentage"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate symbols list is non-empty and contains valid symbols."""
        if not v or len(v) == 0:
            raise ValueError("At least one symbol is required")
        for symbol in v:
            if not isinstance(symbol, str) or len(symbol) < 3:
                raise ValueError(f"Invalid symbol format: {symbol}")
        return v

    @field_validator("min_confidence", "position_size_pct", "stop_loss_pct", "take_profit_pct")
    @classmethod
    def validate_percentage_fields(cls, v: float) -> float:
        """Validate percentage fields are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v

    @field_validator("max_positions")
    @classmethod
    def validate_positive_integer(cls, v: int) -> int:
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class StrategyMetrics(BaseModel):
    """Strategy performance metrics model."""

    total_trades: int = Field(default=0, description="Total number of trades")
    winning_trades: int = Field(default=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    win_rate: float = Field(default=0.0, description="Win rate percentage")
    sharpe_ratio: float | None = Field(default=None, description="Sharpe ratio")
    max_drawdown: float | None = Field(default=None, description="Maximum drawdown")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last metrics update timestamp",
    )

    @field_validator("win_rate")
    @classmethod
    def validate_win_rate(cls, v: float) -> float:
        """Validate win rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Win rate must be between 0 and 1, got {v}")
        return v


# Data Module Enums (moved here to avoid circular dependencies)


class NewsSentiment(Enum):
    """News sentiment enumeration"""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class SocialSentiment(Enum):
    """Social media sentiment enumeration"""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


# =============================================================================
# Error Handling Types
# =============================================================================


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""

    pattern_id: str
    pattern_type: str  # frequency, correlation, trend, anomaly
    component: str
    error_type: str
    frequency: float  # errors per hour
    severity: str
    first_detected: datetime
    last_detected: datetime
    occurrence_count: int
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_action: str
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "component": self.component,
            "error_type": self.error_type,
            "frequency": self.frequency,
            "severity": self.severity,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "is_active": self.is_active,
        }


# =============================================================================
# Execution Engine Types (P-016)
# =============================================================================


class ExecutionAlgorithm(Enum):
    """Execution algorithm enumeration for different execution strategies."""

    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    SMART_ROUTER = "smart_router"
    MARKET = "market"
    LIMIT = "limit"


class ExecutionStatus(Enum):
    """Execution status enumeration for tracking execution progress."""

    PENDING = "pending"
    RUNNING = "running"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


class SlippageType(Enum):
    """Slippage type enumeration for cost analysis."""

    MARKET_IMPACT = "market_impact"
    TIMING = "timing"
    OPPORTUNITY_COST = "opportunity_cost"
    IMPLEMENTATION = "implementation"


class ExecutionResult(BaseModel):
    """Result of an execution operation with detailed metrics."""

    execution_id: str = Field(description="Unique execution identifier")
    original_order: OrderRequest = Field(description="Original order request")
    child_orders: list[OrderResponse] = Field(
        default_factory=list, description="List of child orders created during execution"
    )
    algorithm: ExecutionAlgorithm = Field(description="Execution algorithm used")
    status: ExecutionStatus = Field(description="Current execution status")
    
    # Execution metrics
    total_filled_quantity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total quantity filled"
    )
    average_fill_price: Decimal | None = Field(
        default=None, gt=0, description="Volume-weighted average fill price"
    )
    total_fees: Decimal = Field(default=Decimal("0"), ge=0, description="Total execution fees")
    
    # Timing metrics
    start_time: datetime = Field(description="Execution start time")
    end_time: datetime | None = Field(default=None, description="Execution end time")
    execution_duration: float | None = Field(
        default=None, ge=0, description="Execution duration in seconds"
    )
    
    # Slippage and cost metrics
    expected_price: Decimal | None = Field(
        default=None, gt=0, description="Expected execution price"
    )
    price_slippage: Decimal = Field(
        default=Decimal("0"), description="Price slippage (positive = adverse)"
    )
    market_impact: Decimal = Field(
        default=Decimal("0"), description="Estimated market impact"
    )
    implementation_shortfall: Decimal = Field(
        default=Decimal("0"), description="Implementation shortfall cost"
    )
    
    # Execution details
    number_of_trades: int = Field(default=0, ge=0, description="Number of individual trades")
    participation_rate: float | None = Field(
        default=None, ge=0, le=1, description="Market participation rate"
    )
    is_aggressive: bool = Field(default=False, description="Whether execution was aggressive")
    
    # Error handling
    error_message: str | None = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result creation timestamp"
    )

    @field_validator("total_filled_quantity")
    @classmethod
    def validate_filled_quantity(cls, v: Decimal) -> Decimal:
        """Validate filled quantity is non-negative."""
        if v < 0:
            raise ValueError("Filled quantity cannot be negative")
        return v

    @field_validator("participation_rate")
    @classmethod
    def validate_participation_rate(cls, v: float | None) -> float | None:
        """Validate participation rate is between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Participation rate must be between 0 and 1")
        return v


class SlippageMetrics(BaseModel):
    """Slippage analysis metrics for execution performance."""

    symbol: str = Field(description="Trading symbol")
    order_size: Decimal = Field(gt=0, description="Order size")
    market_price: Decimal = Field(gt=0, description="Market price at order time")
    execution_price: Decimal = Field(gt=0, description="Actual execution price")
    
    # Slippage calculations
    price_slippage_bps: Decimal = Field(description="Price slippage in basis points")
    market_impact_bps: Decimal = Field(description="Market impact in basis points")
    timing_cost_bps: Decimal = Field(description="Timing cost in basis points")
    total_cost_bps: Decimal = Field(description="Total transaction cost in basis points")
    
    # Market conditions
    spread_bps: Decimal = Field(description="Bid-ask spread in basis points")
    volume_ratio: float = Field(description="Order size to average daily volume ratio")
    volatility: float = Field(description="Market volatility at execution time")
    
    # Cost breakdown
    explicit_costs: Decimal = Field(
        default=Decimal("0"), description="Explicit costs (fees, commissions)"
    )
    implicit_costs: Decimal = Field(
        default=Decimal("0"), description="Implicit costs (slippage, impact)"
    )
    
    timestamp: datetime = Field(description="Analysis timestamp")

    @field_validator("volume_ratio")
    @classmethod
    def validate_volume_ratio(cls, v: float) -> float:
        """Validate volume ratio is non-negative."""
        if v < 0:
            raise ValueError("Volume ratio cannot be negative")
        return v


class ExecutionInstruction(BaseModel):
    """Instruction for executing an order with specific algorithm parameters."""

    order: OrderRequest = Field(description="Base order to execute")
    algorithm: ExecutionAlgorithm = Field(description="Execution algorithm to use")
    
    # Algorithm parameters
    time_horizon_minutes: int | None = Field(
        default=None, gt=0, description="Time horizon for execution in minutes"
    )
    participation_rate: float | None = Field(
        default=None, gt=0, le=1, description="Target market participation rate"
    )
    max_slices: int | None = Field(
        default=None, gt=0, description="Maximum number of order slices"
    )
    slice_size: Decimal | None = Field(
        default=None, gt=0, description="Fixed slice size for orders"
    )
    display_quantity: Decimal | None = Field(
        default=None, gt=0, description="Visible quantity for iceberg orders"
    )
    
    # Risk controls
    max_slippage_bps: Decimal | None = Field(
        default=None, gt=0, description="Maximum acceptable slippage in basis points"
    )
    price_tolerance_pct: float | None = Field(
        default=None, gt=0, le=1, description="Price tolerance percentage"
    )
    
    # Routing preferences
    preferred_exchanges: list[str] = Field(
        default_factory=list, description="Preferred exchanges for routing"
    )
    avoid_exchanges: list[str] = Field(
        default_factory=list, description="Exchanges to avoid"
    )
    
    # Execution options
    is_urgent: bool = Field(default=False, description="Whether execution is urgent")
    allow_partial: bool = Field(default=True, description="Allow partial fills")
    start_time: datetime | None = Field(default=None, description="Scheduled start time")
    end_time: datetime | None = Field(default=None, description="Execution deadline")
    
    # Metadata
    execution_id: str | None = Field(default=None, description="Execution tracking ID")
    strategy_name: str | None = Field(default=None, description="Originating strategy name")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )

    @field_validator("participation_rate")
    @classmethod
    def validate_participation_rate(cls, v: float | None) -> float | None:
        """Validate participation rate is between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Participation rate must be between 0 and 1")
        return v

    @field_validator("price_tolerance_pct")
    @classmethod
    def validate_price_tolerance(cls, v: float | None) -> float | None:
        """Validate price tolerance is between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Price tolerance must be between 0 and 1")
        return v


# === Bot Management Types (P-017) ===

class BotStatus(Enum):
    """Bot status enumeration for lifecycle management."""
    
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


class BotType(Enum):
    """Bot type enumeration for different bot categories."""
    
    STRATEGY = "strategy"       # Single strategy bot
    ARBITRAGE = "arbitrage"     # Cross-exchange arbitrage bot
    MARKET_MAKER = "market_maker"  # Market making bot
    HYBRID = "hybrid"           # Multi-strategy hybrid bot
    SCANNER = "scanner"         # Opportunity scanner bot


class ResourceType(Enum):
    """Resource type enumeration for resource management."""
    
    CAPITAL = "capital"
    API_RATE_LIMIT = "api_rate_limit"
    WEBSOCKET_CONNECTIONS = "websocket_connections"
    DATABASE_CONNECTIONS = "database_connections"
    CPU = "cpu"
    MEMORY = "memory"


class BotPriority(Enum):
    """Bot priority levels for resource allocation."""
    
    CRITICAL = "critical"   # Emergency or high-value bots
    HIGH = "high"          # Important production bots
    NORMAL = "normal"      # Standard trading bots
    LOW = "low"           # Experimental or backup bots


class BotConfiguration(BaseModel):
    """Bot configuration model with all parameters."""
    
    # Basic identification
    bot_id: str = Field(..., description="Unique bot identifier")
    bot_name: str = Field(..., description="Human-readable bot name")
    bot_type: BotType = Field(..., description="Bot type category")
    
    # Strategy configuration
    strategy_name: str = Field(..., description="Strategy to run")
    strategy_config: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific configuration"
    )
    
    # Exchange and symbol configuration
    exchanges: list[str] = Field(..., description="Exchanges to use")
    symbols: list[str] = Field(..., description="Trading symbols")
    
    # Resource allocation
    allocated_capital: Decimal = Field(..., gt=0, description="Allocated capital amount")
    max_position_size: Decimal = Field(..., gt=0, description="Maximum position size")
    risk_percentage: float = Field(..., gt=0, le=1, description="Risk percentage per trade")
    
    # Operational parameters
    priority: BotPriority = Field(default=BotPriority.NORMAL, description="Bot priority")
    auto_start: bool = Field(default=False, description="Auto-start on creation")
    heartbeat_interval: int = Field(default=30, ge=5, description="Heartbeat interval in seconds")
    
    # Trading parameters
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")
    max_daily_trades: int | None = Field(default=None, ge=1, description="Maximum daily trades")
    max_concurrent_positions: int = Field(default=5, ge=1, description="Maximum concurrent positions")
    
    # Metadata
    tags: list[str] = Field(default_factory=list, description="Bot tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BotMetrics(BaseModel):
    """Bot performance and operational metrics."""
    
    bot_id: str = Field(..., description="Bot identifier")
    
    # Performance metrics
    total_trades: int = Field(default=0, description="Total number of trades")
    profitable_trades: int = Field(default=0, description="Number of profitable trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total profit/loss")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized PnL")
    
    # Trading statistics
    win_rate: float = Field(default=0.0, ge=0, le=1, description="Win rate percentage")
    average_trade_pnl: Decimal = Field(default=Decimal("0"), description="Average trade PnL")
    max_drawdown: Decimal = Field(default=Decimal("0"), description="Maximum drawdown")
    sharpe_ratio: float | None = Field(default=None, description="Sharpe ratio")
    
    # Operational metrics
    uptime_percentage: float = Field(default=0.0, ge=0, le=1, description="Uptime percentage")
    error_count: int = Field(default=0, description="Number of errors encountered")
    last_heartbeat: datetime | None = Field(default=None, description="Last heartbeat timestamp")
    
    # Resource usage
    cpu_usage: float = Field(default=0.0, ge=0, description="CPU usage percentage")
    memory_usage: float = Field(default=0.0, ge=0, description="Memory usage in MB")
    api_calls_count: int = Field(default=0, description="API calls made")
    
    # Timestamps
    start_time: datetime | None = Field(default=None, description="Bot start time")
    last_trade_time: datetime | None = Field(default=None, description="Last trade timestamp")
    metrics_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Metrics last update time"
    )


class BotState(BaseModel):
    """Bot state information for persistence and recovery."""
    
    bot_id: str = Field(..., description="Bot identifier")
    status: BotStatus = Field(..., description="Current bot status")
    
    # Current positions and orders
    open_positions: list[dict[str, Any]] = Field(
        default_factory=list, description="Current open positions"
    )
    pending_orders: list[dict[str, Any]] = Field(
        default_factory=list, description="Pending orders"
    )
    
    # Strategy state
    strategy_state: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific state data"
    )
    
    # Execution state
    last_execution_id: str | None = Field(default=None, description="Last execution ID")
    execution_queue: list[dict[str, Any]] = Field(
        default_factory=list, description="Queued executions"
    )
    
    # Resource usage
    allocated_capital: Decimal = Field(..., description="Currently allocated capital")
    used_capital: Decimal = Field(default=Decimal("0"), description="Capital in use")
    reserved_capital: Decimal = Field(default=Decimal("0"), description="Reserved capital")
    
    # Timestamps and versioning
    state_version: int = Field(default=1, description="State version for conflict resolution")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last state update time"
    )
    checkpoint_created: datetime | None = Field(
        default=None, description="Last checkpoint creation time"
    )


class ResourceAllocation(BaseModel):
    """Resource allocation model for bot resource management."""
    
    bot_id: str = Field(..., description="Bot identifier")
    resource_type: ResourceType = Field(..., description="Type of resource")
    
    # Allocation details
    allocated_amount: Decimal = Field(..., ge=0, description="Allocated resource amount")
    used_amount: Decimal = Field(default=Decimal("0"), ge=0, description="Currently used amount")
    reserved_amount: Decimal = Field(default=Decimal("0"), ge=0, description="Reserved amount")
    
    # Limits and constraints
    max_amount: Decimal | None = Field(default=None, description="Maximum allowed amount")
    min_amount: Decimal = Field(default=Decimal("0"), description="Minimum required amount")
    
    # Usage tracking
    peak_usage: Decimal = Field(default=Decimal("0"), description="Peak usage recorded")
    average_usage: Decimal = Field(default=Decimal("0"), description="Average usage")
    last_usage_time: datetime | None = Field(default=None, description="Last usage timestamp")
    
    # Metadata
    allocation_priority: BotPriority = Field(
        default=BotPriority.NORMAL, description="Allocation priority"
    )
    expires_at: datetime | None = Field(default=None, description="Allocation expiry time")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Allocation creation time"
    )


class BotEvent(BaseModel):
    """Bot event model for tracking bot lifecycle and actions."""
    
    event_id: str = Field(..., description="Unique event identifier")
    bot_id: str = Field(..., description="Bot identifier")
    event_type: str = Field(..., description="Type of event")
    
    # Event details
    event_data: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )
    severity: ValidationLevel = Field(
        default=ValidationLevel.MEDIUM, description="Event severity"
    )
    
    # Context
    triggered_by: str | None = Field(default=None, description="Event trigger source")
    correlation_id: str | None = Field(default=None, description="Correlation identifier")
    
    # Timestamps
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    processed_at: datetime | None = Field(default=None, description="Processing timestamp")
