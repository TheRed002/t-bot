"""Risk management types for the T-Bot trading system."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizeMethod(Enum):
    """Position sizing methodology."""

    FIXED = "fixed"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ATR_BASED = "atr_based"
    OPTIMAL_F = "optimal_f"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"


class CircuitBreakerStatus(Enum):
    """Circuit breaker status."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


class CircuitBreakerType(Enum):
    """Circuit breaker trigger type."""

    LOSS_LIMIT = "loss_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN = "drawdown"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY = "volatility"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION = "correlation"
    CORRELATION_SPIKE = "correlation_spike"
    MODEL_CONFIDENCE = "model_confidence"
    SYSTEM_ERROR_RATE = "system_error_rate"
    TECHNICAL = "technical"
    MANUAL = "manual"


class EmergencyAction(Enum):
    """Emergency action types."""

    CLOSE_ALL_POSITIONS = "close_all_positions"
    BLOCK_NEW_ORDERS = "block_new_orders"
    CANCEL_PENDING_ORDERS = "cancel_pending_orders"
    REDUCE_POSITION_SIZES = "reduce_position_sizes"
    SWITCH_TO_SAFE_MODE = "switch_to_safe_mode"
    MANUAL_OVERRIDE = "manual_override"
    VOLATILITY_LIMIT = "volatility_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION = "correlation"
    CORRELATION_LIMIT = "correlation_limit"
    CORRELATION_SPIKE = "correlation_spike"
    TECHNICAL = "technical"
    MANUAL = "manual"
    MANUAL_TRIGGER = "manual_trigger"
    MODEL_CONFIDENCE = "model_confidence"
    SYSTEM_ERROR_RATE = "system_error_rate"


class AllocationStrategy(Enum):
    """Capital allocation strategy."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM_BASED = "momentum_based"
    VOLATILITY_INVERSE = "volatility_inverse"
    CUSTOM = "custom"


class RiskMetrics(BaseModel):
    """Risk metrics for positions and strategies."""

    # Portfolio metrics
    portfolio_value: Decimal
    total_exposure: Decimal
    var_1d: Decimal  # 1-day Value at Risk
    var_5d: Decimal | None = None  # 5-day Value at Risk
    expected_shortfall: Decimal | None = None

    # Performance metrics
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None
    calmar_ratio: Decimal | None = None

    # Drawdown metrics
    max_drawdown: Decimal | None = None
    current_drawdown: Decimal | None = None

    # Position metrics
    position_count: int | None = None
    correlation_risk: Decimal | None = None
    liquidity_score: Decimal | None = None

    # Risk assessment
    risk_level: RiskLevel

    # Legacy fields (kept for compatibility)
    position_id: str | None = None
    strategy_id: str | None = None
    symbol: str | None = None
    position_value: Decimal | None = None
    position_risk: Decimal | None = None
    var_95: Decimal | None = None
    var_99: Decimal | None = None
    delta: Decimal | None = None
    gamma: Decimal | None = None
    vega: Decimal | None = None
    theta: Decimal | None = None
    beta: Decimal | None = None
    correlation: Decimal | None = None
    information_ratio: Decimal | None = None
    drawdown_duration: int | None = None
    gross_exposure: Decimal | None = None
    net_exposure: Decimal | None = None
    leverage: Decimal | None = None

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PositionLimits(BaseModel):
    """Position size and risk limits."""

    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    min_position_size: Decimal

    # Legacy fields (kept for compatibility)
    symbol: str | None = None
    max_position_value: Decimal | None = None
    max_loss_per_trade: Decimal | None = None
    max_daily_loss: Decimal | None = None
    max_drawdown: Decimal | None = None
    concentration_limit: Decimal | None = None  # Max % of portfolio

    # Time-based limits
    max_trades_per_day: int | None = None
    max_trades_per_hour: int | None = None
    min_time_between_trades: int | None = None  # seconds

    # Conditional limits
    reduce_size_on_loss: bool = False
    loss_reduction_factor: Decimal = Decimal("1.0")

    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskLimits(BaseModel):
    """Risk limits configuration."""

    max_position_size: Decimal
    max_portfolio_risk: Decimal
    max_correlation: Decimal
    max_leverage: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    max_positions: int
    min_liquidity_ratio: Decimal

    # Optional additional limits
    max_var_limit: Decimal | None = None
    max_concentration: Decimal | None = None
    min_sharpe_ratio: Decimal | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskAlert(BaseModel):
    """Risk alert notification."""

    timestamp: datetime
    event_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    data: dict[str, Any]

    # Optional fields
    symbol: str | None = None
    strategy_id: str | None = None
    action_taken: str | None = None
    resolved_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class CircuitBreakerEvent(BaseModel):
    """Circuit breaker trigger event."""

    breaker_id: str
    breaker_type: CircuitBreakerType
    status: CircuitBreakerStatus
    triggered_at: datetime
    trigger_value: Decimal
    threshold_value: Decimal
    cooldown_period: int  # seconds
    reset_at: datetime | None = None
    affected_symbols: list[str] = Field(default_factory=list)
    affected_strategies: list[str] = Field(default_factory=list)
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapitalAllocation(BaseModel):
    """Capital allocation for strategies and positions."""

    allocation_id: str
    strategy_id: str | None = None
    symbol: str | None = None
    allocated_amount: Decimal
    utilized_amount: Decimal
    available_amount: Decimal
    allocation_percentage: Decimal
    target_allocation_pct: Decimal
    min_allocation: Decimal
    max_allocation: Decimal
    last_rebalance: datetime
    next_rebalance: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FundFlow(BaseModel):
    """Fund flow tracking for deposits and withdrawals."""

    flow_id: str
    flow_type: str  # "deposit" or "withdrawal"
    amount: Decimal
    currency: str
    exchange: str
    status: str
    requested_at: datetime
    completed_at: datetime | None = None
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapitalMetrics(BaseModel):
    """Overall capital and portfolio metrics."""

    total_capital: Decimal
    allocated_amount: Decimal
    available_amount: Decimal
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal

    # Return metrics
    daily_return: Decimal
    weekly_return: Decimal
    monthly_return: Decimal
    yearly_return: Decimal
    total_return: Decimal

    # Risk-adjusted returns
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal

    # Risk metrics
    current_drawdown: Decimal
    max_drawdown: Decimal
    var_95: Decimal
    expected_shortfall: Decimal

    # Allocation metrics
    strategies_active: int
    positions_open: int
    leverage_used: Decimal

    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class CurrencyExposure(BaseModel):
    """Currency exposure tracking."""

    currency: str
    exposure_amount: Decimal
    exposure_pct: Decimal
    hedge_amount: Decimal = Decimal("0")
    net_exposure: Decimal
    exchange_rate: Decimal
    base_currency: str
    updated_at: datetime


class ExchangeAllocation(BaseModel):
    """Capital allocation across exchanges."""

    exchange: str
    allocated_amount: Decimal
    utilized_amount: Decimal
    available_amount: Decimal
    allocation_percentage: Decimal
    num_positions: int
    total_pnl: Decimal
    last_activity: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class WithdrawalRule(BaseModel):
    """Automated withdrawal rules."""

    rule_id: str
    name: str
    enabled: bool = True
    trigger_type: str  # "profit_target", "time_based", "drawdown"
    trigger_value: Any
    withdrawal_pct: Decimal
    min_withdrawal: Decimal
    max_withdrawal: Decimal
    destination: str
    last_triggered: datetime | None = None
    next_check: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapitalProtection(BaseModel):
    """Capital protection settings."""

    protection_id: str
    enabled: bool = True
    min_capital_threshold: Decimal
    stop_trading_threshold: Decimal
    reduce_size_threshold: Decimal
    size_reduction_factor: Decimal
    max_daily_loss: Decimal
    max_weekly_loss: Decimal
    max_monthly_loss: Decimal
    emergency_liquidation: bool = False
    emergency_threshold: Decimal
    notification_settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioState(BaseModel):
    """Complete portfolio state representation."""

    total_value: Decimal = Field(..., ge=0, description="Total portfolio value")
    available_cash: Decimal = Field(..., description="Available cash balance")
    total_positions_value: Decimal = Field(..., ge=0, description="Total value of all positions")
    unrealized_pnl: Decimal = Field(..., description="Unrealized profit/loss")
    realized_pnl: Decimal = Field(..., description="Realized profit/loss")
    positions: dict[str, Any] = Field(
        default_factory=dict, description="Current positions by symbol"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="State timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioMetrics(BaseModel):
    """Unified portfolio metrics model for cross-module consistency."""

    # Core metrics
    total_value: Decimal = Field(default=Decimal("0"), ge=0)
    total_exposure: Decimal = Field(default=Decimal("0"), ge=0)
    total_pnl: Decimal = Field(default=Decimal("0"))
    unrealized_pnl: Decimal = Field(default=Decimal("0"))
    realized_pnl: Decimal = Field(default=Decimal("0"))

    # Portfolio composition
    position_count: int = Field(default=0, ge=0)
    cash: Decimal = Field(default=Decimal("0"), ge=0)
    invested_capital: Decimal = Field(default=Decimal("0"), ge=0)

    # Risk metrics
    leverage: Decimal = Field(default=Decimal("1.0"), ge=1)
    volatility: Decimal = Field(default=Decimal("0"), ge=0)
    beta: Decimal | None = None
    correlation_btc: Decimal | None = None
    largest_position_weight: Decimal = Field(default=Decimal("0"), ge=0, le=1)

    # Performance metrics
    daily_return: Decimal = Field(default=Decimal("0"))
    weekly_return: Decimal = Field(default=Decimal("0"))
    monthly_return: Decimal = Field(default=Decimal("0"))
    total_return: Decimal = Field(default=Decimal("0"))
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bot_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
