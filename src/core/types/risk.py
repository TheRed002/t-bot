"""Risk management types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

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
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
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
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    TECHNICAL = "technical"
    MANUAL = "manual"


class AllocationStrategy(Enum):
    """Capital allocation strategy."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM_BASED = "momentum_based"
    VOLATILITY_INVERSE = "volatility_inverse"
    CUSTOM = "custom"


class RiskMetrics(BaseModel):
    """Risk metrics for positions and strategies."""

    position_id: Optional[str] = None
    strategy_id: Optional[str] = None
    symbol: str
    
    # Position metrics
    position_value: Decimal
    position_risk: Decimal
    var_95: Decimal  # Value at Risk at 95% confidence
    var_99: Decimal  # Value at Risk at 99% confidence
    expected_shortfall: Decimal
    
    # Greeks (for options or equivalent risk)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    
    # Portfolio metrics
    beta: float
    correlation: float
    sharpe_ratio: float
    information_ratio: Optional[float] = None
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: Optional[int] = None
    
    # Exposure metrics
    gross_exposure: Decimal
    net_exposure: Decimal
    leverage: float
    
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionLimits(BaseModel):
    """Position size and risk limits."""

    symbol: str
    max_position_size: Decimal
    max_position_value: Decimal
    max_leverage: float = 1.0
    max_loss_per_trade: Decimal
    max_daily_loss: Decimal
    max_drawdown: float
    concentration_limit: float  # Max % of portfolio
    
    # Time-based limits
    max_trades_per_day: Optional[int] = None
    max_trades_per_hour: Optional[int] = None
    min_time_between_trades: Optional[int] = None  # seconds
    
    # Conditional limits
    reduce_size_on_loss: bool = False
    loss_reduction_factor: float = 1.0
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CircuitBreakerEvent(BaseModel):
    """Circuit breaker trigger event."""

    breaker_id: str
    breaker_type: CircuitBreakerType
    status: CircuitBreakerStatus
    triggered_at: datetime
    trigger_value: float
    threshold_value: float
    cooldown_period: int  # seconds
    reset_at: Optional[datetime] = None
    affected_symbols: List[str] = Field(default_factory=list)
    affected_strategies: List[str] = Field(default_factory=list)
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CapitalAllocation(BaseModel):
    """Capital allocation for strategies and positions."""

    allocation_id: str
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    allocated_capital: Decimal
    used_capital: Decimal
    available_capital: Decimal
    allocation_pct: float
    target_allocation_pct: float
    min_allocation: Decimal
    max_allocation: Decimal
    last_rebalance: datetime
    next_rebalance: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FundFlow(BaseModel):
    """Fund flow tracking for deposits and withdrawals."""

    flow_id: str
    flow_type: str  # "deposit" or "withdrawal"
    amount: Decimal
    currency: str
    exchange: str
    status: str
    requested_at: datetime
    completed_at: Optional[datetime] = None
    reference: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CapitalMetrics(BaseModel):
    """Overall capital and portfolio metrics."""

    total_capital: Decimal
    allocated_capital: Decimal
    available_capital: Decimal
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    
    # Return metrics
    daily_return: float
    weekly_return: float
    monthly_return: float
    yearly_return: float
    total_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    current_drawdown: float
    max_drawdown: float
    var_95: Decimal
    expected_shortfall: Decimal
    
    # Allocation metrics
    strategies_active: int
    positions_open: int
    leverage_used: float
    
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CurrencyExposure(BaseModel):
    """Currency exposure tracking."""

    currency: str
    exposure_amount: Decimal
    exposure_pct: float
    hedge_amount: Decimal = Decimal("0")
    net_exposure: Decimal
    exchange_rate: Decimal
    base_currency: str
    updated_at: datetime


class ExchangeAllocation(BaseModel):
    """Capital allocation across exchanges."""

    exchange: str
    allocated_capital: Decimal
    used_capital: Decimal
    available_capital: Decimal
    allocation_pct: float
    num_positions: int
    total_pnl: Decimal
    last_activity: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WithdrawalRule(BaseModel):
    """Automated withdrawal rules."""

    rule_id: str
    name: str
    enabled: bool = True
    trigger_type: str  # "profit_target", "time_based", "drawdown"
    trigger_value: Any
    withdrawal_pct: float
    min_withdrawal: Decimal
    max_withdrawal: Decimal
    destination: str
    last_triggered: Optional[datetime] = None
    next_check: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CapitalProtection(BaseModel):
    """Capital protection settings."""

    protection_id: str
    enabled: bool = True
    min_capital_threshold: Decimal
    stop_trading_threshold: Decimal
    reduce_size_threshold: Decimal
    size_reduction_factor: float
    max_daily_loss: Decimal
    max_weekly_loss: Decimal
    max_monthly_loss: Decimal
    emergency_liquidation: bool = False
    emergency_threshold: Decimal
    notification_settings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)