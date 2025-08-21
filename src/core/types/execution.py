"""Execution-related types for the T-Bot trading system."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    SNIPER = "sniper"
    SMART = "smart"
    SMART_ROUTER = "smart_router"


class ExecutionStatus(Enum):
    """Execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIAL = "partial"


class SlippageType(Enum):
    """Slippage classification."""

    MARKET_IMPACT = "market_impact"
    TIMING = "timing"
    SPREAD = "spread"
    FEES = "fees"
    PRICE_IMPROVEMENT = "price_improvement"


class ExecutionInstruction(BaseModel):
    """Execution instruction for order placement."""

    instruction_id: str
    symbol: str
    side: str  # "buy" or "sell"
    target_quantity: Decimal
    algorithm: ExecutionAlgorithm

    # Algorithm parameters
    urgency: float = Field(ge=0.0, le=1.0, default=0.5)
    start_time: datetime | None = None
    end_time: datetime | None = None

    # TWAP/VWAP specific
    slice_count: int | None = None
    slice_interval: int | None = None  # seconds

    # Iceberg specific
    visible_quantity: Decimal | None = None

    # Limit order specific
    limit_price: Decimal | None = None
    post_only: bool = False

    # Smart routing
    preferred_venues: list[str] = Field(default_factory=list)
    avoid_venues: list[str] = Field(default_factory=list)

    # Risk controls
    max_slippage_pct: float = 0.01
    max_spread_pct: float = 0.002
    cancel_on_disconnect: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of execution algorithm."""

    instruction_id: str
    symbol: str
    status: ExecutionStatus

    # Execution details
    target_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal

    # Pricing
    target_price: Decimal | None = None
    average_price: Decimal
    worst_price: Decimal
    best_price: Decimal

    # Slippage analysis
    expected_cost: Decimal
    actual_cost: Decimal
    slippage_bps: float  # basis points
    slippage_amount: Decimal

    # Execution quality
    fill_rate: float
    execution_time: int  # seconds
    num_fills: int
    num_orders: int

    # Fees
    total_fees: Decimal
    maker_fees: Decimal
    taker_fees: Decimal

    # Timestamps
    started_at: datetime
    completed_at: datetime | None = None

    # Detailed fills
    fills: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.target_quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.target_quantity * 100)

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.FAILED,
        ]


class SlippageMetrics(BaseModel):
    """Slippage analysis metrics."""

    symbol: str
    timeframe: str  # "1h", "1d", "1w", "1m"

    # Aggregate metrics
    total_trades: int
    total_volume: Decimal

    # Slippage by type
    market_impact_bps: float
    timing_cost_bps: float
    spread_cost_bps: float
    total_slippage_bps: float

    # Slippage by size
    small_order_slippage: float  # < 10% ADV
    medium_order_slippage: float  # 10-50% ADV
    large_order_slippage: float  # > 50% ADV

    # Slippage by time
    market_open_slippage: float
    market_close_slippage: float
    intraday_slippage: float

    # Slippage by market condition
    high_volatility_slippage: float
    low_volatility_slippage: float
    trending_slippage: float
    ranging_slippage: float

    # Cost analysis
    total_slippage_cost: Decimal
    avg_slippage_per_trade: Decimal

    # Improvement metrics
    price_improvement_count: int
    price_improvement_amount: Decimal

    period_start: datetime
    period_end: datetime
    updated_at: datetime

    metadata: dict[str, Any] = Field(default_factory=dict)
