"""Strategy-related types for the T-Bot trading system."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrategyType(Enum):
    """Strategy type enumeration."""

    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    BREAKOUT = "breakout"
    CUSTOM = "custom"


class StrategyStatus(Enum):
    """Strategy operational status."""

    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MarketRegime(Enum):
    """Market regime classification."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"

    HIGH_CORRELATION = "high_correlation"
    MEDIUM_CORRELATION = "medium_correlation"
    LOW_CORRELATION = "low_correlation"


class NewsSentiment(Enum):
    """News sentiment classification."""

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class SocialSentiment(Enum):
    """Social media sentiment classification."""

    EXTREMELY_NEGATIVE = "extremely_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    EXTREMELY_POSITIVE = "extremely_positive"


class StrategyConfig(BaseModel):
    """Strategy configuration parameters."""

    strategy_id: str
    strategy_type: StrategyType
    name: str
    symbol: str
    timeframe: str
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Common strategy parameters
    lookback_period: int | None = None
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_position_size: Decimal | None = None
    min_confidence: float | None = None
    max_positions: int | None = None
    position_size_pct: float | None = None
    exchange_type: str | None = None
    requires_risk_manager: bool = False
    requires_exchange: bool = False

    # Mean reversion specific
    window_size: int | None = None
    num_std: float | None = None

    # Momentum specific
    momentum_threshold: float | None = None
    acceleration_factor: float | None = None

    # Market making specific
    spread_pct: float | None = None
    inventory_target: Decimal | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


class StrategyMetrics(BaseModel):
    """Strategy performance metrics."""

    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int | None = None
    calmar_ratio: float = 0.0
    current_position: Decimal | None = None
    last_signal: str | None = None
    last_trade_at: datetime | None = None
    signals_generated: int = 0
    last_signal_time: datetime | None = None
    last_updated: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_win_rate(self) -> None:
        """Update win rate based on trades."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

    def calculate_profit_factor(self) -> Decimal:
        """Calculate profit factor (gross profits / gross losses)."""
        if self.avg_loss == 0 or self.losing_trades == 0:
            return Decimal("inf") if self.winning_trades > 0 else Decimal("0")

        gross_profits = self.avg_win * Decimal(str(self.winning_trades))
        gross_losses = abs(self.avg_loss) * Decimal(str(self.losing_trades))

        return (
            (gross_profits / gross_losses).quantize(Decimal("0.0001"))
            if gross_losses != 0
            else Decimal("inf")
        )


class RegimeChangeEvent(BaseModel):
    """Market regime change event."""

    timestamp: datetime
    symbol: str
    previous_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
