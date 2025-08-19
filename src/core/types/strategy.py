"""Strategy-related types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

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
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


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
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Common strategy parameters
    lookback_period: Optional[int] = None
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_position_size: Optional[Decimal] = None
    
    # Mean reversion specific
    window_size: Optional[int] = None
    num_std: Optional[float] = None
    
    # Momentum specific
    momentum_threshold: Optional[float] = None
    acceleration_factor: Optional[float] = None
    
    # Market making specific
    spread_pct: Optional[float] = None
    inventory_target: Optional[Decimal] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


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
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: Optional[int] = None
    calmar_ratio: float = 0.0
    current_position: Optional[Decimal] = None
    last_signal: Optional[str] = None
    last_trade_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update_win_rate(self) -> None:
        """Update win rate based on trades."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if self.avg_loss == 0 or self.losing_trades == 0:
            return float('inf') if self.winning_trades > 0 else 0.0
        
        gross_profits = self.avg_win * self.winning_trades
        gross_losses = abs(self.avg_loss) * self.losing_trades
        
        return float(gross_profits / gross_losses) if gross_losses != 0 else float('inf')


class RegimeChangeEvent(BaseModel):
    """Market regime change event."""

    timestamp: datetime
    symbol: str
    previous_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)