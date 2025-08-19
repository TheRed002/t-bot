"""Strategy configuration for the T-Bot trading system."""

from typing import Dict, Any, Optional
from pydantic import Field, field_validator
from decimal import Decimal
from .base import BaseConfig


class StrategyConfig(BaseConfig):
    """Strategy-specific configuration."""
    
    # Strategy selection
    active_strategies: list[str] = Field(
        default_factory=list,
        description="List of active strategy IDs"
    )
    
    # Common strategy parameters
    default_timeframe: str = Field(
        default="1h",
        description="Default timeframe for strategies",
        alias="STRATEGY_TIMEFRAME"
    )
    default_lookback_period: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Default lookback period for indicators"
    )
    
    # Mean Reversion parameters
    mean_reversion_window: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Window size for mean calculation"
    )
    mean_reversion_std_dev: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Number of standard deviations for entry/exit"
    )
    mean_reversion_entry_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Entry threshold for mean reversion"
    )
    mean_reversion_exit_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Exit threshold for mean reversion"
    )
    
    # Momentum parameters
    momentum_period: int = Field(
        default=14,
        ge=5,
        le=100,
        description="Period for momentum calculation"
    )
    momentum_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Momentum threshold for signals"
    )
    momentum_acceleration_factor: float = Field(
        default=0.02,
        ge=0.01,
        le=0.2,
        description="Acceleration factor for momentum"
    )
    
    # Arbitrage parameters
    arbitrage_min_spread: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Minimum spread for arbitrage opportunities"
    )
    arbitrage_execution_time: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Max execution time in milliseconds"
    )
    
    # Market Making parameters
    market_making_spread: float = Field(
        default=0.002,
        ge=0.0001,
        le=0.1,
        description="Spread for market making orders"
    )
    market_making_inventory_target: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Target inventory level (0.5 = balanced)"
    )
    market_making_order_levels: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of order levels on each side"
    )
    
    # Trend Following parameters
    trend_ema_fast: int = Field(
        default=12,
        ge=5,
        le=50,
        description="Fast EMA period for trend following"
    )
    trend_ema_slow: int = Field(
        default=26,
        ge=20,
        le=200,
        description="Slow EMA period for trend following"
    )
    trend_macd_signal: int = Field(
        default=9,
        ge=5,
        le=20,
        description="MACD signal line period"
    )
    
    # Signal generation
    signal_min_strength: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum signal strength to act on"
    )
    signal_combination_method: str = Field(
        default="weighted_average",
        description="Method for combining multiple signals"
    )
    
    # Backtesting
    backtest_start_date: Optional[str] = Field(
        default=None,
        description="Start date for backtesting (YYYY-MM-DD)"
    )
    backtest_end_date: Optional[str] = Field(
        default=None,
        description="End date for backtesting (YYYY-MM-DD)"
    )
    backtest_initial_capital: Decimal = Field(
        default=Decimal("10000"),
        description="Initial capital for backtesting"
    )
    backtest_commission: float = Field(
        default=0.001,
        ge=0.0,
        le=0.01,
        description="Commission rate for backtesting"
    )
    backtest_slippage: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.01,
        description="Slippage rate for backtesting"
    )
    
    # Performance tracking
    track_metrics: bool = Field(
        default=True,
        description="Enable performance metrics tracking"
    )
    metrics_update_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Metrics update interval in seconds"
    )
    
    @field_validator("default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of {valid_timeframes}, got {v}")
        return v
    
    @field_validator("signal_combination_method")
    @classmethod
    def validate_combination_method(cls, v: str) -> str:
        """Validate signal combination method."""
        valid_methods = ['weighted_average', 'majority_vote', 'unanimous', 'any']
        if v not in valid_methods:
            raise ValueError(f"Combination method must be one of {valid_methods}, got {v}")
        return v
    
    def get_strategy_params(self, strategy_type: str) -> Dict[str, Any]:
        """Get parameters for a specific strategy type."""
        params = {
            "timeframe": self.default_timeframe,
            "lookback_period": self.default_lookback_period,
            "signal_min_strength": self.signal_min_strength
        }
        
        if strategy_type == "mean_reversion":
            params.update({
                "window_size": self.mean_reversion_window,
                "num_std": self.mean_reversion_std_dev,
                "entry_threshold": self.mean_reversion_entry_threshold,
                "exit_threshold": self.mean_reversion_exit_threshold
            })
        elif strategy_type == "momentum":
            params.update({
                "period": self.momentum_period,
                "threshold": self.momentum_threshold,
                "acceleration_factor": self.momentum_acceleration_factor
            })
        elif strategy_type == "arbitrage":
            params.update({
                "min_spread": self.arbitrage_min_spread,
                "execution_time": self.arbitrage_execution_time
            })
        elif strategy_type == "market_making":
            params.update({
                "spread": self.market_making_spread,
                "inventory_target": self.market_making_inventory_target,
                "order_levels": self.market_making_order_levels
            })
        elif strategy_type == "trend_following":
            params.update({
                "ema_fast": self.trend_ema_fast,
                "ema_slow": self.trend_ema_slow,
                "macd_signal": self.trend_macd_signal
            })
        
        return params