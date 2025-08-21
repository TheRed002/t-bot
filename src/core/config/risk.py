"""Risk management configuration for the T-Bot trading system."""

from decimal import Decimal

from pydantic import Field, field_validator

from .base import BaseConfig


class RiskConfig(BaseConfig):
    """Risk management configuration."""

    # Position sizing
    position_sizing_method: str = Field(
        default="fixed", description="Position sizing method", alias="RISK_POSITION_METHOD"
    )
    max_position_size: Decimal = Field(
        default=Decimal("1000"),
        description="Maximum position size in base currency",
        alias="RISK_MAX_POSITION",
    )
    max_positions_per_symbol: int = Field(
        default=1, ge=1, le=10, description="Maximum positions per symbol"
    )
    max_total_positions: int = Field(
        default=10, ge=1, le=100, description="Maximum total positions across all symbols"
    )

    # Risk per trade
    risk_per_trade: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Maximum risk per trade (2% default)",
        alias="RISK_PER_TRADE",
    )
    max_leverage: float = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="Maximum leverage to use",
        alias="RISK_MAX_LEVERAGE",
    )

    # Stop loss and take profit
    default_stop_loss: float = Field(
        default=0.02,
        ge=0.001,
        le=0.5,
        description="Default stop loss percentage",
        alias="RISK_STOP_LOSS",
    )
    default_take_profit: float = Field(
        default=0.04,
        ge=0.001,
        le=1.0,
        description="Default take profit percentage",
        alias="RISK_TAKE_PROFIT",
    )
    trailing_stop: bool = Field(
        default=False, description="Enable trailing stop loss", alias="RISK_TRAILING_STOP"
    )
    trailing_stop_distance: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Trailing stop distance"
    )

    # Drawdown limits
    max_drawdown: float = Field(
        default=0.2,
        ge=0.01,
        le=0.5,
        description="Maximum allowed drawdown (20% default)",
        alias="RISK_MAX_DRAWDOWN",
    )
    max_daily_loss: Decimal = Field(
        default=Decimal("500"),
        description="Maximum daily loss allowed",
        alias="RISK_MAX_DAILY_LOSS",
    )
    max_weekly_loss: Decimal = Field(
        default=Decimal("2000"), description="Maximum weekly loss allowed"
    )
    max_monthly_loss: Decimal = Field(
        default=Decimal("5000"), description="Maximum monthly loss allowed"
    )

    # Circuit breakers
    enable_circuit_breakers: bool = Field(
        default=True, description="Enable circuit breakers", alias="RISK_CIRCUIT_BREAKERS"
    )
    loss_limit_circuit_breaker: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Loss limit to trigger circuit breaker"
    )
    volatility_circuit_breaker: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Volatility threshold for circuit breaker"
    )
    circuit_breaker_cooldown: int = Field(
        default=3600, ge=60, le=86400, description="Circuit breaker cooldown in seconds"
    )

    # Correlation limits
    max_correlation: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Maximum correlation between positions"
    )
    correlation_window: int = Field(
        default=30, ge=10, le=100, description="Correlation calculation window in days"
    )

    # Kelly Criterion parameters (if used)
    kelly_fraction: float = Field(
        default=0.25, ge=0.1, le=1.0, description="Kelly criterion fraction (0.25 = quarter Kelly)"
    )
    kelly_win_probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Estimated win probability for Kelly"
    )
    kelly_win_loss_ratio: float | None = Field(
        default=None, ge=0.1, le=10.0, description="Win/loss ratio for Kelly"
    )

    # Risk parity parameters (if used)
    risk_parity_target_volatility: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Target volatility for risk parity"
    )
    risk_parity_rebalance_threshold: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Rebalance threshold for risk parity"
    )

    # Portfolio limits
    max_portfolio_concentration: float = Field(
        default=0.3, ge=0.1, le=1.0, description="Maximum concentration in single asset"
    )
    min_portfolio_diversification: int = Field(
        default=3, ge=1, le=20, description="Minimum number of assets in portfolio"
    )

    # Capital protection
    capital_preservation_mode: bool = Field(
        default=False, description="Enable capital preservation mode"
    )
    min_capital_threshold: Decimal = Field(
        default=Decimal("5000"), description="Minimum capital threshold"
    )
    emergency_stop_threshold: Decimal = Field(
        default=Decimal("3000"), description="Emergency stop trading threshold"
    )

    @field_validator("position_sizing_method")
    @classmethod
    def validate_sizing_method(cls, v: str) -> str:
        """Validate position sizing method."""
        valid_methods = [
            "fixed",
            "kelly_criterion",
            "risk_parity",
            "volatility_adjusted",
            "equal_weight",
        ]
        if v not in valid_methods:
            raise ValueError(f"Position sizing method must be one of {valid_methods}, got {v}")
        return v

    def get_position_size_params(self) -> dict:
        """Get parameters for position sizing."""
        params = {
            "method": self.position_sizing_method,
            "max_size": self.max_position_size,
            "risk_per_trade": self.risk_per_trade,
        }

        if self.position_sizing_method == "kelly_criterion":
            params.update(
                {
                    "kelly_fraction": self.kelly_fraction,
                    "win_probability": self.kelly_win_probability,
                    "win_loss_ratio": self.kelly_win_loss_ratio,
                }
            )
        elif self.position_sizing_method == "risk_parity":
            params.update(
                {
                    "target_volatility": self.risk_parity_target_volatility,
                    "rebalance_threshold": self.risk_parity_rebalance_threshold,
                }
            )

        return params

    def is_risk_exceeded(self, current_loss: Decimal) -> bool:
        """Check if any risk limit is exceeded."""
        if current_loss >= self.max_daily_loss:
            return True
        if self.capital_preservation_mode and current_loss >= self.min_capital_threshold * Decimal(
            "0.1"
        ):
            return True
        return False
