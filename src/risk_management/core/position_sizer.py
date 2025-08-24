"""Position sizing with strategy pattern to eliminate duplication."""

from abc import ABC, abstractmethod
from decimal import Decimal

import numpy as np

from src.core.dependency_injection import injectable
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.logging import get_logger
from src.core.types.risk import PositionSizeMethod
from src.core.types.trading import Signal
from src.utils.decimal_utils import safe_divide
from src.utils.decorators import UnifiedDecorator as dec

logger = get_logger(__name__)


class PositionSizingStrategy(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """
        Calculate position size.

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            risk_per_trade: Risk per trade as decimal
            **kwargs: Additional strategy-specific parameters

        Returns:
            Position size in base currency
        """
        pass


class FixedPercentageStrategy(PositionSizingStrategy):
    """Fixed percentage of portfolio sizing."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate fixed percentage position size."""
        # Simple fixed percentage
        position_size = portfolio_value * risk_per_trade

        # Adjust for signal strength if available
        if hasattr(signal, "strength") and signal.strength:
            position_size *= Decimal(str(signal.strength))
        elif hasattr(signal, "confidence") and signal.confidence:
            position_size *= Decimal(str(signal.confidence))

        return position_size


class KellyCriterionStrategy(PositionSizingStrategy):
    """Kelly Criterion optimal sizing."""

    def __init__(self, kelly_fraction: Decimal = Decimal("0.25")):
        """
        Initialize Kelly strategy.

        Args:
            kelly_fraction: Fraction of full Kelly to use (for safety)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate Kelly Criterion position size."""
        # Get win probability and win/loss ratio
        win_prob = Decimal(str(kwargs.get("win_probability", 0.5)))
        win_loss_ratio = Decimal(str(kwargs.get("win_loss_ratio", 1.5)))

        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        loss_prob = Decimal("1") - win_prob

        kelly_percentage = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio

        # Apply Kelly fraction for safety
        kelly_percentage *= self.kelly_fraction

        # Ensure positive and within limits
        if kelly_percentage <= Decimal("0"):
            return Decimal("0")

        # Cap at configured risk per trade
        kelly_percentage = min(kelly_percentage, risk_per_trade)

        return portfolio_value * kelly_percentage


class VolatilityAdjustedStrategy(PositionSizingStrategy):
    """Volatility-adjusted position sizing using ATR."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate volatility-adjusted position size."""
        # Get ATR (Average True Range) for volatility
        atr = Decimal(str(kwargs.get("atr", 0)))
        current_price = Decimal(str(kwargs.get("current_price", 0)))

        if atr <= 0 or current_price <= 0:
            # Fallback to fixed percentage
            return portfolio_value * risk_per_trade

        # Calculate position size based on ATR
        # Risk amount = Portfolio * Risk%
        risk_amount = portfolio_value * risk_per_trade

        # Position size = Risk amount / (ATR * multiplier)
        atr_multiplier = Decimal(str(kwargs.get("atr_multiplier", 2.0)))
        stop_distance = atr * atr_multiplier

        shares = safe_divide(risk_amount, stop_distance)
        position_size = shares * current_price

        # Ensure position size doesn't exceed portfolio value
        position_size = min(position_size, portfolio_value * Decimal("0.95"))

        return position_size


class ConfidenceWeightedStrategy(PositionSizingStrategy):
    """ML confidence-weighted position sizing."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate confidence-weighted position size."""
        # Use strength if available, otherwise confidence for backward compatibility
        confidence_value = None
        if hasattr(signal, "strength") and signal.strength:
            confidence_value = Decimal(str(signal.strength))
        elif hasattr(signal, "confidence") and signal.confidence:
            confidence_value = Decimal(str(signal.confidence))

        if not confidence_value:
            # No confidence/strength, use minimum size
            return portfolio_value * risk_per_trade * Decimal("0.1")

        confidence = confidence_value

        # Map confidence to position size multiplier
        # Low confidence (< 0.6): 10-50% of max size
        # Medium confidence (0.6-0.8): 50-80% of max size
        # High confidence (> 0.8): 80-100% of max size

        if confidence < Decimal("0.6"):
            multiplier = Decimal("0.1") + (confidence * Decimal("0.67"))
        elif confidence < Decimal("0.8"):
            multiplier = Decimal("0.5") + ((confidence - Decimal("0.6")) * Decimal("1.5"))
        else:
            multiplier = Decimal("0.8") + ((confidence - Decimal("0.8")) * Decimal("1.0"))

        base_size = portfolio_value * risk_per_trade
        return base_size * multiplier


class OptimalFStrategy(PositionSizingStrategy):
    """Optimal F position sizing based on historical performance."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate Optimal F position size."""
        # Get historical trade results
        trade_results = kwargs.get("trade_results", [])

        if not trade_results or len(trade_results) < 10:
            # Not enough history, use conservative sizing
            return portfolio_value * risk_per_trade * Decimal("0.5")

        # Calculate Optimal F
        # This is a simplified version - full implementation would use
        # geometric mean maximization

        wins = [r for r in trade_results if r > 0]
        losses = [abs(r) for r in trade_results if r < 0]

        if not wins or not losses:
            return portfolio_value * risk_per_trade * Decimal("0.5")

        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        win_rate = len(wins) / len(trade_results)

        # Simplified Optimal F calculation
        optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        optimal_f = max(0, min(optimal_f, 0.25))  # Cap at 25% for safety

        return portfolio_value * Decimal(str(optimal_f))


@injectable(singleton=True)
class PositionSizer:
    """
    Centralized position sizer using strategy pattern.

    This eliminates duplication of position sizing logic across modules.
    """

    def __init__(self):
        """Initialize position sizer."""
        self._strategies: dict[PositionSizeMethod, PositionSizingStrategy] = {
            PositionSizeMethod.FIXED_PERCENTAGE: FixedPercentageStrategy(),
            PositionSizeMethod.KELLY_CRITERION: KellyCriterionStrategy(),
            PositionSizeMethod.VOLATILITY_ADJUSTED: VolatilityAdjustedStrategy(),
            PositionSizeMethod.CONFIDENCE_WEIGHTED: ConfidenceWeightedStrategy(),
            PositionSizeMethod.OPTIMAL_F: OptimalFStrategy(),
        }
        self._logger = logger

        # Position limits
        self.max_position_size: Decimal | None = None
        self.min_position_size: Decimal | None = None

    @dec.enhance(validate=True, log=True)
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: Decimal,
        method: PositionSizeMethod,
        risk_per_trade: Decimal,
        **kwargs,
    ) -> Decimal:
        """
        Calculate position size using specified method.

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            method: Position sizing method
            risk_per_trade: Risk per trade as decimal
            **kwargs: Additional method-specific parameters

        Returns:
            Position size in base currency

        Raises:
            RiskManagementError: If calculation fails
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if portfolio_value <= 0:
            raise ValidationError("Portfolio value must be positive")

        if risk_per_trade <= 0 or risk_per_trade > Decimal("0.1"):
            raise ValidationError("Risk per trade must be between 0 and 10%")

        # Get strategy
        strategy = self._strategies.get(method)
        if not strategy:
            raise RiskManagementError(f"Unknown position sizing method: {method}")

        try:
            # Calculate size using strategy
            position_size = strategy.calculate_size(
                signal, portfolio_value, risk_per_trade, **kwargs
            )

            # Apply limits
            position_size = self._apply_limits(position_size, portfolio_value)

            self._logger.info(
                "Position size calculated",
                method=method.value,
                size=float(position_size),
                portfolio_value=float(portfolio_value),
            )

            return position_size

        except Exception as e:
            raise RiskManagementError(f"Position size calculation failed: {e}")

    def _apply_limits(self, position_size: Decimal, portfolio_value: Decimal) -> Decimal:
        """Apply position size limits."""
        # Apply maximum position size
        if self.max_position_size:
            max_size = min(self.max_position_size, portfolio_value * Decimal("0.5"))
            position_size = min(position_size, max_size)

        # Apply minimum position size
        if self.min_position_size:
            position_size = max(position_size, self.min_position_size)

        # Ensure position doesn't exceed portfolio value
        position_size = min(position_size, portfolio_value * Decimal("0.95"))

        return position_size

    def set_limits(self, max_size: Decimal | None = None, min_size: Decimal | None = None) -> None:
        """
        Set position size limits.

        Args:
            max_size: Maximum position size
            min_size: Minimum position size
        """
        if max_size and max_size > 0:
            self.max_position_size = max_size

        if min_size and min_size > 0:
            self.min_position_size = min_size

    def add_custom_strategy(
        self, method: PositionSizeMethod, strategy: PositionSizingStrategy
    ) -> None:
        """
        Add a custom position sizing strategy.

        Args:
            method: Method identifier
            strategy: Strategy implementation
        """
        self._strategies[method] = strategy
        self._logger.info(f"Added custom position sizing strategy: {method.value}")
