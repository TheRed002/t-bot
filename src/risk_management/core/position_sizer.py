"""Position sizing with strategy pattern using centralized utilities.

This module now uses centralized position sizing utilities to eliminate
code duplication across the risk management module.
"""

from decimal import Decimal

from src.core.dependency_injection import injectable
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.logging import get_logger
from src.core.types import PositionSizeMethod, Signal
from src.utils.decimal_utils import format_decimal
from src.utils.decorators import UnifiedDecorator as dec
from src.utils.position_sizing import (
    calculate_position_metrics,
    calculate_position_size,
    update_position_history,
    validate_position_size,
)

logger = get_logger(__name__)


# Strategy classes removed - now using centralized utilities
# All position sizing logic is handled by src.utils.position_sizing


@injectable(singleton=True)
class PositionSizer:
    """
    Centralized position sizer using strategy pattern.

    This eliminates duplication of position sizing logic across modules.
    """

    def __init__(self) -> None:
        """Initialize position sizer."""
        self.logger = logger

        # Position history for volatility calculations
        self.position_history: dict[str, list[Decimal]] = {}

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
        Calculate position size using centralized utility.

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

        try:
            # Add position metrics if available
            if signal.symbol in self.position_history:
                position_metrics = calculate_position_metrics(signal.symbol, self.position_history)
                kwargs.update(position_metrics)

            # Calculate size using centralized utility
            position_size = calculate_position_size(
                method, signal, portfolio_value, risk_per_trade, **kwargs
            )

            # Apply custom limits if set
            if self.max_position_size or self.min_position_size:
                position_size = self._apply_limits(position_size, portfolio_value)

            self.logger.info(
                "Position size calculated",
                method=method.value,
                size=format_decimal(position_size),
                portfolio_value=format_decimal(portfolio_value),
            )

            return position_size

        except Exception as e:
            raise RiskManagementError(f"Position size calculation failed: {e}") from e

    def _apply_limits(self, position_size: Decimal, portfolio_value: Decimal) -> Decimal:
        """Apply custom position size limits if set."""
        original_size = position_size

        # Apply maximum position size
        if self.max_position_size:
            max_size = min(self.max_position_size, portfolio_value * Decimal("0.5"))
            position_size = min(position_size, max_size)

        # Apply minimum position size
        if self.min_position_size:
            position_size = max(position_size, self.min_position_size)

        # Use centralized validation for standard limits
        is_valid, validated_size = validate_position_size(position_size, portfolio_value)
        if not is_valid:
            return Decimal("0")

        if validated_size != original_size:
            self.logger.info(f"Position size adjusted from {original_size} to {validated_size}")

        return validated_size

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

    def update_price_history(self, symbol: str, price: Decimal) -> None:
        """
        Update price history for position sizing calculations.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        update_position_history(symbol, price, self.position_history)

    def get_position_metrics(self, symbol: str) -> dict:
        """
        Get position-specific metrics for sizing.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of position metrics
        """
        return calculate_position_metrics(symbol, self.position_history)
