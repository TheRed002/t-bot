"""
Position Sizing Service Implementation.

This service handles all position sizing logic through dependency injection,
following proper service layer patterns.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.base.service import BaseService
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import PositionSizeMethod, Signal
from src.utils.decimal_utils import ONE, ZERO, format_decimal, safe_divide, to_decimal

if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.state import StateService


class PositionSizingService(BaseService):
    """Service for calculating position sizes using various methods."""

    def __init__(
        self,
        database_service: "DatabaseService",
        state_service: "StateService",
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize position sizing service.

        Args:
            database_service: Database service for data access
            state_service: State service for state management
            config: Application configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="PositionSizingService",
            config=config.__dict__ if config else {},
            correlation_id=correlation_id,
        )

        self.database_service = database_service
        self.state_service = state_service
        self.config = config

    async def calculate_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None = None,
    ) -> Decimal:
        """
        Calculate position size using specified method.

        Args:
            signal: Trading signal
            available_capital: Available capital
            current_price: Current market price
            method: Sizing method

        Returns:
            Calculated position size

        Raises:
            RiskManagementError: If calculation fails
            ValidationError: If inputs are invalid
        """
        try:
            # Validate inputs
            self._validate_inputs(signal, available_capital, current_price)

            # Use default method if not specified
            if method is None:
                method = PositionSizeMethod.FIXED_PERCENTAGE

            # Calculate based on method
            if method == PositionSizeMethod.FIXED_PERCENTAGE:
                position_size = await self._fixed_percentage_sizing(signal, available_capital)
            elif method == PositionSizeMethod.KELLY_CRITERION:
                position_size = await self._kelly_criterion_sizing(signal, available_capital)
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                position_size = await self._volatility_adjusted_sizing(signal, available_capital)
            elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
                position_size = await self._confidence_weighted_sizing(signal, available_capital)
            else:
                raise ValidationError(f"Unsupported position sizing method: {method}")

            # Apply limits
            position_size = self._apply_limits(position_size, available_capital)

            self._logger.info(
                "Position size calculated",
                symbol=signal.symbol,
                method=method.value if method else "default",
                size=format_decimal(position_size),
            )

            return position_size

        except Exception as e:
            self._logger.error(f"Position size calculation failed: {e}")
            raise RiskManagementError(f"Position size calculation failed: {e}") from e

    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool:
        """
        Validate calculated position size.

        Args:
            position_size: Calculated position size
            available_capital: Available capital

        Returns:
            True if size is valid
        """
        try:
            min_size = available_capital * to_decimal("0.01")
            if position_size < min_size:
                return False

            max_size = available_capital * to_decimal("0.25")
            if position_size > max_size:
                return False

            return True

        except Exception as e:
            self._logger.error(f"Position size validation failed: {e}")
            return False

    def _validate_inputs(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> None:
        """Validate calculation inputs."""
        if not signal or not signal.symbol:
            raise ValidationError("Invalid signal")

        if available_capital <= ZERO:
            raise ValidationError("Available capital must be positive")

        if current_price <= ZERO:
            raise ValidationError("Current price must be positive")

        if not hasattr(signal, "strength") or not (0 < signal.strength <= 1):
            raise ValidationError("Signal must have valid strength between 0 and 1")

    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using fixed percentage method."""
        # Default to 5% of capital
        base_percentage = to_decimal("0.05")
        base_size = available_capital * base_percentage

        # Adjust for signal strength
        strength_multiplier = to_decimal(signal.strength)
        return base_size * strength_multiplier

    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using Kelly Criterion."""
        try:
            # Get historical returns from database
            returns = await self._get_historical_returns(signal.symbol)

            if len(returns) < 30:
                self._logger.warning(
                    "Insufficient data for Kelly, using fixed percentage",
                    symbol=signal.symbol,
                    data_points=len(returns),
                )
                return await self._fixed_percentage_sizing(signal, available_capital)

            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(returns)

            if kelly_fraction <= ZERO:
                # Negative edge, use minimum size
                return available_capital * to_decimal("0.01")

            # Apply Half-Kelly for safety
            half_kelly = kelly_fraction * to_decimal("0.5")

            # Adjust for signal strength
            strength_adjusted = half_kelly * to_decimal(signal.strength)

            return available_capital * strength_adjusted

        except Exception as e:
            self._logger.error(f"Kelly calculation failed: {e}")
            return await self._fixed_percentage_sizing(signal, available_capital)

    async def _volatility_adjusted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using volatility adjustment."""
        try:
            # Get price history from database
            prices = await self._get_price_history(signal.symbol)

            if len(prices) < 20:
                return await self._fixed_percentage_sizing(signal, available_capital)

            # Calculate volatility
            volatility = self._calculate_volatility(prices)

            target_vol = to_decimal("0.02")
            vol_adjustment = safe_divide(target_vol, to_decimal(volatility), ONE)

            # Limit adjustment
            vol_adjustment = max(to_decimal("0.1"), min(vol_adjustment, to_decimal("5.0")))

            # Base size adjusted for volatility and strength
            base_size = available_capital * to_decimal("0.05")
            return base_size * vol_adjustment * to_decimal(signal.strength)

        except Exception as e:
            self._logger.error(f"Volatility adjustment failed: {e}")
            return await self._fixed_percentage_sizing(signal, available_capital)

    async def _confidence_weighted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using confidence weighting."""
        base_size = available_capital * to_decimal("0.05")

        # Non-linear confidence scaling
        confidence = to_decimal(signal.strength)
        confidence_weight = confidence**2  # Square for non-linear scaling

        return base_size * confidence_weight

    def _apply_limits(self, position_size: Decimal, available_capital: Decimal) -> Decimal:
        """Apply position size limits."""
        max_size = available_capital * to_decimal("0.25")
        if position_size > max_size:
            position_size = max_size

        min_size = available_capital * to_decimal("0.01")
        if position_size < min_size:
            return ZERO

        return position_size

    def _calculate_kelly_fraction(self, returns: list[Decimal]) -> Decimal:
        """Calculate Kelly fraction from historical returns."""
        import numpy as np

        returns_array = np.array(returns)

        # Separate wins and losses
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]

        if len(wins) == 0 or len(losses) == 0:
            return ZERO

        # Calculate probabilities
        win_prob = to_decimal(len(wins) / len(returns_array))
        loss_prob = ONE - win_prob

        # Calculate average win/loss
        avg_win = to_decimal(np.mean(wins))
        avg_loss = to_decimal(abs(np.mean(losses)))

        if avg_loss <= ZERO:
            return ZERO

        win_loss_ratio = safe_divide(avg_win, avg_loss, ZERO)
        kelly_fraction = safe_divide(win_prob * win_loss_ratio - loss_prob, win_loss_ratio, ZERO)

        return kelly_fraction

    def _calculate_volatility(self, prices: list[Decimal]) -> Decimal:
        """Calculate price volatility using Decimal precision."""
        import numpy as np

        price_array = np.array([float(p) for p in prices])
        returns = np.diff(price_array) / price_array[:-1]
        volatility = np.std(returns)
        return to_decimal(str(volatility))

    async def _get_historical_returns(self, symbol: str) -> list[Decimal]:
        """Get historical returns from database."""
        try:
            # Use database service to get historical data
            # This is a placeholder - actual implementation would depend on database schema
            return []  # Historical data retrieval not implemented
        except Exception as e:
            self._logger.error(f"Failed to get historical returns for {symbol}: {e}")
            return []

    async def _get_price_history(self, symbol: str) -> list[Decimal]:
        """Get price history from database."""
        try:
            # Use database service to get price history
            # This is a placeholder - actual implementation would depend on database schema
            return []  # Price history retrieval not implemented
        except Exception as e:
            self._logger.error(f"Failed to get price history for {symbol}: {e}")
            return []
