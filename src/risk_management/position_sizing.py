"""
Position Sizing Module for P-008 Risk Management Framework.

This module implements various position sizing algorithms including:
- Fixed percentage sizing
- Kelly Criterion optimal sizing
- Volatility-adjusted sizing using ATR
- Confidence-weighted sizing for ML strategies

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from decimal import Decimal
from typing import Any

import numpy as np

from src.core.config import Config
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import PositionSizeMethod, Signal

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.utils.decimal_utils import (
    ONE,
    ZERO,
    clamp_decimal,
    format_decimal,
    safe_divide,
    to_decimal,
)

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution

# P-010A: Reverse integration - CapitalAllocator integration
# Note: CapitalAllocator can be imported when needed, avoiding circular imports

logger = get_logger(__name__)


class PositionSizer:
    """
    Position sizing calculator with multiple algorithms.

    This class implements various position sizing methods to optimize
    risk-adjusted returns while respecting portfolio limits.
    """

    def __init__(self, config: Config):
        """
        Initialize position sizer with configuration.

        Args:
            config: Application configuration containing risk settings
        """
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        self.logger = logger.bind(component="position_sizer")

        # Historical data for calculations - use Decimal for precision
        self.price_history: dict[str, list[Decimal]] = {}
        self.return_history: dict[str, list[Decimal]] = {}

        self.logger.info("Position sizer initialized")

    @time_execution
    async def calculate_position_size(
        self, signal: Signal, portfolio_value: Decimal, method: PositionSizeMethod = None
    ) -> Decimal:
        """
        Calculate position size using specified method.

        Args:
            signal: Trading signal with direction and confidence
            portfolio_value: Current total portfolio value
            method: Position sizing method to use (defaults to config setting)

        Returns:
            Decimal: Calculated position size in base currency

        Raises:
            RiskManagementError: If position size calculation fails
            PositionLimitError: If calculated size exceeds limits
        """
        try:
            # Use default method if not specified
            if method is None:
                method = PositionSizeMethod(self.risk_config.default_position_size_method)

            # Validate inputs
            if not signal or not signal.confidence:
                raise ValidationError("Invalid signal for position sizing")

            if portfolio_value <= 0:
                raise ValidationError("Invalid portfolio value for position sizing")

            # Calculate position size based on method
            if method == PositionSizeMethod.FIXED_PCT:
                position_size = await self._fixed_percentage_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.KELLY_CRITERION:
                position_size = await self._kelly_criterion_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                position_size = await self._volatility_adjusted_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
                position_size = await self._confidence_weighted_sizing(signal, portfolio_value)
            else:
                raise ValidationError(f"Unsupported position sizing method: {method}")

            # Apply maximum position size limit (25% absolute maximum)
            max_position_size = min(
                portfolio_value * to_decimal(self.risk_config.max_position_size_pct),
                portfolio_value * to_decimal("0.25"),  # 25% hard limit
            )
            if position_size > max_position_size:
                self.logger.warning(
                    "Position size exceeds maximum limit, capping",
                    calculated_size=format_decimal(position_size),
                    max_size=format_decimal(max_position_size),
                )
                position_size = max_position_size

            # Validate minimum position size (1% for safety)
            min_position_size = portfolio_value * to_decimal("0.01")  # 1% minimum for real trades
            if position_size < min_position_size:
                self.logger.warning(
                    "Position size below minimum, rejecting trade",
                    calculated_size=format_decimal(position_size),
                    min_size=format_decimal(min_position_size),
                )
                return ZERO

            self.logger.info(
                "Position size calculated",
                method=method.value,
                signal_symbol=signal.symbol,
                signal_confidence=signal.confidence,
                portfolio_value=format_decimal(portfolio_value),
                position_size=format_decimal(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error(
                "Position size calculation failed",
                error=str(e),
                signal_symbol=signal.symbol if signal else None,
            )
            raise RiskManagementError(f"Position size calculation failed: {e}")

    @time_execution
    async def _fixed_percentage_sizing(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using fixed percentage method.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Position size as fixed percentage of portfolio
        """
        # Base position size as percentage of portfolio
        base_size = portfolio_value * to_decimal(self.risk_config.default_position_size_pct)

        # Adjust for signal confidence
        confidence_multiplier = to_decimal(signal.confidence)
        position_size = base_size * confidence_multiplier

        self.logger.debug(
            "Fixed percentage sizing",
            base_size=format_decimal(base_size),
            confidence_multiplier=format_decimal(confidence_multiplier),
            final_size=format_decimal(position_size),
        )

        return position_size

    @time_execution
    async def _kelly_criterion_sizing(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using Kelly Criterion with Half-Kelly for safety.

        Implements the proper Kelly formula: f = (p*b - q) / b
        where:
        - f = fraction of capital to wager
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio (average win / average loss)

        Uses Half-Kelly (f * 0.5) for conservative position sizing.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Position size using Kelly Criterion with proper bounds
        """
        try:
            # Get historical returns for the symbol
            symbol = signal.symbol
            returns = self.return_history.get(symbol, [])

            if len(returns) < self.risk_config.kelly_lookback_days:
                self.logger.warning(
                    "Insufficient data for Kelly Criterion, using fixed percentage",
                    symbol=symbol,
                    available_data=len(returns),
                    required_data=self.risk_config.kelly_lookback_days,
                )
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            # All returns are already Decimal, just slice them
            returns_decimal = returns[-self.risk_config.kelly_lookback_days :]

            # Calculate win probability and win/loss ratio
            winning_returns = [r for r in returns_decimal if r > 0]
            losing_returns = [r for r in returns_decimal if r < 0]

            # Edge case: all returns are zero or same sign
            if not winning_returns or not losing_returns:
                self.logger.warning(
                    "Insufficient win/loss data for Kelly Criterion",
                    winning_trades=len(winning_returns),
                    losing_trades=len(losing_returns),
                )
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            # Calculate probabilities using Decimal
            total_trades = to_decimal(len(returns_decimal))
            win_probability = to_decimal(len(winning_returns)) / total_trades
            loss_probability = ONE - win_probability

            # Calculate average win and loss magnitudes
            avg_win = sum(winning_returns) / to_decimal(len(winning_returns))
            avg_loss = abs(sum(losing_returns) / to_decimal(len(losing_returns)))

            # Prevent division by zero
            if avg_loss <= to_decimal("0.0001"):
                self.logger.warning("Average loss too small for Kelly calculation")
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            # Calculate win/loss ratio (b in Kelly formula)
            win_loss_ratio = avg_win / avg_loss

            # Calculate Kelly fraction: f = (p*b - q) / b
            numerator = (win_probability * win_loss_ratio) - loss_probability
            kelly_fraction = numerator / win_loss_ratio

            # Handle negative Kelly (negative edge)
            if kelly_fraction <= ZERO:
                self.logger.warning(
                    "Negative Kelly fraction detected (negative edge)",
                    kelly_fraction=format_decimal(kelly_fraction),
                    win_probability=format_decimal(win_probability),
                    win_loss_ratio=format_decimal(win_loss_ratio),
                )
                # Use minimum position size for negative edge
                return portfolio_value * to_decimal("0.01")  # 1% minimum

            # Apply Half-Kelly for safety (multiply by 0.5)
            half_kelly_fraction = kelly_fraction * to_decimal("0.5")

            # Apply confidence adjustment
            adjusted_fraction = half_kelly_fraction * to_decimal(signal.confidence)

            # Apply bounds: min 1%, max 25% of portfolio
            # Enforce bounds using clamp_decimal
            final_fraction = clamp_decimal(
                adjusted_fraction,
                to_decimal("0.01"),  # 1% minimum
                to_decimal("0.25"),  # 25% maximum
            )

            # Calculate position size
            position_size = portfolio_value * final_fraction

            self.logger.debug(
                "Kelly Criterion sizing (Half-Kelly)",
                win_probability=format_decimal(win_probability),
                loss_probability=format_decimal(loss_probability),
                avg_win=format_decimal(avg_win),
                avg_loss=format_decimal(avg_loss),
                win_loss_ratio=format_decimal(win_loss_ratio),
                full_kelly=format_decimal(kelly_fraction),
                half_kelly=format_decimal(half_kelly_fraction),
                confidence_adjusted=format_decimal(adjusted_fraction),
                final_fraction=format_decimal(final_fraction),
                position_size=format_decimal(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error(
                "Kelly Criterion calculation failed",
                error=str(e),
                symbol=signal.symbol if signal else None,
            )
            # Fallback to fixed percentage
            return await self._fixed_percentage_sizing(signal, portfolio_value)

    @time_execution
    async def _volatility_adjusted_sizing(
        self, signal: Signal, portfolio_value: Decimal
    ) -> Decimal:
        """
        Calculate position size using volatility-adjusted method.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Position size adjusted for volatility
        """
        try:
            symbol = signal.symbol
            prices = self.price_history.get(symbol, [])

            if len(prices) < self.risk_config.volatility_window:
                self.logger.warning(
                    "Insufficient data for volatility adjustment, using fixed percentage",
                    symbol=symbol,
                    available_data=len(prices),
                    required_data=self.risk_config.volatility_window,
                )
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            # Calculate volatility (standard deviation of returns)
            # Convert Decimal prices to float for numpy operations
            prices_array = np.array(
                [float(p) for p in prices[-self.risk_config.volatility_window :]]
            )
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns)

            # Calculate volatility adjustment factor
            target_volatility = self.risk_config.volatility_target
            volatility_adjustment = target_volatility / max(
                volatility, 0.001
            )  # Avoid division by zero

            # Cap volatility adjustment to reasonable bounds
            volatility_adjustment = max(0.1, min(volatility_adjustment, 5.0))

            # Base position size
            base_size = portfolio_value * to_decimal(self.risk_config.default_position_size_pct)

            # Apply volatility adjustment and confidence
            position_size = (
                base_size * to_decimal(volatility_adjustment) * to_decimal(signal.confidence)
            )

            self.logger.debug(
                "Volatility-adjusted sizing",
                volatility=volatility,
                target_volatility=target_volatility,
                adjustment=volatility_adjustment,
                position_size=format_decimal(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error("Volatility adjustment calculation failed", error=str(e))
            # Fallback to fixed percentage
            return await self._fixed_percentage_sizing(signal, portfolio_value)

    @time_execution
    async def _confidence_weighted_sizing(
        self, signal: Signal, portfolio_value: Decimal
    ) -> Decimal:
        """
        Calculate position size using confidence-weighted method for ML strategies.

        Args:
            signal: Trading signal with confidence from ML model
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Position size weighted by ML confidence
        """
        # Base position size
        base_size = portfolio_value * to_decimal(self.risk_config.default_position_size_pct)

        # Apply confidence weighting with non-linear scaling
        # Higher confidence gets proportionally larger position
        confidence = to_decimal(signal.confidence)
        confidence_weight = confidence**2  # Square for non-linear scaling

        position_size = base_size * confidence_weight

        self.logger.debug(
            "Confidence-weighted sizing",
            confidence=format_decimal(confidence),
            confidence_weight=format_decimal(confidence_weight),
            position_size=format_decimal(position_size),
        )

        return position_size

    @time_execution
    async def update_price_history(self, symbol: str, price: float) -> None:
        """
        Update price history for volatility calculations.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        # Convert to Decimal for precision
        decimal_price = to_decimal(price)
        self.price_history[symbol].append(decimal_price)

        # Keep only recent history to manage memory
        max_history = max(self.risk_config.volatility_window * 2, 100)
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]

        # Calculate and store returns
        if len(self.price_history[symbol]) > 1:
            if symbol not in self.return_history:
                self.return_history[symbol] = []

            prev_price = self.price_history[symbol][-2]
            if prev_price > ZERO:
                return_rate = safe_divide(decimal_price - prev_price, prev_price, ZERO)
                self.return_history[symbol].append(return_rate)

                # Keep only recent returns
                if len(self.return_history[symbol]) > max_history:
                    self.return_history[symbol] = self.return_history[symbol][-max_history:]

    @time_execution
    async def get_position_size_summary(
        self, signal: Signal, portfolio_value: Decimal
    ) -> dict[str, Any]:
        """
        Get comprehensive position size summary for all methods.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value

        Returns:
            Dict containing position sizes for all methods
        """
        summary = {}

        for method in PositionSizeMethod:
            try:
                size = await self.calculate_position_size(signal, portfolio_value, method)
                summary[method.value] = {
                    "position_size": format_decimal(size),
                    "portfolio_percentage": format_decimal(
                        safe_divide(size, portfolio_value, ZERO)
                    ),
                }
            except Exception as e:
                summary[method.value] = {
                    "error": str(e),
                    "position_size": 0,
                    "portfolio_percentage": 0,
                }

        return summary

    @time_execution
    async def validate_position_size(
        self, position_size: Decimal, portfolio_value: Decimal
    ) -> bool:
        """
        Validate calculated position size against limits.

        Args:
            position_size: Calculated position size
            portfolio_value: Current portfolio value

        Returns:
            bool: True if position size is valid
        """
        try:
            # Check minimum position size (1% of portfolio for safety)
            min_size = portfolio_value * to_decimal("0.01")  # 1% minimum
            if position_size < min_size:
                self.logger.warning(
                    "Position size below minimum",
                    position_size=format_decimal(position_size),
                    min_size=format_decimal(min_size),
                )
                return False

            # Check maximum position size (25% absolute maximum)
            max_size = min(
                portfolio_value * to_decimal(self.risk_config.max_position_size_pct),
                portfolio_value * to_decimal("0.25"),  # 25% hard limit
            )
            if position_size > max_size:
                self.logger.warning(
                    "Position size exceeds maximum",
                    position_size=format_decimal(position_size),
                    max_size=format_decimal(max_size),
                )
                return False

            return True

        except Exception as e:
            self.logger.error("Position size validation failed", error=str(e))
            return False
