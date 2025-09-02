"""
Position Sizing Module - DEPRECATED.

DEPRECATED: This module is deprecated in favor of RiskService.
The new RiskService (src/risk_management/service.py) provides all position sizing
functionality with:
- DatabaseService integration (no direct DB access)
- StateService integration for state management
- Comprehensive caching layer
- Enhanced error handling with circuit breakers
- Multiple position sizing algorithms including Kelly Criterion

This module is maintained for backward compatibility only.
New implementations should use RiskService.calculate_position_size() directly.

Legacy algorithms (now in RiskService):
- Fixed percentage sizing
- Kelly Criterion optimal sizing
- Volatility-adjusted sizing using ATR
- Confidence-weighted sizing for ML strategies

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import RiskManagementError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import PositionSizeMethod, Signal

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService

# MANDATORY: Import from P-002A
from src.error_handling import ErrorHandler

# MANDATORY: Import from P-007A
from src.monitoring.financial_precision import safe_decimal_to_float
from src.utils.constants import POSITION_SIZING_LIMITS
from src.utils.decimal_utils import (
    ONE,
    ZERO,
    clamp_decimal,
    format_decimal,
    safe_divide,
    to_decimal,
)
from src.utils.decorators import time_execution

# P-010A: Reverse integration - CapitalAllocator integration


class PositionSizer(BaseComponent):
    """
    DEPRECATED: Position sizing calculator with multiple algorithms.

    This class is deprecated in favor of RiskService which provides:
    - Enterprise-grade service architecture
    - DatabaseService integration (no direct DB access)
    - Enhanced caching and monitoring
    - Real-time risk management

    This class implements various position sizing methods to optimize
    risk-adjusted returns while respecting portfolio limits.

    DEPRECATED METHODS -> USE RiskService INSTEAD:
    - calculate_position_size() -> RiskService.calculate_position_size()
    - _fixed_percentage_sizing() -> Built into RiskService
    - _kelly_criterion_sizing() -> Built into RiskService with caching
    - _volatility_adjusted_sizing() -> Built into RiskService
    - _confidence_weighted_sizing() -> Built into RiskService
    """

    def __init__(self, config: Config, database_service: "DatabaseService | None" = None):
        """
        Initialize DEPRECATED position sizer with configuration.

        DEPRECATED: Use RiskService.calculate_position_size() directly.

        Args:
            config: Application configuration containing risk settings
            database_service: Database service for data access (not used in legacy mode)
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        self.database_service = database_service

        # DEPRECATED: Historical data for calculations - use Decimal for precision
        # RiskService handles this with proper caching and state management
        self.price_history: dict[str, list[Decimal]] = {}
        self.return_history: dict[str, list[float]] = {}

        if database_service:
            self.logger.warning(
                "PositionSizer initialized with DatabaseService - "
                "consider migrating to RiskService for full integration"
            )
        else:
            self.logger.warning(
                "DEPRECATED PositionSizer initialized in legacy mode - "
                "migrate to RiskService for enterprise features"
            )

    @time_execution
    async def calculate_position_size(
        self, signal: Signal, portfolio_value: Decimal, method: PositionSizeMethod | None = None
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
                try:
                    method = PositionSizeMethod(self.risk_config.default_position_size_method)
                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        "Invalid default position sizing method in config, using FIXED_PERCENTAGE",
                        config_method=getattr(self.risk_config, "default_position_size_method", None),
                        error=str(e),
                    )
                    method = PositionSizeMethod.FIXED_PERCENTAGE

            # Enhanced input validation
            if not signal:
                raise ValidationError("Signal cannot be None")

            if not hasattr(signal, "symbol") or not signal.symbol:
                raise ValidationError("Signal must have a valid symbol")

            confidence = getattr(signal, "confidence", None) or getattr(signal, "strength", None)
            if confidence is None:
                raise ValidationError("Signal must have a confidence or strength value")

            if not isinstance(confidence, int | float | Decimal) or not (0 < confidence <= 1):
                raise ValidationError(f"Signal confidence/strength must be between 0 and 1, got: {confidence}")

            if not isinstance(portfolio_value, Decimal | int | float):
                raise ValidationError(f"Portfolio value must be numeric, got {type(portfolio_value).__name__}")

            if portfolio_value <= 0:
                raise ValidationError(f"Portfolio value must be positive, got: {portfolio_value}")

            # Calculate position size based on method
            if method == PositionSizeMethod.FIXED_PERCENTAGE:
                position_size = await self._fixed_percentage_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.KELLY_CRITERION:
                position_size = await self._kelly_criterion_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                position_size = await self._volatility_adjusted_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
                position_size = await self._confidence_weighted_sizing(signal, portfolio_value)
            else:
                raise ValidationError(f"Unsupported position sizing method: {method}")

            max_position_size = min(
                portfolio_value * to_decimal(self.risk_config.risk_per_trade),
                portfolio_value * to_decimal(str(POSITION_SIZING_LIMITS["max_position_size_pct"])),
            )
            if position_size > max_position_size:
                self.logger.warning(
                    "Position size exceeds maximum limit, capping",
                    calculated_size=format_decimal(position_size),
                    max_size=format_decimal(max_position_size),
                )
                position_size = max_position_size

            min_position_size = portfolio_value * to_decimal(str(POSITION_SIZING_LIMITS["min_position_size_pct"]))
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
                signal_confidence=confidence,
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
            raise RiskManagementError(f"Position size calculation failed: {e}") from e

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
        base_size = portfolio_value * to_decimal(self.risk_config.risk_per_trade)

        # Adjust for signal confidence
        # Get confidence from either confidence or strength field
        confidence = getattr(signal, "confidence", None) or getattr(signal, "strength", None)
        confidence_multiplier = to_decimal(confidence if confidence is not None else 0)
        position_size = base_size * confidence_multiplier

        self.logger.info(
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

            # Convert returns to Decimal and slice them
            returns_decimal = [to_decimal(r) for r in returns[-self.risk_config.kelly_lookback_days :]]

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
            avg_win = sum(winning_returns, ZERO) / to_decimal(len(winning_returns))
            avg_loss = abs(sum(losing_returns, ZERO) / to_decimal(len(losing_returns)))

            # Prevent division by zero
            if avg_loss <= to_decimal(POSITION_SIZING_LIMITS["min_avg_loss_threshold"]):
                self.logger.warning("Average loss too small for Kelly calculation")
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            win_loss_ratio = safe_divide(avg_win, avg_loss, ZERO)
            if win_loss_ratio <= ZERO:
                self.logger.warning("Invalid win/loss ratio calculated")
                return await self._fixed_percentage_sizing(signal, portfolio_value)

            numerator = (win_probability * win_loss_ratio) - loss_probability
            kelly_fraction = safe_divide(numerator, win_loss_ratio, ZERO)

            if kelly_fraction <= ZERO:
                self.logger.warning(
                    "Negative Kelly fraction detected (negative edge)",
                    kelly_fraction=format_decimal(kelly_fraction),
                    win_probability=format_decimal(win_probability),
                    win_loss_ratio=format_decimal(win_loss_ratio),
                )
                return portfolio_value * to_decimal(POSITION_SIZING_LIMITS["min_position_size_pct"])

            half_kelly_fraction = kelly_fraction * to_decimal(POSITION_SIZING_LIMITS["kelly_safety_factor"])

            # Apply confidence adjustment
            # Get confidence from either confidence or strength field
            confidence = getattr(signal, "confidence", None) or getattr(signal, "strength", None)
            adjusted_fraction = half_kelly_fraction * to_decimal(confidence if confidence is not None else 0)

            # Apply bounds: min 1%, max 25% of portfolio
            # Enforce bounds using clamp_decimal
            final_fraction = clamp_decimal(
                adjusted_fraction,
                to_decimal(POSITION_SIZING_LIMITS["min_position_size_pct"]),
                to_decimal(POSITION_SIZING_LIMITS["max_position_size_pct"]),
            )

            # Calculate position size
            position_size = portfolio_value * final_fraction

            self.logger.info(
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
    async def _volatility_adjusted_sizing(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
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

            # Convert Decimal prices to float for numpy operations
            prices_array = np.array([safe_decimal_to_float(p, f"price_{symbol}_{i}") for i, p in enumerate(prices[-self.risk_config.volatility_window :])])
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = to_decimal(str(np.std(returns)))

            # Calculate volatility adjustment factor
            target_volatility = self.risk_config.volatility_target
            min_volatility = Decimal(str(self.risk_config.min_volatility or "0.001"))  # Minimum volatility to avoid division by zero
            volatility_adjustment = safe_divide(target_volatility, max(volatility, min_volatility), ONE)

            # Cap volatility adjustment to reasonable bounds
            min_adjustment = Decimal(str(self.risk_config.min_volatility_adjustment or "0.1"))
            max_adjustment = Decimal(str(self.risk_config.max_volatility_adjustment or "5.0"))
            volatility_adjustment = clamp_decimal(volatility_adjustment, min_adjustment, max_adjustment)

            # Base position size
            base_size = portfolio_value * to_decimal(self.risk_config.risk_per_trade)

            # Apply volatility adjustment and confidence
            # Get confidence from either confidence or strength field
            confidence = getattr(signal, "confidence", None) or getattr(signal, "strength", None)
            position_size = (
                base_size * to_decimal(volatility_adjustment) * to_decimal(confidence if confidence is not None else 0)
            )

            self.logger.info(
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
    async def _confidence_weighted_sizing(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using confidence-weighted method for ML strategies.

        Args:
            signal: Trading signal with confidence from ML model
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Position size weighted by ML confidence
        """
        # Base position size
        base_size = portfolio_value * to_decimal(self.risk_config.risk_per_trade)

        # Apply confidence weighting with non-linear scaling
        # Higher confidence gets proportionally larger position
        # Get confidence from either confidence or strength field
        confidence_value = getattr(signal, "confidence", None) or getattr(signal, "strength", None)
        confidence = to_decimal(confidence_value if confidence_value is not None else 0)
        confidence_weight = confidence**2  # Square for non-linear scaling

        position_size = base_size * confidence_weight

        self.logger.info(
            "Confidence-weighted sizing",
            confidence=format_decimal(confidence),
            confidence_weight=format_decimal(confidence_weight),
            position_size=format_decimal(position_size),
        )

        return position_size

    @time_execution
    async def update_price_history(self, symbol: str, price: Decimal) -> None:
        """
        Update price history for volatility calculations.

        Args:
            symbol: Trading symbol
            price: Current price as Decimal for financial precision

        Raises:
            ValidationError: If input parameters are invalid
        """
        try:
            # Enhanced input validation
            if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
                raise ValidationError(f"Invalid symbol: {symbol} (must be non-empty string)")

            symbol = symbol.strip().upper()  # Normalize symbol

            if not isinstance(price, Decimal | int | float):
                raise ValidationError(f"Price must be Decimal, int, or float, got {type(price).__name__}")

            # Convert to Decimal if needed
            if not isinstance(price, Decimal):
                try:
                    price = to_decimal(price)
                except (ValueError, TypeError) as e:
                    raise ValidationError(f"Cannot convert price to Decimal: {e}") from e

            # Validate price is positive with bounds checking
            if price <= ZERO:
                raise ValidationError(f"Price must be positive: {price}")

            # Apply reasonable price bounds for security
            min_reasonable_price = to_decimal(str(self.risk_config.min_reasonable_price or "0.000001"))
            max_reasonable_price = to_decimal(str(self.risk_config.max_reasonable_price or "1000000"))
            if not (min_reasonable_price <= price <= max_reasonable_price):
                self.logger.warning(
                    f"Price for {symbol} outside reasonable bounds: {price} "
                    f"(bounds: {min_reasonable_price} - {max_reasonable_price})"
                )
                return

            # Initialize symbol data if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            # Use Decimal directly for financial precision
            self.price_history[symbol].append(price)

            # Keep only recent history to manage memory with error handling
            try:
                max_history = max(getattr(self.risk_config, "volatility_window", 20) * 2, 100)
                if len(self.price_history[symbol]) > max_history:
                    self.price_history[symbol] = self.price_history[symbol][-max_history:]
            except (AttributeError, TypeError) as e:
                self.logger.warning(f"Error accessing volatility_window config, using default: {e}")
                max_history = 100
                if len(self.price_history[symbol]) > max_history:
                    self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Calculate and store returns with enhanced error handling
            if len(self.price_history[symbol]) > 1:
                try:
                    if symbol not in self.return_history:
                        self.return_history[symbol] = []

                    prev_price = self.price_history[symbol][-2]
                    if prev_price > ZERO:
                        return_rate = safe_divide(price - prev_price, prev_price, ZERO)

                        # Validate return rate
                        return_float = safe_decimal_to_float(return_rate, f"return_rate_{symbol}")
                        if not (np.isnan(return_float) or np.isinf(return_float)):
                            # Apply reasonable bounds for returns to prevent manipulation
                            min_reasonable_return = Decimal("-1.0")  # -100% maximum loss
                            max_reasonable_return = Decimal("10.0")  # +1000% maximum gain
                            if min_reasonable_return <= return_rate <= max_reasonable_return:
                                self.return_history[symbol].append(return_float)
                            else:
                                self.logger.warning(
                                    f"Extreme return rate calculated for {symbol}: {return_rate}, "
                                    "excluding from history"
                                )
                        else:
                            self.logger.warning(
                                f"Invalid return rate calculated for {symbol}: {return_rate} " "(NaN or Inf)"
                            )

                        # Keep only recent returns
                        if len(self.return_history[symbol]) > max_history:
                            self.return_history[symbol] = self.return_history[symbol][-max_history:]
                    else:
                        self.logger.warning(
                            f"Previous price is zero or negative for {symbol}: {prev_price}, " "cannot calculate return"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error calculating returns for {symbol}: {e}",
                        extra={"current_price": str(price), "error_type": type(e).__name__},
                    )

        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error updating price history for {symbol}: {e}",
                extra={
                    "symbol": symbol,
                    "price": str(price) if price else None,
                    "error_type": type(e).__name__,
                },
            )
            raise ValidationError(f"Failed to update price history for {symbol}: {e}") from e

    @time_execution
    async def get_position_size_summary(self, signal: Signal, portfolio_value: Decimal) -> dict[str, Any]:
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
                    "portfolio_percentage": format_decimal(safe_divide(size, portfolio_value, ZERO)),
                }
            except Exception as e:
                summary[method.value] = {
                    "error": str(e),
                    "position_size": "0",
                    "portfolio_percentage": "0",
                }

        return summary

    @time_execution
    async def validate_position_size(self, position_size: Decimal, portfolio_value: Decimal) -> bool:
        """
        Validate calculated position size against limits.

        Args:
            position_size: Calculated position size
            portfolio_value: Current portfolio value

        Returns:
            bool: True if position size is valid
        """
        try:
            min_size = portfolio_value * to_decimal(POSITION_SIZING_LIMITS["min_position_size_pct"])
            if position_size < min_size:
                self.logger.warning(
                    "Position size below minimum",
                    position_size=format_decimal(position_size),
                    min_size=format_decimal(min_size),
                )
                return False

            max_size = min(
                portfolio_value * to_decimal(self.risk_config.risk_per_trade),
                portfolio_value * to_decimal(str(POSITION_SIZING_LIMITS["max_position_size_pct"])),
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
