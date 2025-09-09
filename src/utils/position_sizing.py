"""
Centralized position sizing utilities to eliminate code duplication.

This module provides unified position sizing algorithms, eliminating
duplication across position_sizing.py, core/position_sizer.py, and services.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

import numpy as np

from src.core.logging import get_logger
from src.core.types.risk import PositionSizeMethod
from src.core.types.trading import Signal
from src.utils.decimal_utils import ONE, ZERO, clamp_decimal, safe_divide, to_decimal

logger = get_logger(__name__)


class PositionSizingAlgorithm(ABC):
    """Abstract base class for position sizing algorithms."""

    @abstractmethod
    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size based on algorithm-specific logic."""
        pass

    def validate_inputs(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal
    ) -> bool:
        """Validate common inputs for position sizing."""
        if not signal or not hasattr(signal, "symbol") or not signal.symbol:
            logger.error("Invalid signal: missing symbol")
            return False

        if portfolio_value <= ZERO:
            logger.error(f"Invalid portfolio value: {portfolio_value}")
            return False

        if risk_per_trade <= ZERO or risk_per_trade > to_decimal("0.25"):
            logger.error(f"Invalid risk per trade: {risk_per_trade}")
            return False

        return True


class FixedPercentageAlgorithm(PositionSizingAlgorithm):
    """Fixed percentage position sizing algorithm."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size as fixed percentage of portfolio."""
        if not self.validate_inputs(signal, portfolio_value, risk_per_trade):
            return ZERO

        # Base size as percentage of portfolio
        base_size = portfolio_value * risk_per_trade

        # Apply signal confidence/strength
        confidence = get_signal_confidence(signal)
        position_size = base_size * confidence

        return position_size


class KellyCriterionAlgorithm(PositionSizingAlgorithm):
    """Kelly Criterion position sizing algorithm."""

    def __init__(self, kelly_fraction: Decimal = to_decimal("0.5")):
        """Initialize with safety fraction (half-Kelly by default)."""
        self.kelly_fraction = kelly_fraction

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size using Kelly Criterion."""
        if not self.validate_inputs(signal, portfolio_value, risk_per_trade):
            return ZERO

        # Get historical performance metrics
        win_probability = to_decimal(str(kwargs.get("win_probability", 0.55)))
        win_loss_ratio = to_decimal(str(kwargs.get("win_loss_ratio", 1.5)))

        # Validate Kelly inputs
        if win_probability <= ZERO or win_probability >= ONE:
            logger.warning("Invalid win probability for Kelly, using fixed sizing")
            return self._fallback_to_fixed(signal, portfolio_value, risk_per_trade)

        if win_loss_ratio <= ZERO:
            logger.warning("Invalid win/loss ratio for Kelly, using fixed sizing")
            return self._fallback_to_fixed(signal, portfolio_value, risk_per_trade)

        # Kelly formula: f = (p*b - q) / b
        loss_probability = ONE - win_probability
        kelly_fraction = safe_divide(
            (win_probability * win_loss_ratio) - loss_probability, win_loss_ratio, ZERO
        )

        # Apply safety factor (half-Kelly or configured fraction)
        safe_kelly_fraction = kelly_fraction * self.kelly_fraction

        # Ensure positive and within reasonable bounds
        bounded_fraction = clamp_decimal(
            safe_kelly_fraction,
            to_decimal("0.01"),  # Min 1%
            risk_per_trade,  # Max configured risk per trade
        )

        # Apply signal confidence
        confidence = get_signal_confidence(signal)
        final_fraction = bounded_fraction * confidence

        return portfolio_value * final_fraction

    def _fallback_to_fixed(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal
    ) -> Decimal:
        """Fallback to fixed percentage sizing."""
        fixed_algo = FixedPercentageAlgorithm()
        return fixed_algo.calculate_size(signal, portfolio_value, risk_per_trade)


class VolatilityAdjustedAlgorithm(PositionSizingAlgorithm):
    """Volatility-adjusted position sizing algorithm."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size adjusted for volatility."""
        if not self.validate_inputs(signal, portfolio_value, risk_per_trade):
            return ZERO

        # Get volatility metrics with Decimal conversion
        current_volatility = to_decimal(
            str(kwargs.get("current_volatility", 0.02))
        )  # Default 2% daily vol
        target_volatility = to_decimal(
            str(kwargs.get("target_volatility", 0.015))
        )  # Target 1.5% daily vol

        if current_volatility <= ZERO:
            logger.warning("Invalid volatility data, using fixed sizing")
            return self._fallback_to_fixed(signal, portfolio_value, risk_per_trade)

        # Calculate volatility adjustment with Decimal precision
        volatility_adjustment = min(
            safe_divide(target_volatility, current_volatility, to_decimal("1.0")),
            to_decimal("2.0"),  # Max 2x adjustment
        )
        volatility_adjustment = max(
            volatility_adjustment,
            to_decimal("0.5"),  # Min 0.5x adjustment
        )

        # Base position size
        base_size = portfolio_value * risk_per_trade

        # Apply volatility and confidence adjustments with Decimal precision
        confidence = get_signal_confidence(signal)
        adjusted_size = base_size * volatility_adjustment * confidence

        return adjusted_size

    def _fallback_to_fixed(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal
    ) -> Decimal:
        """Fallback to fixed percentage sizing."""
        fixed_algo = FixedPercentageAlgorithm()
        return fixed_algo.calculate_size(signal, portfolio_value, risk_per_trade)


class ConfidenceWeightedAlgorithm(PositionSizingAlgorithm):
    """Confidence-weighted position sizing for ML signals."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size weighted by signal confidence."""
        if not self.validate_inputs(signal, portfolio_value, risk_per_trade):
            return ZERO

        confidence = get_signal_confidence(signal)

        # Non-linear confidence scaling
        # Low confidence (< 0.6): Reduced scaling
        # High confidence (> 0.8): Enhanced scaling
        if confidence < to_decimal("0.6"):
            confidence_multiplier = confidence * to_decimal("0.8")  # Reduced impact
        elif confidence > to_decimal("0.8"):
            confidence_multiplier = to_decimal("0.8") + (
                confidence - to_decimal("0.8")
            ) * to_decimal("1.5")
        else:
            confidence_multiplier = confidence

        # Ensure multiplier stays within bounds
        confidence_multiplier = clamp_decimal(
            confidence_multiplier, to_decimal("0.1"), to_decimal("1.0")
        )

        base_size = portfolio_value * risk_per_trade
        return base_size * confidence_multiplier


class ATRBasedAlgorithm(PositionSizingAlgorithm):
    """Average True Range based position sizing."""

    def calculate_size(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal, **kwargs
    ) -> Decimal:
        """Calculate position size based on ATR."""
        if not self.validate_inputs(signal, portfolio_value, risk_per_trade):
            return ZERO

        # Get ATR data
        atr = to_decimal(str(kwargs.get("atr", 0)))
        current_price = to_decimal(str(kwargs.get("current_price", 0)))
        atr_multiplier = to_decimal(str(kwargs.get("atr_multiplier", 2.0)))

        if atr <= ZERO or current_price <= ZERO:
            logger.warning("Invalid ATR or price data, using fixed sizing")
            return self._fallback_to_fixed(signal, portfolio_value, risk_per_trade)

        # Calculate risk amount
        risk_amount = portfolio_value * risk_per_trade

        # Calculate stop distance based on ATR
        stop_distance = atr * atr_multiplier

        # Calculate shares based on risk amount and stop distance
        shares = safe_divide(risk_amount, stop_distance, ZERO)
        if shares <= ZERO:
            return ZERO

        # Calculate position size
        position_size = shares * current_price

        # Apply confidence scaling
        confidence = get_signal_confidence(signal)
        final_size = position_size * confidence

        # Ensure position doesn't exceed reasonable portfolio percentage
        max_position = portfolio_value * to_decimal("0.5")  # Max 50% of portfolio
        return min(final_size, max_position)

    def _fallback_to_fixed(
        self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal
    ) -> Decimal:
        """Fallback to fixed percentage sizing."""
        fixed_algo = FixedPercentageAlgorithm()
        return fixed_algo.calculate_size(signal, portfolio_value, risk_per_trade)


# Utility functions


def get_signal_confidence(signal: Signal) -> Decimal:
    """
    Extract confidence/strength from signal with backward compatibility.

    Args:
        signal: Trading signal

    Returns:
        Signal confidence as Decimal (0-1)
    """
    # Try different attribute names for backward compatibility
    for attr_name in ["confidence", "strength", "score"]:
        if hasattr(signal, attr_name):
            confidence_value = getattr(signal, attr_name)
            if confidence_value is not None:
                confidence = to_decimal(str(confidence_value))
                # Ensure within valid range
                return clamp_decimal(confidence, ZERO, ONE)

    # Default confidence if none found
    logger.warning(f"No confidence attribute found for signal {signal.symbol}, using default 0.5")
    return to_decimal("0.5")


def validate_position_size(
    position_size: Decimal,
    portfolio_value: Decimal,
    min_position_pct: Decimal = to_decimal("0.01"),
    max_position_pct: Decimal = to_decimal("0.25"),
) -> tuple[bool, Decimal]:
    """
    Validate and adjust position size within bounds.

    Args:
        position_size: Calculated position size
        portfolio_value: Current portfolio value
        min_position_pct: Minimum position percentage
        max_position_pct: Maximum position percentage

    Returns:
        Tuple of (is_valid, adjusted_size)
    """
    if portfolio_value <= ZERO:
        return False, ZERO

    # Calculate position percentage
    position_pct = safe_divide(position_size, portfolio_value, ZERO)

    # Check minimum size
    if position_pct < min_position_pct:
        logger.warning(f"Position size {position_pct:.2%} below minimum {min_position_pct:.2%}")
        return False, ZERO

    # Adjust if exceeds maximum
    if position_pct > max_position_pct:
        adjusted_size = portfolio_value * max_position_pct
        logger.warning(
            f"Position size {position_pct:.2%} exceeds maximum {max_position_pct:.2%}, "
            f"adjusting to {max_position_pct:.2%}"
        )
        return True, adjusted_size

    return True, position_size


def calculate_position_size(
    method: PositionSizeMethod,
    signal: Signal,
    portfolio_value: Decimal,
    risk_per_trade: Decimal,
    **kwargs,
) -> Decimal:
    """
    Calculate position size using specified method.

    Args:
        method: Position sizing method
        signal: Trading signal
        portfolio_value: Current portfolio value
        risk_per_trade: Risk per trade as decimal
        **kwargs: Method-specific parameters

    Returns:
        Calculated position size
    """
    # Algorithm mapping
    algorithms = {
        PositionSizeMethod.FIXED_PERCENTAGE: FixedPercentageAlgorithm(),
        PositionSizeMethod.KELLY_CRITERION: KellyCriterionAlgorithm(),
        PositionSizeMethod.VOLATILITY_ADJUSTED: VolatilityAdjustedAlgorithm(),
        PositionSizeMethod.CONFIDENCE_WEIGHTED: ConfidenceWeightedAlgorithm(),
        PositionSizeMethod.ATR_BASED: ATRBasedAlgorithm(),
    }

    algorithm = algorithms.get(method)
    if not algorithm:
        logger.error(f"Unknown position sizing method: {method}")
        # Fallback to fixed percentage
        algorithm = FixedPercentageAlgorithm()

    try:
        # Calculate raw position size
        raw_size = algorithm.calculate_size(signal, portfolio_value, risk_per_trade, **kwargs)

        # Validate and adjust position size
        is_valid, adjusted_size = validate_position_size(raw_size, portfolio_value)

        if not is_valid:
            logger.warning(f"Position size validation failed for {signal.symbol}")
            return ZERO

        logger.debug(
            f"Position size calculated: {method.value}, "
            f"symbol={signal.symbol}, size={adjusted_size}"
        )

        return adjusted_size

    except Exception as e:
        logger.error(f"Position size calculation failed: {e}")
        return ZERO


def update_position_history(
    symbol: str, price: Decimal, position_history: dict[str, list[Decimal]], max_history: int = 252
) -> None:
    """
    Update position price history for volatility calculations.

    Args:
        symbol: Trading symbol
        price: Current price
        position_history: Position history dictionary
        max_history: Maximum history length
    """
    if not symbol or price <= ZERO:
        logger.warning(f"Invalid symbol or price for history update: {symbol}, {price}")
        return

    # Initialize symbol history if needed
    if symbol not in position_history:
        position_history[symbol] = []

    # Add new price
    position_history[symbol].append(price)

    # Maintain history size
    if len(position_history[symbol]) > max_history:
        position_history[symbol] = position_history[symbol][-max_history:]


def calculate_position_metrics(
    symbol: str, position_history: dict[str, list[Decimal]]
) -> dict[str, Any]:
    """
    Calculate position-specific metrics for sizing.

    Args:
        symbol: Trading symbol
        position_history: Position history dictionary

    Returns:
        Dictionary of position metrics
    """
    if symbol not in position_history or len(position_history[symbol]) < 10:
        return {
            "volatility": 0.02,  # Default 2% daily volatility
            "atr": to_decimal("0.02"),
            "returns_count": 0,
        }

    try:
        prices = position_history[symbol]

        # Calculate returns with Decimal precision
        returns_decimal = []
        for i in range(1, len(prices)):
            if prices[i - 1] > ZERO:
                return_val = safe_divide(prices[i] - prices[i - 1], prices[i - 1], ZERO)
                returns_decimal.append(return_val)

        if not returns_decimal:
            return {
                "volatility": to_decimal("0.02"),
                "atr": to_decimal("0.02"),
                "returns_count": 0,
            }

        # Convert to float for numpy calculation, but maintain Decimal precision for result
        returns_float = [float(r) for r in returns_decimal]
        volatility_float = np.std(returns_float) if len(returns_float) > 1 else 0.02
        volatility = to_decimal(str(volatility_float))

        # Estimate ATR (simplified as volatility * current price) with Decimal precision
        current_price = prices[-1]
        atr = current_price * volatility

        return {
            "volatility": volatility,
            "atr": atr,
            "returns_count": len(returns_decimal),
            "current_price": current_price,
        }

    except Exception as e:
        logger.error(f"Position metrics calculation failed for {symbol}: {e}")
        return {
            "volatility": to_decimal("0.02"),
            "atr": to_decimal("0.02"),
            "returns_count": 0,
        }
