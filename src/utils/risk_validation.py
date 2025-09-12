"""
Centralized risk validation utilities to eliminate code duplication.

This module provides unified validation patterns for risk management,
eliminating duplication across multiple risk validation implementations.
"""

from decimal import Decimal
from typing import Any

from src.core.logging import get_logger
from src.core.types.risk import RiskLevel, RiskLimits
from src.core.types.trading import OrderRequest, Position, Signal, SignalDirection
from src.utils.decimal_utils import ONE, ZERO, safe_divide, to_decimal

logger = get_logger(__name__)


class UnifiedRiskValidator:
    """
    Centralized risk validator that eliminates validation code duplication.

    This class provides all risk validation patterns used across the system.
    """

    def __init__(self, risk_limits: RiskLimits | None = None):
        """
        Initialize unified risk validator.

        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits or self._get_default_limits()
        self._logger = logger

    def _get_default_limits(self) -> RiskLimits:
        """Get default risk limits."""
        return RiskLimits(
            max_position_size=to_decimal("10000"),
            max_portfolio_risk=to_decimal("0.1"),
            max_correlation=to_decimal("0.7"),
            max_leverage=to_decimal("3.0"),
            max_drawdown=to_decimal("0.2"),
            max_daily_loss=to_decimal("0.05"),
            max_positions=10,
            min_liquidity_ratio=to_decimal("0.2"),
        )

    def validate_signal(
        self,
        signal: Signal,
        current_risk_level: RiskLevel | None = None,
        emergency_stop_active: bool = False,
    ) -> tuple[bool, str]:
        """
        Validate trading signal against risk constraints.

        Args:
            signal: Signal to validate
            current_risk_level: Current portfolio risk level
            emergency_stop_active: Whether emergency stop is active

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check emergency stop
            if emergency_stop_active:
                return False, "Emergency stop is active"

            # Basic signal structure validation
            if not self._validate_signal_structure(signal):
                return False, "Invalid signal structure"

            # Check signal confidence/strength
            confidence = self._get_signal_confidence(signal)
            min_confidence = self._get_min_confidence_for_risk_level(current_risk_level)

            if confidence < min_confidence:
                return (
                    False,
                    f"Signal confidence {confidence:.2f} below minimum {min_confidence:.2f}",
                )

            # Check signal direction validity
            if not self._validate_signal_direction(signal):
                return False, "Invalid signal direction"

            # Check stop loss requirement for entry signals
            if not self._validate_stop_loss_requirement(signal):
                return False, "Signal must include stop loss for entry orders"

            return True, "Signal validation passed"

        except Exception as e:
            self._logger.error(f"Signal validation error: {e}")
            return False, f"Validation error: {e}"

    def validate_order(
        self,
        order: OrderRequest,
        portfolio_value: Decimal,
        current_positions: list[Position] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate order against risk limits.

        Args:
            order: Order to validate
            portfolio_value: Current portfolio value
            current_positions: Current open positions

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic order structure validation
            if not self._validate_order_structure(order):
                return False, "Invalid order structure"

            # Calculate order value
            order_value = self._calculate_order_value(order)
            if order_value <= ZERO:
                return False, "Invalid order value"

            # Check position size limit
            if order_value > self.risk_limits.max_position_size:
                return False, "Order value exceeds max position size limit"

            # Check portfolio risk limit
            if portfolio_value > ZERO:
                portfolio_risk = safe_divide(order_value, portfolio_value, ZERO)
                if portfolio_risk > self.risk_limits.max_portfolio_risk:
                    return False, f"Order risk {portfolio_risk:.2%} exceeds max portfolio risk"

            # Check position count limits
            if current_positions and len(current_positions) >= self.risk_limits.max_positions:
                # Check if adding to existing position
                existing_position = next(
                    (p for p in current_positions if p.symbol == order.symbol), None
                )
                if not existing_position:
                    return False, f"Maximum positions ({self.risk_limits.max_positions}) reached"

            # Check leverage limits
            leverage = getattr(order, "leverage", None)
            if leverage and leverage > self.risk_limits.max_leverage:
                return False, "Order leverage exceeds maximum allowed"

            return True, "Order validation passed"

        except Exception as e:
            self._logger.error(f"Order validation error: {e}")
            return False, f"Validation error: {e}"

    def validate_position(self, position: Position, portfolio_value: Decimal) -> tuple[bool, str]:
        """
        Validate position against risk limits.

        Args:
            position: Position to validate
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic position structure validation
            if not self._validate_position_structure(position):
                return False, "Invalid position structure"

            # Calculate position value
            if not position.current_price or position.current_price <= ZERO:
                return False, "Position missing valid current price"

            position_value = abs(position.quantity * position.current_price)

            # Check position concentration
            if portfolio_value > ZERO:
                concentration = safe_divide(position_value, portfolio_value, ZERO)
                max_concentration = to_decimal("0.3")  # 30% max concentration

                if concentration > max_concentration:
                    return False, f"Position concentration {concentration:.2%} exceeds maximum"

            # Check unrealized loss
            if position.unrealized_pnl and position.unrealized_pnl < ZERO:
                loss_percentage = safe_divide(abs(position.unrealized_pnl), position_value, ZERO)
                if loss_percentage > to_decimal("0.15"):  # 15% loss threshold
                    self._logger.warning(
                        f"Position {position.symbol} has significant loss: {loss_percentage:.2%}"
                    )

            return True, "Position validation passed"

        except Exception as e:
            self._logger.error(f"Position validation error: {e}")
            return False, f"Validation error: {e}"

    def validate_portfolio(self, portfolio_data: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate portfolio against risk limits.

        Args:
            portfolio_data: Portfolio metrics and data

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Extract portfolio metrics
            total_value = to_decimal(str(portfolio_data.get("total_value", 0)))
            positions = portfolio_data.get("positions", [])
            current_drawdown = to_decimal(str(portfolio_data.get("drawdown", 0)))
            daily_pnl = to_decimal(str(portfolio_data.get("daily_pnl", 0)))

            # Check drawdown limit
            if current_drawdown > self.risk_limits.max_drawdown:
                return False, f"Portfolio drawdown {current_drawdown:.2%} exceeds maximum"

            # Check daily loss limit
            if daily_pnl < ZERO and total_value > ZERO:
                daily_loss = safe_divide(abs(daily_pnl), total_value, ZERO)
                if daily_loss > self.risk_limits.max_daily_loss:
                    return False, f"Daily loss {daily_loss:.2%} exceeds maximum"

            # Check position count
            if len(positions) > self.risk_limits.max_positions:
                return False, f"Position count exceeds maximum ({self.risk_limits.max_positions})"

            # Check correlation risk
            correlation = to_decimal(str(portfolio_data.get("correlation", 0)))
            if correlation > self.risk_limits.max_correlation:
                self._logger.warning(f"High portfolio correlation detected: {correlation:.2f}")

            return True, "Portfolio validation passed"

        except Exception as e:
            self._logger.error(f"Portfolio validation error: {e}")
            return False, f"Validation error: {e}"

    def validate_risk_metrics(
        self, var_1d: Decimal, current_drawdown: Decimal, portfolio_value: Decimal
    ) -> tuple[bool, str]:
        """
        Validate risk metrics against thresholds.

        Args:
            var_1d: 1-day Value at Risk
            current_drawdown: Current portfolio drawdown
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            warnings = []
            critical_issues = []

            # Check VaR thresholds
            if portfolio_value > ZERO:
                var_pct = safe_divide(var_1d, portfolio_value, ZERO)

                if var_pct > to_decimal("0.10"):  # 10% VaR threshold
                    critical_issues.append(f"VaR {var_pct:.2%} is critically high")
                elif var_pct > to_decimal("0.05"):  # 5% VaR threshold
                    warnings.append(f"VaR {var_pct:.2%} is high")

            # Check drawdown thresholds
            if current_drawdown > self.risk_limits.max_drawdown:
                critical_issues.append(
                    f"Drawdown {current_drawdown:.2%} exceeds limit "
                    f"{self.risk_limits.max_drawdown:.2%}"
                )
            elif current_drawdown > to_decimal("0.10"):  # 10% warning threshold
                warnings.append(f"Drawdown {current_drawdown:.2%} is elevated")

            # Log warnings
            for warning in warnings:
                self._logger.warning(warning)

            # Return critical issues
            if critical_issues:
                return False, "; ".join(critical_issues)

            return True, "Risk metrics validation passed"

        except Exception as e:
            self._logger.error(f"Risk metrics validation error: {e}")
            return False, f"Validation error: {e}"

    # Private helper methods

    def _validate_signal_structure(self, signal: Signal) -> bool:
        """Validate basic signal structure."""
        if not signal:
            self._logger.error("Signal is None")
            return False

        if not hasattr(signal, "symbol") or not signal.symbol:
            self._logger.error("Signal missing symbol")
            return False

        if not hasattr(signal, "direction"):
            self._logger.error("Signal missing direction")
            return False

        return True

    def _validate_order_structure(self, order: OrderRequest) -> bool:
        """Validate basic order structure."""
        if not order:
            self._logger.error("Order is None")
            return False

        if not hasattr(order, "symbol") or not order.symbol:
            self._logger.error("Order missing symbol")
            return False

        if not hasattr(order, "side"):
            self._logger.error("Order missing side")
            return False

        if not hasattr(order, "quantity") or order.quantity <= ZERO:
            self._logger.error("Order missing or invalid quantity")
            return False

        return True

    def _validate_position_structure(self, position: Position) -> bool:
        """Validate basic position structure."""
        if not position:
            self._logger.error("Position is None")
            return False

        if not hasattr(position, "symbol") or not position.symbol:
            self._logger.error("Position missing symbol")
            return False

        if not hasattr(position, "quantity") or position.quantity == ZERO:
            self._logger.error("Position missing or zero quantity")
            return False

        return True

    def _get_signal_confidence(self, signal: Signal) -> Decimal:
        """Extract confidence/strength from signal."""
        for attr_name in ["confidence", "strength", "score"]:
            if hasattr(signal, attr_name):
                confidence_value = getattr(signal, attr_name)
                if confidence_value is not None:
                    confidence = to_decimal(str(confidence_value))
                    # Ensure within valid range
                    return max(ZERO, min(confidence, ONE))

        # Default confidence
        return to_decimal("0.5")

    def _get_min_confidence_for_risk_level(self, risk_level: RiskLevel | None) -> Decimal:
        """Get minimum confidence based on risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return to_decimal("0.8")
        elif risk_level == RiskLevel.HIGH:
            return to_decimal("0.6")
        elif risk_level == RiskLevel.MEDIUM:
            return to_decimal("0.4")
        else:
            return to_decimal("0.3")  # Low or None

    def _validate_signal_direction(self, signal: Signal) -> bool:
        """Validate signal direction."""
        if not hasattr(signal, "direction"):
            return False

        valid_directions = [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        return signal.direction in valid_directions

    def _validate_stop_loss_requirement(self, signal: Signal) -> bool:
        """Validate stop loss requirement for entry signals."""
        # Only entry signals (BUY/SELL) require stop loss
        if hasattr(signal, "direction"):
            if signal.direction in [SignalDirection.BUY, SignalDirection.SELL]:
                return hasattr(signal, "stop_loss") and signal.stop_loss is not None

        return True  # No stop loss required for other signal types

    def _calculate_order_value(self, order: OrderRequest) -> Decimal:
        """Calculate order value."""
        quantity = order.quantity

        # Use order price if available, otherwise assume it will be validated elsewhere
        price = getattr(order, "price", None)
        if price and price > ZERO:
            return quantity * price

        # For market orders without price, return quantity as approximate value
        return quantity

    def update_limits(self, new_limits: RiskLimits) -> None:
        """
        Update risk limits.

        Args:
            new_limits: New risk limits configuration
        """
        self.risk_limits = new_limits
        self._logger.info("Risk limits updated")


def validate_financial_inputs(**kwargs) -> tuple[bool, str]:
    """
    Validate common financial calculation inputs.

    Args:
        **kwargs: Financial parameters to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate portfolio value
        portfolio_value = kwargs.get("portfolio_value")
        if portfolio_value is not None:
            if not isinstance(portfolio_value, Decimal) or portfolio_value <= ZERO:
                return False, "Invalid portfolio value"

        # Validate prices
        for price_key in ["price", "entry_price", "current_price", "stop_loss", "take_profit"]:
            price = kwargs.get(price_key)
            if price is not None:
                if not isinstance(price, Decimal) or price <= ZERO:
                    return False, f"Invalid {price_key}"

        # Validate quantities
        quantity = kwargs.get("quantity")
        if quantity is not None:
            if not isinstance(quantity, Decimal) or quantity == ZERO:
                return False, "Invalid quantity"

        # Validate percentages
        for pct_key in ["confidence", "risk_per_trade", "drawdown"]:
            pct = kwargs.get(pct_key)
            if pct is not None:
                if not isinstance(pct, Decimal) or pct < ZERO or pct > ONE:
                    return False, f"Invalid {pct_key} percentage"

        return True, "Financial inputs valid"

    except Exception as e:
        logger.error(f"Financial input validation error: {e}")
        return False, f"Validation error: {e}"


def check_position_limits(
    new_position_count: int,
    max_positions: int,
    symbol: str | None = None,
    symbol_positions: int = 0,
    max_per_symbol: int = 3,
) -> tuple[bool, str]:
    """
    Check position count limits.

    Args:
        new_position_count: Total positions after adding new position
        max_positions: Maximum total positions allowed
        symbol: Symbol for the new position
        symbol_positions: Current positions for the symbol
        max_per_symbol: Maximum positions per symbol

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check total position limit
    if new_position_count > max_positions:
        return False, f"Total position limit ({max_positions}) would be exceeded"

    # Check per-symbol limit
    if symbol and symbol_positions >= max_per_symbol:
        return False, f"Symbol position limit ({max_per_symbol}) reached for {symbol}"

    return True, "Position limits check passed"


def validate_correlation_risk(
    correlation_matrix: dict[tuple[str, str], Decimal], max_correlation: Decimal = to_decimal("0.7")
) -> tuple[bool, str]:
    """
    Validate portfolio correlation risk.

    Args:
        correlation_matrix: Pairwise correlation matrix
        max_correlation: Maximum allowed correlation

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        high_correlations = []

        for (symbol1, symbol2), correlation in correlation_matrix.items():
            if abs(correlation) > max_correlation:
                high_correlations.append(f"{symbol1}-{symbol2}: {correlation:.2f}")

        if high_correlations:
            return False, f"High correlations detected: {', '.join(high_correlations[:3])}"

        return True, "Correlation risk validation passed"

    except Exception as e:
        logger.error(f"Correlation validation error: {e}")
        return False, f"Validation error: {e}"
