"""
Arbitrage Helper Utilities - Common functions extracted from arbitrage strategies.

This module provides shared functionality for arbitrage strategies including fee calculations,
spread analysis, and opportunity prioritization.

CRITICAL: All utilities follow the coding standards and use proper financial precision.
"""

from decimal import Decimal
from typing import Any

from src.core.exceptions import ArbitrageError, ValidationError
from src.core.logging import get_logger
from src.utils.constants import GLOBAL_FEE_STRUCTURE, PRECISION_LEVELS
from src.utils.decimal_utils import ZERO, round_to_precision, safe_divide, to_decimal
from src.utils.decorators import log_errors, time_execution

# Import functions when needed dynamically to avoid circular imports

logger = get_logger(__name__)


class FeeCalculator:
    """
    Common fee calculation utilities for arbitrage strategies.

    Provides standardized fee calculations for cross-exchange and triangular arbitrage
    with proper validation and error handling.
    """

    @staticmethod
    @log_errors
    @time_execution
    def calculate_cross_exchange_fees(buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """
        Calculate fees for cross-exchange arbitrage using proper validation and formatting.

        Args:
            buy_price: Price to buy at
            sell_price: Price to sell at

        Returns:
            Total estimated fees

        Raises:
            ValidationError: If prices are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input prices
            # Lazy import to avoid circular dependency
            from src.utils.validators import ValidationFramework

            ValidationFramework.validate_price(buy_price)
            ValidationFramework.validate_price(sell_price)

            # Get fee structure from constants
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%

            # Calculate fees with proper rounding
            buy_fees = round_to_precision(buy_price * taker_fee_rate, PRECISION_LEVELS["fee"])
            sell_fees = round_to_precision(sell_price * taker_fee_rate, PRECISION_LEVELS["fee"])

            # Calculate slippage cost
            slippage_cost = round_to_precision(
                (sell_price - buy_price) * Decimal("0.0005"), PRECISION_LEVELS["price"]
            )

            # Calculate total fees
            total_fees = buy_fees + sell_fees + slippage_cost

            # Lazy import to avoid circular dependency
            from src.utils.formatters import format_currency

            logger.debug(
                "Cross-exchange fee calculation completed",
                buy_price=format_currency(buy_price),
                sell_price=format_currency(sell_price),
                buy_fees=format_currency(buy_fees),
                sell_fees=format_currency(sell_fees),
                slippage_cost=format_currency(slippage_cost),
                total_fees=format_currency(total_fees),
            )

            return total_fees

        except Exception as e:
            logger.error(
                "Cross-exchange fee calculation failed",
                buy_price=str(buy_price),
                sell_price=str(sell_price),
                error=str(e),
            )
            raise ArbitrageError(f"Cross-exchange fee calculation failed: {e!s}")

    @staticmethod
    @log_errors
    @time_execution
    def calculate_triangular_fees(rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal:
        """
        Calculate fees for triangular arbitrage using proper validation and formatting.

        Args:
            rate1: First pair rate
            rate2: Second pair rate
            rate3: Third pair rate

        Returns:
            Total estimated fees

        Raises:
            ValidationError: If rates are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input rates
            # Lazy import to avoid circular dependency
            from src.utils.validators import ValidationFramework

            for rate, name in [(rate1, "rate1"), (rate2, "rate2"), (rate3, "rate3")]:
                ValidationFramework.validate_price(rate)

            # Get fee structure from constants
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%

            # Calculate fees for each step
            step1_fees = round_to_precision(rate1 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step2_fees = round_to_precision(rate2 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step3_fees = round_to_precision(rate3 * taker_fee_rate, PRECISION_LEVELS["fee"])

            # Calculate slippage costs for each step
            slippage_cost1 = round_to_precision(
                rate1 * Decimal("0.0005"), PRECISION_LEVELS["price"]
            )
            slippage_cost2 = round_to_precision(
                rate2 * Decimal("0.0005"), PRECISION_LEVELS["price"]
            )
            slippage_cost3 = round_to_precision(
                rate3 * Decimal("0.0005"), PRECISION_LEVELS["price"]
            )

            # Calculate total fees
            total_fees = (
                step1_fees
                + step2_fees
                + step3_fees
                + slippage_cost1
                + slippage_cost2
                + slippage_cost3
            )

            # Lazy import to avoid circular dependency
            from src.utils.formatters import format_currency

            logger.debug(
                "Triangular fee calculation completed",
                rate1=format_currency(rate1),
                rate2=format_currency(rate2),
                rate3=format_currency(rate3),
                step1_fees=format_currency(step1_fees),
                step2_fees=format_currency(step2_fees),
                step3_fees=format_currency(step3_fees),
                slippage_costs=[
                    format_currency(cost)
                    for cost in [slippage_cost1, slippage_cost2, slippage_cost3]
                ],
                total_fees=format_currency(total_fees),
            )

            return total_fees

        except Exception as e:
            logger.error(
                "Triangular fee calculation failed",
                rate1=str(rate1),
                rate2=str(rate2),
                rate3=str(rate3),
                error=str(e),
            )
            raise ArbitrageError(f"Triangular fee calculation failed: {e!s}")


class OpportunityAnalyzer:
    """
    Common opportunity analysis utilities for arbitrage strategies.
    """

    @staticmethod
    @log_errors
    @time_execution
    def calculate_priority(profit_percentage: Decimal, arbitrage_type: str) -> Decimal:
        """
        Calculate opportunity priority based on profit and type.

        Args:
            profit_percentage: Profit percentage
            arbitrage_type: Type of arbitrage (cross_exchange or triangular)

        Returns:
            Priority score (higher is better)

        Raises:
            ValidationError: If inputs are invalid
            ArbitrageError: If priority calculation fails
        """
        try:
            # Convert profit_percentage to float if it's a Decimal
            if hasattr(profit_percentage, "__float__"):
                profit_percentage = float(profit_percentage)

            # Validate inputs
            if profit_percentage < 0.0 or profit_percentage > 10.0:  # 0-1000% range
                raise ValidationError(f"Invalid profit percentage: {profit_percentage}")

            if arbitrage_type not in ["cross_exchange", "triangular"]:
                raise ValidationError(f"Invalid arbitrage type: {arbitrage_type}")

            # Base priority on profit percentage
            base_priority = profit_percentage

            # Adjust for arbitrage type (cross-exchange typically more reliable)
            if arbitrage_type == "cross_exchange":
                type_multiplier = 1.2
            else:
                type_multiplier = 1.0

            # Adjust for market conditions (placeholder for future enhancement)
            market_multiplier = 1.0

            # Calculate final priority
            priority = base_priority * type_multiplier * market_multiplier

            # Validate final result
            if priority < 0:
                priority = 0
            elif priority > 1000:  # Reasonable upper limit
                priority = 1000

            # Lazy import to avoid circular dependency
            from src.utils.formatters import format_percentage

            logger.debug(
                "Priority calculation completed",
                profit_percentage=format_percentage(Decimal(str(profit_percentage))),
                arbitrage_type=arbitrage_type,
                type_multiplier=type_multiplier,
                market_multiplier=market_multiplier,
                final_priority=priority,
            )

            return priority

        except Exception as e:
            logger.error(
                "Priority calculation failed",
                profit_percentage=profit_percentage,
                arbitrage_type=arbitrage_type,
                error=str(e),
            )
            raise ArbitrageError(f"Priority calculation failed: {e!s}")

    @staticmethod
    def prioritize_opportunities(signals: list[Any], max_opportunities: int = 10) -> list[Any]:
        """
        Prioritize arbitrage opportunities by profit potential and feasibility.

        Args:
            signals: List of arbitrage signals
            max_opportunities: Maximum number of opportunities to return

        Returns:
            Prioritized list of signals
        """
        try:
            # Sort by priority (highest first)
            sorted_signals = sorted(
                signals,
                key=lambda s: s.metadata.get("opportunity_priority", 0)
                if hasattr(s, "metadata")
                else 0,
                reverse=True,
            )

            # Limit to maximum opportunities
            limited_signals = sorted_signals[:max_opportunities]

            # Log top opportunities
            if limited_signals:
                top_opportunity = limited_signals[0]
                if hasattr(top_opportunity, "metadata"):
                    logger.info(
                        "Top arbitrage opportunity",
                        symbol=getattr(top_opportunity, "symbol", "unknown"),
                        arbitrage_type=top_opportunity.metadata.get("arbitrage_type"),
                        profit_percentage=top_opportunity.metadata.get("net_profit_percentage"),
                        priority=top_opportunity.metadata.get("opportunity_priority"),
                    )

            return limited_signals

        except Exception as e:
            logger.error("Opportunity prioritization failed", error=str(e))
            return signals[:max_opportunities]  # Return first N on error


class SpreadAnalyzer:
    """
    Common spread analysis utilities for arbitrage strategies.
    """

    @staticmethod
    def calculate_spread_percentage(buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """
        Calculate spread as a percentage.

        Args:
            buy_price: Buy price
            sell_price: Sell price

        Returns:
            Spread percentage
        """
        try:
            if buy_price <= ZERO:
                return ZERO

            spread = sell_price - buy_price
            spread_percentage = safe_divide(spread, buy_price, ZERO) * to_decimal("100")

            return spread_percentage

        except Exception as e:
            logger.error("Spread percentage calculation failed", error=str(e))
            return ZERO

    @staticmethod
    def calculate_net_profit(
        gross_spread: Decimal, fees: Decimal, base_price: Decimal
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate net profit and profit percentage after fees.

        Args:
            gross_spread: Gross spread amount
            fees: Total fees
            base_price: Base price for percentage calculation

        Returns:
            Tuple of (net_profit_amount, net_profit_percentage)
        """
        try:
            net_profit_amount = gross_spread - fees

            if base_price <= ZERO:
                return ZERO, ZERO

            net_profit_percentage = safe_divide(net_profit_amount, base_price, ZERO) * to_decimal(
                "100"
            )

            return net_profit_amount, net_profit_percentage

        except Exception as e:
            logger.error("Net profit calculation failed", error=str(e))
            return ZERO, ZERO


class PositionSizingCalculator:
    """
    Position sizing utilities for arbitrage strategies.
    """

    @staticmethod
    @log_errors
    def calculate_arbitrage_position_size(
        total_capital: Decimal,
        risk_per_trade: Decimal,
        max_position_size: Decimal,
        profit_potential: Decimal,
        signal_strength: Decimal,
        arbitrage_type: str,
    ) -> Decimal:
        """
        Calculate position size for arbitrage opportunity.

        Args:
            total_capital: Total available capital
            risk_per_trade: Risk percentage per trade (0-1)
            max_position_size: Maximum position size percentage (0-1)
            profit_potential: Expected profit as percentage (0-100)
            signal_strength: Signal strength (0-1)
            arbitrage_type: Type of arbitrage

        Returns:
            Calculated position size

        Raises:
            ValidationError: If inputs are invalid
            ArbitrageError: If calculation fails
        """
        try:
            # Validate inputs
            # Lazy import to avoid circular dependency
            from src.utils.validators import ValidationFramework

            ValidationFramework.validate_quantity(total_capital)

            if not (0.0 <= risk_per_trade <= 1.0):
                raise ValidationError(f"Invalid risk per trade: {risk_per_trade}")

            if not (0.0 <= max_position_size <= 1.0):
                raise ValidationError(f"Invalid max position size: {max_position_size}")

            if not (0.0 <= signal_strength <= 1.0):
                raise ValidationError(f"Invalid signal strength: {signal_strength}")

            # Calculate base position size
            base_size = total_capital * Decimal(str(risk_per_trade))

            # Apply maximum position size limit
            max_size = total_capital * Decimal(str(max_position_size))
            if base_size > max_size:
                base_size = max_size

            # Adjust for arbitrage type
            if arbitrage_type == "cross_exchange":
                type_multiplier = Decimal("1.0")
            else:  # triangular
                type_multiplier = Decimal("0.7")  # More complex, smaller size

            # Scale by profit potential and signal strength
            profit_multiplier = min(Decimal("2.0"), profit_potential / Decimal("10"))
            strength_multiplier = Decimal(str(signal_strength))

            # Calculate final position size
            position_size = round_to_precision(
                base_size * strength_multiplier * profit_multiplier * type_multiplier,
                PRECISION_LEVELS.get("position", 8),
            )

            # Ensure minimum position size
            min_size = total_capital * Decimal("0.001")  # 0.1% minimum
            if position_size < min_size:
                position_size = min_size

            # Lazy import to avoid circular dependency
            from src.utils.formatters import format_currency, format_percentage

            logger.debug(
                "Arbitrage position size calculated",
                base_size=format_currency(base_size),
                profit_potential=format_percentage(profit_potential),
                strength=format_percentage(Decimal(str(signal_strength * 100))),
                arbitrage_type=arbitrage_type,
                final_size=format_currency(position_size),
            )

            return position_size

        except Exception as e:
            logger.error(
                "Arbitrage position size calculation failed",
                total_capital=float(total_capital),
                error=str(e),
            )
            raise ArbitrageError(f"Position size calculation failed: {e!s}")


class MarketDataValidator:
    """
    Market data validation utilities for arbitrage strategies.
    """

    @staticmethod
    def validate_price_data(price_data: dict[str, Any]) -> bool:
        """
        Validate market price data for arbitrage calculations.

        Args:
            price_data: Dictionary containing price information

        Returns:
            True if data is valid for arbitrage
        """
        try:
            required_fields = ["bid", "ask", "price"]

            for field in required_fields:
                if field not in price_data or price_data[field] is None:
                    logger.warning("Missing price field", field=field)
                    return False

                if price_data[field] <= 0:
                    logger.warning("Invalid price value", field=field, value=price_data[field])
                    return False

            # Check bid-ask spread sanity
            bid = price_data["bid"]
            ask = price_data["ask"]

            if bid >= ask:
                logger.warning("Invalid bid-ask spread", bid=bid, ask=ask)
                return False

            # Check price is within bid-ask range
            price = price_data["price"]
            if not (bid <= price <= ask):
                logger.warning("Price outside bid-ask range", price=price, bid=bid, ask=ask)
                return False

            return True

        except Exception as e:
            logger.error("Price data validation failed", error=str(e))
            return False

    @staticmethod
    def check_arbitrage_thresholds(
        profit_percentage: float,
        min_threshold: float,
        execution_time_ms: float,
        max_execution_time: float = 500.0,
    ) -> bool:
        """
        Check if arbitrage opportunity meets execution thresholds.

        Args:
            profit_percentage: Expected profit percentage
            min_threshold: Minimum profit threshold
            execution_time_ms: Expected execution time in milliseconds
            max_execution_time: Maximum acceptable execution time

        Returns:
            True if opportunity meets thresholds
        """
        try:
            # Check profit threshold
            if profit_percentage < min_threshold:
                logger.debug(
                    "Profit below threshold", profit=profit_percentage, threshold=min_threshold
                )
                return False

            # Check execution time
            if execution_time_ms > max_execution_time:
                logger.debug(
                    "Execution time too high",
                    execution_time=execution_time_ms,
                    max_time=max_execution_time,
                )
                return False

            return True

        except Exception as e:
            logger.error("Threshold check failed", error=str(e))
            return False
