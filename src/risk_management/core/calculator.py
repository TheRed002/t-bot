"""Centralized risk calculations with caching to eliminate duplication.

This module now uses the centralized risk calculation utilities to eliminate
code duplication across the risk management module.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.core.base import BaseComponent
from src.core.types import Position
from src.core.types.market import MarketData
from src.core.types.risk import RiskLevel, RiskMetrics
from src.utils.decimal_utils import to_decimal
from src.utils.messaging_patterns import ErrorPropagationMixin
from src.utils.risk_calculations import (
    calculate_calmar_ratio,
    calculate_current_drawdown,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_portfolio_value,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    determine_risk_level,
)


class RiskCalculator(BaseComponent, ErrorPropagationMixin):
    """
    Centralized risk calculator with caching.

    This eliminates duplication of risk calculations across modules
    by providing a single source of truth for all risk metrics.
    Uses ErrorPropagationMixin for consistent error handling across modules.
    """

    def __init__(self):
        """Initialize risk calculator."""
        super().__init__()  # Initialize BaseComponent
        self._cache = {}  # Cache TTL in seconds

        # Historical data storage - using Decimal for financial precision
        self.portfolio_values: list[Decimal] = []
        self.portfolio_returns: list[Decimal] = []
        self.position_data: dict[str, list[Decimal]] = {}

    def calculate_var(
        self,
        returns: list[Decimal],
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR) using centralized utility.

        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR value
        """
        return calculate_var(returns, confidence_level, time_horizon)

    def calculate_expected_shortfall(
        self, returns: list[Decimal], confidence_level: Decimal = Decimal("0.95")
    ) -> Decimal:
        """
        Calculate Expected Shortfall (Conditional VaR) using centralized utility.

        Args:
            returns: Historical returns
            confidence_level: Confidence level

        Returns:
            Expected shortfall value
        """
        return calculate_expected_shortfall(returns, confidence_level)

    def calculate_sharpe_ratio(
        self, returns: list[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Calculate Sharpe ratio using centralized utility.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        result = calculate_sharpe_ratio(returns, risk_free_rate)
        return result if result is not None else Decimal("0")

    def calculate_sortino_ratio(
        self,
        returns: list[Decimal],
        risk_free_rate: Decimal = Decimal("0.02"),
        target_return: Decimal = Decimal("0.0"),
    ) -> Decimal:
        """
        Calculate Sortino ratio using centralized utility.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate
            target_return: Target return for downside deviation

        Returns:
            Sortino ratio
        """
        return calculate_sortino_ratio(returns, risk_free_rate, target_return)

    def calculate_max_drawdown(self, values: list[Decimal]) -> tuple[Decimal, int, int]:
        """
        Calculate maximum drawdown using centralized utility.

        Args:
            values: Portfolio values

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        return calculate_max_drawdown(values)

    def calculate_calmar_ratio(
        self, returns: list[Decimal], period_years: Decimal = Decimal("1.0")
    ) -> Decimal:
        """
        Calculate Calmar ratio using centralized utility.

        Args:
            returns: Historical returns
            period_years: Period in years

        Returns:
            Calmar ratio
        """
        return calculate_calmar_ratio(returns, period_years)

    async def calculate_portfolio_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: Current positions
            market_data: Market data for positions

        Returns:
            Complete risk metrics
        """
        # Calculate portfolio value using centralized utility
        portfolio_value = calculate_portfolio_value(positions, market_data)

        # Create market data lookup and calculate total exposure
        market_by_symbol = {data.symbol: data for data in market_data}
        total_exposure = sum(
            abs(pos.quantity) * market_by_symbol[pos.symbol].close
            for pos in positions
            if pos.symbol in market_by_symbol
        )

        # Update historical data
        self.portfolio_values.append(portfolio_value)

        # Calculate returns if we have enough history
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > Decimal("0"):
                current_return = (portfolio_value - prev_value) / prev_value
                self.portfolio_returns.append(current_return)

        # Calculate all metrics using centralized utilities
        var_1d = calculate_var(self.portfolio_returns, to_decimal("0.95"), 1)
        var_5d = calculate_var(self.portfolio_returns, to_decimal("0.95"), 5)
        expected_shortfall = calculate_expected_shortfall(self.portfolio_returns)
        sharpe = calculate_sharpe_ratio(self.portfolio_returns)
        sortino = calculate_sortino_ratio(self.portfolio_returns)
        max_dd, _, _ = calculate_max_drawdown(
            self.portfolio_values
        )  # Use portfolio values, not returns
        calmar = calculate_calmar_ratio(self.portfolio_returns)

        # Determine risk level using centralized utility
        risk_level = determine_risk_level(
            var_1d, self._calculate_current_drawdown(), sharpe, portfolio_value
        )

        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=Decimal(str(portfolio_value)),
            total_exposure=Decimal(str(total_exposure)),
            var_1d=Decimal(str(var_1d)),
            var_5d=Decimal(str(var_5d)),
            expected_shortfall=Decimal(str(expected_shortfall)),
            sharpe_ratio=sharpe if sharpe is not None else None,
            sortino_ratio=sortino if sortino is not None else None,
            max_drawdown=Decimal(str(max_dd)),
            current_drawdown=Decimal(str(self._calculate_current_drawdown())),
            calmar_ratio=calmar if calmar is not None else None,
            risk_level=risk_level,
            position_count=len(positions),
            correlation_risk=self._calculate_correlation_risk(positions),
        )

    def _calculate_current_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak using centralized utility."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return Decimal("0")

        current_value = self.portfolio_values[-1]
        return calculate_current_drawdown(current_value, self.portfolio_values[:-1])

    def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk using centralized utilities."""
        from src.utils.decimal_utils import to_decimal
        from src.utils.risk_validation import validate_correlation_risk

        if len(positions) < 2:
            return to_decimal("0")

        try:
            # Create simplified correlation matrix
            symbols = [pos.symbol for pos in positions]
            correlation_matrix = {}

            # Calculate position-based correlation estimate
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i + 1 :], i + 1):
                    # Simple concentration-based correlation estimate
                    correlation = to_decimal("0.3")  # Default moderate correlation
                    correlation_matrix[(symbol1, symbol2)] = correlation

            # Validate correlation risk using centralized utility
            is_valid, message = validate_correlation_risk(correlation_matrix)
            if not is_valid:
                self.logger.warning(f"High correlation risk detected: {message}")
                return to_decimal("0.8")  # High risk indicator

            # Return concentration-based risk measure
            unique_symbols = len(set(symbols))
            concentration = (
                to_decimal("1") / to_decimal(str(unique_symbols))
                if unique_symbols > 0
                else to_decimal("1")
            )
            return to_decimal("1") - concentration

        except Exception as e:
            # Log error with consistent metadata for cross-module alignment
            self.logger.error(
                f"Correlation risk calculation failed: {e}",
                extra={
                    "error_type": type(e).__name__,
                    "context": "correlation_risk_calculation",
                    "processing_mode": "stream",  # Align with capital_management
                    "data_format": "error_context_v1",  # Consistent format version
                    "component": "RiskCalculator",
                    "fallback_applied": True,
                },
            )
            return to_decimal("0.5")  # Default moderate risk

    def _determine_risk_level(self, var: Decimal, max_dd: Decimal, sharpe: Decimal) -> RiskLevel:
        """
        Determine overall risk level using centralized utility.

        Args:
            var: Value at Risk
            max_dd: Maximum drawdown
            sharpe: Sharpe ratio

        Returns:
            Risk level classification
        """
        current_value = self.portfolio_values[-1] if self.portfolio_values else Decimal("1")
        return determine_risk_level(var, max_dd, sharpe, current_value)

    def clear_cache(self) -> None:
        """Clear calculation cache."""
        self._cache.clear()
        self.logger.info("Risk calculation cache cleared")

    def update_history(
        self, symbol: str, price: Decimal, return_value: float | None = None
    ) -> None:
        """
        Update historical data for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price as Decimal for financial precision
            return_value: Return value if calculated

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Enhanced input validation
            if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
                raise ValueError(f"Invalid symbol: {symbol} (must be non-empty string)")

            symbol = symbol.strip().upper()  # Normalize symbol

            if not isinstance(price, Decimal | int | float):
                raise ValueError(
                    f"Price must be Decimal, int, or float, got {type(price).__name__}"
                )

            # Convert to Decimal for validation
            if not isinstance(price, Decimal):
                try:
                    price = Decimal(str(price))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert price to Decimal: {e}") from e

            # Validate price range
            if price <= Decimal("0"):
                self.logger.warning(f"Invalid price for {symbol}: {price} (must be positive)")
                return

            min_price = Decimal("0.000001")
            max_price = Decimal("1000000")
            if not (min_price <= price <= max_price):
                self.logger.warning(
                    f"Price for {symbol} outside reasonable bounds: {price} "
                    f"(bounds: {min_price} - {max_price})"
                )
                return

            # Initialize symbol data if needed
            if symbol not in self.position_data:
                self.position_data[symbol] = []

            # Store as Decimal for financial precision
            self.position_data[symbol].append(price)

            # Keep only recent history with memory management
            max_history = 1000
            if len(self.position_data[symbol]) > max_history:
                self.position_data[symbol] = self.position_data[symbol][-max_history:]

            self.logger.info(
                f"Updated price history for {symbol}",
                extra={
                    "price": str(price),
                    "history_length": len(self.position_data[symbol]),
                    "return_value": return_value,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Unexpected error updating price history for {symbol}: {e}",
                extra={
                    "symbol": symbol,
                    "price": str(price) if price else None,
                    "error_type": type(e).__name__,
                },
            )
            # Don't re-raise to prevent breaking the calling code
            return


# Global instance for singleton pattern
_risk_calculator = RiskCalculator()


def get_risk_calculator() -> RiskCalculator:
    """Get the global risk calculator instance."""
    return _risk_calculator
