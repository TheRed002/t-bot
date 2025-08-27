"""Centralized risk calculations with caching to eliminate duplication."""

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

from src.core.base.component import BaseComponent
from src.core.types.market import MarketData
from src.core.types.risk import RiskLevel, RiskMetrics
from src.core.types.trading import Position


class RiskCalculator(BaseComponent):
    """
    Centralized risk calculator with caching.

    This eliminates duplication of risk calculations across modules
    by providing a single source of truth for all risk metrics.
    """

    def __init__(self):
        """Initialize risk calculator."""
        super().__init__()  # Initialize BaseComponent
        self._cache = {}  # Cache TTL in seconds

        # Historical data storage
        self.portfolio_values: list[float] = []
        self.portfolio_returns: list[float] = []
        self.position_data: dict[str, list[float]] = {}

    def calculate_var(
        self, returns: list[float], confidence_level: float = 0.95, time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR value

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Enhanced input validation
            if not returns:
                self.logger.warning("Empty returns list provided for VaR calculation")
                return 0.0

            if not isinstance(returns, list | tuple):
                raise ValueError(f"Returns must be a list or tuple, got {type(returns).__name__}")

            if len(returns) < 10:  # Need at least 10 returns for reliable VaR
                self.logger.warning(
                    f"Insufficient returns data for VaR: {len(returns)} returns (minimum 10 required)"
                )
                return 0.0

            # Validate confidence level with detailed error message
            if not isinstance(confidence_level, (int, float)):
                raise ValueError(
                    f"Confidence level must be numeric, got {type(confidence_level).__name__}"
                )

            if not (0.5 <= confidence_level <= 0.999):
                self.logger.warning(
                    f"Invalid confidence level {confidence_level} (must be between 0.5 and 0.999), using 0.95"
                )
                confidence_level = 0.95

            # Validate time horizon
            if not isinstance(time_horizon, int) or time_horizon <= 0:
                self.logger.warning(
                    f"Invalid time horizon {time_horizon} (must be positive integer), using 1 day"
                )
                time_horizon = 1

            # Validate and clean returns data
            valid_returns = []
            for i, ret in enumerate(returns):
                try:
                    float_ret = float(ret)
                    if not (np.isnan(float_ret) or np.isinf(float_ret)):
                        # Apply reasonable bounds for returns (-100% to +1000%)
                        if -1.0 <= float_ret <= 10.0:
                            valid_returns.append(float_ret)
                        else:
                            self.logger.warning(
                                f"Extreme return value at index {i}: {float_ret}, excluding from calculation"
                            )
                    else:
                        self.logger.warning(
                            f"Invalid return value at index {i}: {ret} (NaN or Inf), excluding from calculation"
                        )
                except (TypeError, ValueError) as e:
                    self.logger.warning(
                        f"Cannot convert return at index {i} to float: {ret}, error: {e}"
                    )
                    continue

            if not valid_returns or len(valid_returns) < 10:
                self.logger.warning(
                    f"Insufficient valid returns for VaR: {len(valid_returns)} valid out of {len(returns)} total"
                )
                return 0.0

            # Convert to numpy array for calculations
            returns_array = np.array(valid_returns)

            # Calculate VaR using percentile method
            var_percentile = (1 - confidence_level) * 100

            try:
                var_daily = np.percentile(returns_array, var_percentile)
            except Exception as e:
                self.logger.error(f"Error calculating percentile: {e}")
                return 0.0

            # Validate percentile result
            if np.isnan(var_daily) or np.isinf(var_daily):
                self.logger.warning("Invalid VaR percentile calculated (NaN or Inf)")
                return 0.0

            # Scale to time horizon with bounds checking
            try:
                scaling_factor = np.sqrt(time_horizon)
                if np.isnan(scaling_factor) or np.isinf(scaling_factor):
                    self.logger.warning("Invalid scaling factor calculated")
                    scaling_factor = 1.0
                var_scaled = var_daily * scaling_factor
            except Exception as e:
                self.logger.error(f"Error scaling VaR to time horizon: {e}")
                return 0.0

            # Apply reasonable bounds (max 50% VaR)
            var_result = min(abs(var_scaled), 0.5)

            # Final validation
            if np.isnan(var_result) or np.isinf(var_result) or var_result < 0:
                self.logger.warning(f"Invalid final VaR result: {var_result}, returning 0")
                return 0.0

            return var_result

        except Exception as e:
            self.logger.error(
                f"Unexpected error in VaR calculation: {e}",
                extra={
                    "returns_count": len(returns) if returns else 0,
                    "confidence_level": confidence_level,
                    "time_horizon": time_horizon,
                    "error_type": type(e).__name__,
                },
            )
            return 0.0

    def calculate_expected_shortfall(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: Historical returns
            confidence_level: Confidence level

        Returns:
            Expected shortfall value
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns_array, var_percentile)

        # Calculate mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return abs(var_threshold)

        return abs(np.mean(tail_returns))

    def calculate_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)

        # Validate returns data
        if np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array)):
            self.logger.warning("Invalid returns data for Sharpe ratio calculation")
            return 0.0

        excess_returns = returns_array - risk_free_rate
        volatility = np.std(returns_array)

        # Check for zero or near-zero volatility
        if volatility == 0 or volatility < 1e-8:
            self.logger.debug("Zero or near-zero volatility, Sharpe ratio set to 0")
            return 0.0

        sharpe = np.mean(excess_returns) / volatility

        # Validate result
        if np.isnan(sharpe) or np.isinf(sharpe):
            self.logger.warning("Invalid Sharpe ratio calculated")
            return 0.0

        return sharpe

    def calculate_sortino_ratio(
        self, returns: list[float], risk_free_rate: float = 0.02, target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate
            target_return: Target return for downside deviation

        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        # Calculate downside deviation
        downside_returns = np.minimum(returns_array - target_return, 0)
        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            return 0.0

        return np.mean(excess_returns) / downside_deviation

    def calculate_max_drawdown(self, values: list[float]) -> tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Args:
            values: Portfolio values

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if not values or len(values) < 2:
            return 0.0, 0, 0

        values_array = np.array(values)
        cumulative_returns = (1 + values_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = np.min(drawdown)
        trough_index = np.argmin(drawdown)

        # Find the peak before the trough
        peak_index = np.argmax(cumulative_returns[: trough_index + 1])

        return abs(max_drawdown), peak_index, trough_index

    def calculate_calmar_ratio(self, returns: list[float], period_years: float = 1.0) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Historical returns
            period_years: Period in years

        Returns:
            Calmar ratio
        """
        if not returns:
            return 0.0

        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd, _, _ = self.calculate_max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

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
        # Calculate portfolio value and total exposure with safe symbol matching
        market_by_symbol = {md.symbol: md for md in market_data}

        portfolio_value = sum(
            float(pos.quantity) * float(market_by_symbol[pos.symbol].close)
            for pos in positions
            if pos.symbol in market_by_symbol
        )

        # Total exposure is the sum of absolute position values
        total_exposure = sum(
            abs(float(pos.quantity)) * float(market_by_symbol[pos.symbol].close)
            for pos in positions
            if pos.symbol in market_by_symbol
        )

        # Update historical data
        self.portfolio_values.append(portfolio_value)

        # Calculate returns if we have enough history
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            self.portfolio_returns = returns.tolist()

        # Calculate all metrics
        var_1d = self.calculate_var(self.portfolio_returns, 0.95, 1)
        var_5d = self.calculate_var(self.portfolio_returns, 0.95, 5)
        expected_shortfall = self.calculate_expected_shortfall(self.portfolio_returns)
        sharpe = self.calculate_sharpe_ratio(self.portfolio_returns)
        sortino = self.calculate_sortino_ratio(self.portfolio_returns)
        max_dd, _, _ = self.calculate_max_drawdown(self.portfolio_returns)
        calmar = self.calculate_calmar_ratio(self.portfolio_returns)

        # Determine risk level
        risk_level = self._determine_risk_level(var_1d, max_dd, sharpe)

        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=Decimal(str(portfolio_value)),
            total_exposure=Decimal(str(total_exposure)),
            var_1d=Decimal(str(var_1d)),
            var_5d=Decimal(str(var_5d)),
            expected_shortfall=Decimal(str(expected_shortfall)),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=Decimal(str(max_dd)),
            current_drawdown=Decimal(str(self._calculate_current_drawdown())),
            calmar_ratio=calmar,
            risk_level=risk_level,
            position_count=len(positions),
            correlation_risk=self._calculate_correlation_risk(positions),
        )

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0

        peak = max(self.portfolio_values)
        current = self.portfolio_values[-1]

        if peak == 0:
            return 0.0

        return (peak - current) / peak

    def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return Decimal("0")

        # Simplified correlation risk calculation
        # In production, this would use actual correlation matrices
        unique_symbols = len(set(pos.symbol for pos in positions))
        concentration = 1.0 / unique_symbols if unique_symbols > 0 else 1.0

        return Decimal(str(1 - concentration))

    def _determine_risk_level(self, var: float, max_dd: float, sharpe: float) -> RiskLevel:
        """
        Determine overall risk level based on metrics.

        Args:
            var: Value at Risk
            max_dd: Maximum drawdown
            sharpe: Sharpe ratio

        Returns:
            Risk level classification
        """
        risk_score = 0

        # VaR scoring
        if var > 0.1:  # > 10%
            risk_score += 3
        elif var > 0.05:  # > 5%
            risk_score += 2
        elif var > 0.02:  # > 2%
            risk_score += 1

        # Drawdown scoring
        if max_dd > 0.2:  # > 20%
            risk_score += 3
        elif max_dd > 0.1:  # > 10%
            risk_score += 2
        elif max_dd > 0.05:  # > 5%
            risk_score += 1

        # Sharpe ratio scoring (inverse)
        if sharpe < 0.5:
            risk_score += 3
        elif sharpe < 1.0:
            risk_score += 2
        elif sharpe < 1.5:
            risk_score += 1

        # Map score to risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def clear_cache(self) -> None:
        """Clear calculation cache."""
        self._cache.clear()
        self.logger.debug("Risk calculation cache cleared")

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

            if not isinstance(price, (Decimal, int, float)):
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

            # Apply reasonable price bounds (between $0.000001 and $1,000,000)
            min_price = Decimal("0.000001")
            max_price = Decimal("1000000")
            if not (min_price <= price <= max_price):
                self.logger.warning(
                    f"Price for {symbol} outside reasonable bounds: {price} (bounds: {min_price} - {max_price})"
                )
                return

            # Initialize symbol data if needed
            if symbol not in self.position_data:
                self.position_data[symbol] = []

            # Convert to float only for storage (numpy compatibility)
            try:
                price_float = float(price)
                if np.isnan(price_float) or np.isinf(price_float):
                    self.logger.warning(
                        f"Price conversion resulted in NaN or Inf for {symbol}: {price}"
                    )
                    return
                self.position_data[symbol].append(price_float)
            except (TypeError, ValueError, OverflowError) as e:
                self.logger.error(f"Error converting price to float for {symbol}: {e}")
                return

            # Keep only recent history with memory management
            max_history = 1000
            if len(self.position_data[symbol]) > max_history:
                self.position_data[symbol] = self.position_data[symbol][-max_history:]

            # Log successful update for debugging
            self.logger.debug(
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
