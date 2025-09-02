"""
Correlation Monitoring System for Real-time Portfolio Risk Management.

This module implements comprehensive correlation monitoring for portfolio positions
to detect systemic risk events where multiple positions may move against the portfolio
simultaneously. High correlation indicates exposure to common risk factors.

CRITICAL: This integrates with circuit breaker system and position limits.
Uses Decimal precision for all financial calculations.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import RiskManagementError
from src.core.types import MarketData, Position


class CorrelationLevel(Enum):
    """Correlation level classification."""

    HIGH = "high_correlation"
    MEDIUM = "medium_correlation"
    LOW = "low_correlation"
    UNKNOWN = "unknown"


@dataclass
class CorrelationThresholds:
    """Configuration for correlation-based risk thresholds."""

    warning_threshold: Decimal = Decimal("0.6")  # 60% correlation warning
    critical_threshold: Decimal = Decimal("0.8")  # 80% correlation critical
    max_positions_high_corr: int = 3  # Max positions when correlation > warning
    max_positions_critical_corr: int = 1  # Max positions when correlation > critical
    lookback_periods: int = 50  # Periods for correlation calculation
    min_periods: int = 10  # Minimum periods required


@dataclass
class CorrelationMetrics:
    """Correlation metrics for portfolio analysis."""

    average_correlation: Decimal
    max_pairwise_correlation: Decimal
    correlation_spike: bool
    correlated_pairs_count: int
    portfolio_concentration_risk: Decimal
    timestamp: datetime
    correlation_matrix: dict[tuple[str, str], Decimal]


class CorrelationMonitor(BaseComponent):
    """
    Real-time correlation monitoring system for portfolio positions.

    Monitors rolling correlation between portfolio positions to detect
    periods of high correlation that indicate systemic risk exposure.
    """

    def __init__(self, config: Config, thresholds: CorrelationThresholds | None = None):
        """
        Initialize correlation monitor.

        Args:
            config: Application configuration
            thresholds: Correlation thresholds, uses defaults if None
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.thresholds = thresholds or CorrelationThresholds()

        self.price_history: dict[str, deque[tuple[Decimal, datetime]]] = defaultdict(
            lambda: deque(maxlen=self.thresholds.lookback_periods)
        )

        # Return history storage: symbol -> deque of returns
        self.return_history: dict[str, deque[Decimal]] = defaultdict(
            lambda: deque(maxlen=self.thresholds.lookback_periods)
        )

        # Correlation cache with timestamps for performance
        self._correlation_cache: dict[tuple[str, str], tuple[Decimal, datetime]] = {}
        self._cache_timeout = timedelta(minutes=5)  # 5-minute cache timeout

        # Thread safety
        self._lock = asyncio.Lock()

        self.logger.info(
            "Correlation monitor initialized",
            warning_threshold=self.thresholds.warning_threshold,
            critical_threshold=self.thresholds.critical_threshold,
            lookback_periods=self.thresholds.lookback_periods,
        )

    async def update_price_data(self, market_data: MarketData) -> None:
        """
        Update price data for correlation calculations.

        Args:
            market_data: Latest market data for a symbol
        """
        async with self._lock:
            symbol = market_data.symbol
            # Use close price as the current price
            price = Decimal(str(market_data.close))
            timestamp = market_data.timestamp

            # Store new price
            self.price_history[symbol].append((price, timestamp))

            # Calculate return if we have previous price
            if len(self.price_history[symbol]) >= 2:
                prev_price, _ = self.price_history[symbol][-2]
                if prev_price > 0:
                    return_pct = (price - prev_price) / prev_price
                    self.return_history[symbol].append(return_pct)

            keys_to_remove = [key for key in self._correlation_cache.keys() if symbol in key]
            for key in keys_to_remove:
                del self._correlation_cache[key]

    async def calculate_pairwise_correlation(
        self, symbol1: str, symbol2: str, min_periods: int | None = None
    ) -> Decimal | None:
        """
        Calculate correlation between two symbols with caching.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            min_periods: Minimum periods required, uses threshold default if None

        Returns:
            Decimal correlation or None if insufficient data
        """
        if symbol1 == symbol2:
            return Decimal("1.0")

        min_periods = min_periods or self.thresholds.min_periods
        cache_key: tuple[str, str] = (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)

        # Check cache first
        if cache_key in self._correlation_cache:
            correlation, cache_time = self._correlation_cache[cache_key]
            if datetime.now(timezone.utc) - cache_time < self._cache_timeout:
                return correlation

        async with self._lock:
            returns1 = list(self.return_history[symbol1])
            returns2 = list(self.return_history[symbol2])

            # Check if we have sufficient data
            if len(returns1) < min_periods or len(returns2) < min_periods:
                return None

            min_len = min(len(returns1), len(returns2))
            if min_len < min_periods:
                return None

            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]

            try:
                # Calculate correlation using Decimal precision
                # First, calculate means using Decimal arithmetic
                mean1 = sum(returns1) / Decimal(str(len(returns1)))
                mean2 = sum(returns2) / Decimal(str(len(returns2)))

                # Calculate covariance and standard deviations using Decimal
                covariance = Decimal("0.0")
                var1 = Decimal("0.0")
                var2 = Decimal("0.0")

                # Validate equal lengths before zip
                if len(returns1) != len(returns2):
                    self.logger.warning(
                        "Mismatched return series lengths for correlation calculation",
                        symbol1=symbol1,
                        symbol2=symbol2,
                        len1=len(returns1),
                        len2=len(returns2),
                    )
                    return None

                for r1, r2 in zip(returns1, returns2, strict=True):
                    diff1 = r1 - mean1
                    diff2 = r2 - mean2
                    covariance += diff1 * diff2
                    var1 += diff1 * diff1
                    var2 += diff2 * diff2

                n = Decimal(str(len(returns1)))
                if n > Decimal("1"):
                    covariance /= n - Decimal("1")  # Sample covariance
                    var1 /= n - Decimal("1")  # Sample variance
                    var2 /= n - Decimal("1")  # Sample variance

                # Calculate correlation coefficient
                if var1 > 0 and var2 > 0:
                    std1 = var1.sqrt()
                    std2 = var2.sqrt()
                    correlation = covariance / (std1 * std2)

                    # Ensure correlation is within valid range [-1, 1]
                    correlation = max(Decimal("-1.0"), min(Decimal("1.0"), correlation))
                else:
                    correlation = Decimal("0.0")

                # Cache result
                self._correlation_cache[cache_key] = (correlation, datetime.now(timezone.utc))

                return correlation

            except Exception as e:
                self.logger.warning("Correlation calculation failed", symbol1=symbol1, symbol2=symbol2, error=str(e))
                return None

    async def calculate_portfolio_correlation(self, positions: list[Position]) -> CorrelationMetrics:
        """
        Calculate comprehensive correlation metrics for portfolio.

        Args:
            positions: List of current portfolio positions

        Returns:
            CorrelationMetrics with correlation analysis

        Raises:
            RiskManagementError: If correlation calculation fails
        """
        try:
            symbols = [pos.symbol for pos in positions]

            if len(symbols) < 2:
                # Single position or empty portfolio
                return CorrelationMetrics(
                    average_correlation=Decimal("0.0"),
                    max_pairwise_correlation=Decimal("0.0"),
                    correlation_spike=False,
                    correlated_pairs_count=0,
                    portfolio_concentration_risk=Decimal("0.0"),
                    timestamp=datetime.now(timezone.utc),
                    correlation_matrix={},
                )

            correlation_matrix = {}
            correlations = []
            high_corr_pairs = 0

            # Calculate all pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for _j, symbol2 in enumerate(symbols[i + 1 :], i + 1):
                    correlation = await self.calculate_pairwise_correlation(symbol1, symbol2)

                    if correlation is not None:
                        key: tuple[str, str] = (symbol1, symbol2) if symbol1 < symbol2 else (symbol2, symbol1)
                        correlation_matrix[key] = correlation
                        correlations.append(abs(correlation))  # Use absolute value

                        if abs(correlation) > self.thresholds.warning_threshold:
                            high_corr_pairs += 1

            if not correlations:
                # No valid correlations calculated
                return CorrelationMetrics(
                    average_correlation=Decimal("0.0"),
                    max_pairwise_correlation=Decimal("0.0"),
                    correlation_spike=False,
                    correlated_pairs_count=0,
                    portfolio_concentration_risk=Decimal("0.0"),
                    timestamp=datetime.now(timezone.utc),
                    correlation_matrix=correlation_matrix,
                )

            # Calculate metrics using Decimal precision
            if correlations:
                # Calculate average using Decimal arithmetic
                avg_correlation = sum(Decimal(str(c)) for c in correlations) / Decimal(str(len(correlations)))
                max_correlation = max(Decimal(str(c)) for c in correlations)
            else:
                avg_correlation = Decimal("0.0")
                max_correlation = Decimal("0.0")

            # Detect correlation spike
            correlation_spike = (
                max_correlation > self.thresholds.critical_threshold
                or avg_correlation > self.thresholds.warning_threshold
            )

            total_value = sum(
                abs(pos.current_price * pos.quantity)
                for pos in positions
                if pos.current_price is not None and pos.quantity is not None
            )
            concentration_risk = Decimal("0.0")

            if total_value > 0:
                # Weight correlations by position sizes using Decimal precision
                weighted_correlations = []
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i + 1 :]:
                        weight_key: tuple[str, str] = (
                            (pos1.symbol, pos2.symbol) if pos1.symbol < pos2.symbol else (pos2.symbol, pos1.symbol)
                        )
                        if weight_key in correlation_matrix:
                            # Use Decimal for all weight calculations with null checks
                            if (
                                pos1.current_price is None
                                or pos1.quantity is None
                                or pos2.current_price is None
                                or pos2.quantity is None
                            ):
                                continue
                            value1 = abs(pos1.current_price * pos1.quantity)
                            value2 = abs(pos2.current_price * pos2.quantity)
                            weight1 = value1 / total_value
                            weight2 = value2 / total_value
                            combined_weight = weight1 * weight2
                            weighted_corr = abs(correlation_matrix[weight_key]) * combined_weight
                            weighted_correlations.append(weighted_corr)

                if weighted_correlations:
                    concentration_risk = sum(weighted_correlations) * Decimal("2.0")  # Scale factor

            return CorrelationMetrics(
                average_correlation=avg_correlation,
                max_pairwise_correlation=max_correlation,
                correlation_spike=correlation_spike,
                correlated_pairs_count=high_corr_pairs,
                portfolio_concentration_risk=concentration_risk,
                timestamp=datetime.now(timezone.utc),
                correlation_matrix=correlation_matrix,
            )

        except Exception as e:
            self.logger.error(
                "Portfolio correlation calculation failed",
                error=str(e),
                symbol_count=len(positions),
            )
            raise RiskManagementError(
                f"Portfolio correlation calculation failed: {e}",
                error_code="CORRELATION_CALCULATION_FAILED",
            ) from e

    async def get_position_limits_for_correlation(self, correlation_metrics: CorrelationMetrics) -> dict[str, Any]:
        """
        Calculate position limits based on current correlation levels.

        Args:
            correlation_metrics: Current correlation metrics

        Returns:
            Dict with position limit adjustments
        """
        limits = {
            "max_positions": None,
            "correlation_based_reduction": Decimal("1.0"),  # No reduction
            "warning_level": "normal",
        }

        if correlation_metrics.max_pairwise_correlation > self.thresholds.critical_threshold:
            limits.update(
                {
                    "max_positions": self.thresholds.max_positions_critical_corr,
                    "correlation_based_reduction": Decimal("0.3"),  # 70% reduction
                    "warning_level": "critical",
                }
            )
        elif correlation_metrics.max_pairwise_correlation > self.thresholds.warning_threshold:
            limits.update(
                {
                    "max_positions": self.thresholds.max_positions_high_corr,
                    "correlation_based_reduction": Decimal("0.6"),  # 40% reduction
                    "warning_level": "warning",
                }
            )

        return limits

    async def cleanup_old_data(self, cutoff_time: datetime) -> None:
        """
        Clean up old price and return data with proper resource management.

        Args:
            cutoff_time: Remove data older than this time
        """
        try:
            # Use timeout to prevent hanging during cleanup
            async def _do_cleanup():
                async with self._lock:
                    symbols_cleaned = 0
                    cache_entries_cleaned = 0

                    # Clean up price and return data
                    for symbol in list(self.price_history.keys()):
                        try:
                            # Remove old price data
                            price_deque = self.price_history[symbol]

                            while price_deque and price_deque[0][1] < cutoff_time:
                                price_deque.popleft()

                            # If deque is empty or very small, remove the symbol
                            if not price_deque or len(price_deque) < 5:
                                del self.price_history[symbol]
                                # Check if symbol exists in return_history before deleting
                                if symbol in self.return_history:
                                    del self.return_history[symbol]
                                symbols_cleaned += 1
                        except Exception as e:
                            self.logger.error(f"Error cleaning data for symbol {symbol}: {e}")
                            # Continue with other symbols

                    # Clear correlation cache with size tracking
                    cache_entries_cleaned = len(self._correlation_cache)
                    self._correlation_cache.clear()

                    # Log cleanup statistics for monitoring
                    self.logger.info(
                        "Data cleanup completed",
                        symbols_cleaned=symbols_cleaned,
                        cache_entries_cleaned=cache_entries_cleaned,
                        remaining_symbols=len(self.price_history),
                        cutoff_time=cutoff_time.isoformat(),
                    )

            # Execute cleanup with timeout
            await asyncio.wait_for(_do_cleanup(), timeout=10.0)

        except asyncio.TimeoutError:
            self.logger.error("Data cleanup timed out after 10 seconds")
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
            # Don't re-raise to prevent blocking other operations

    def _periodic_cache_cleanup(self) -> None:
        """
        Periodic cleanup of correlation cache to prevent memory leaks.
        Called when cache grows too large.
        """
        try:
            current_time = datetime.now(timezone.utc)
            entries_removed = 0

            # Create new cache with only recent entries
            new_cache = {}
            for key, (correlation, cache_time) in self._correlation_cache.items():
                if current_time - cache_time < self._cache_timeout * 2:  # Keep entries for 2x timeout
                    new_cache[key] = (correlation, cache_time)
                else:
                    entries_removed += 1

            self._correlation_cache = new_cache

            if entries_removed > 0:
                self.logger.info(f"Cleaned up {entries_removed} old correlation cache entries")

        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    async def cleanup_resources(self) -> None:
        """
        Clean up all resources used by the correlation monitor.
        Should be called when shutting down or resetting the monitor.
        """
        try:
            async with self._lock:
                # Clear all data structures
                symbols_count = len(self.price_history)
                cache_size = len(self._correlation_cache)

                self.price_history.clear()
                self.return_history.clear()
                self._correlation_cache.clear()

                self.logger.info(
                    "Correlation monitor resources cleaned up",
                    symbols_cleared=symbols_count,
                    cache_entries_cleared=cache_size,
                )

        except Exception as e:
            self.logger.error(f"Error cleaning up correlation monitor resources: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current monitor status with resource usage information."""
        try:
            # Calculate memory usage estimates
            total_price_points = sum(len(history) for history in self.price_history.values())
            total_return_points = sum(len(history) for history in self.return_history.values())

            return {
                "monitored_symbols": len(self.price_history),
                "cache_size": len(self._correlation_cache),
                "total_price_points": total_price_points,
                "total_return_points": total_return_points,
                "cache_timeout_minutes": self._cache_timeout.total_seconds() / 60,
                "thresholds": {
                    "warning": str(self.thresholds.warning_threshold),
                    "critical": str(self.thresholds.critical_threshold),
                    "lookback_periods": self.thresholds.lookback_periods,
                    "min_periods": self.thresholds.min_periods,
                },
                "data_points_per_symbol": {symbol: len(history) for symbol, history in self.return_history.items()},
                "resource_usage": {
                    "estimated_memory_mb": (total_price_points + total_return_points)
                    * 8
                    / 1024
                    / 1024,  # Rough estimate
                    "symbols_with_insufficient_data": len(
                        [s for s, h in self.return_history.items() if len(h) < self.thresholds.min_periods]
                    ),
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e), "monitored_symbols": 0, "cache_size": 0}
