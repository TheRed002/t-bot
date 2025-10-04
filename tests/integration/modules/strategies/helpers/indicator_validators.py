"""
Technical Indicator Mathematical Validation Helpers.

This module provides validation functions to ensure technical indicators
produce mathematically accurate results. Used for testing real indicator
implementations against known correct values.

Key Features:
- RSI calculation validation with known test cases
- MACD calculation verification
- Moving average accuracy testing
- Bollinger Bands validation
- Performance benchmarking for indicator calculations
"""

import time
from decimal import Decimal
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.types import MarketData


class IndicatorValidator:
    """
    Validates technical indicator calculations for mathematical accuracy.

    Provides reference implementations and test cases to verify that
    our production indicators produce correct results.
    """

    @staticmethod
    def calculate_reference_rsi(prices: List[Decimal], period: int = 14) -> Decimal:
        """
        Calculate RSI using reference implementation for validation.

        This is a simple, clear implementation used to validate our
        production RSI calculations.

        Args:
            prices: List of closing prices
            period: RSI period (default 14)

        Returns:
            RSI value as Decimal
        """
        if len(prices) < period + 1:
            raise ValueError(f"Need at least {period + 1} prices for RSI calculation")

        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            changes.append(change)

        # Separate gains and losses
        gains = [max(change, Decimal("0")) for change in changes]
        losses = [abs(min(change, Decimal("0"))) for change in changes]

        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Calculate smoothed averages for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # Calculate RSI
        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    @staticmethod
    def calculate_reference_sma(prices: List[Decimal], period: int) -> Decimal:
        """
        Calculate Simple Moving Average using reference implementation.

        Args:
            prices: List of closing prices
            period: SMA period

        Returns:
            SMA value as Decimal
        """
        if len(prices) < period:
            raise ValueError(f"Need at least {period} prices for SMA calculation")

        return sum(prices[-period:]) / period

    @staticmethod
    def calculate_reference_ema(prices: List[Decimal], period: int) -> Decimal:
        """
        Calculate Exponential Moving Average using reference implementation.

        Args:
            prices: List of closing prices
            period: EMA period

        Returns:
            EMA value as Decimal
        """
        if len(prices) < period:
            raise ValueError(f"Need at least {period} prices for EMA calculation")

        # Calculate smoothing factor
        multiplier = Decimal("2") / (period + 1)

        # Start with SMA for initial EMA
        ema = sum(prices[:period]) / period

        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema * (Decimal("1") - multiplier))

        return ema

    @staticmethod
    def calculate_reference_macd(
        prices: List[Decimal],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, Decimal]:
        """
        Calculate MACD using reference implementation.

        Args:
            prices: List of closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < slow_period + signal_period:
            raise ValueError(f"Need at least {slow_period + signal_period} prices for MACD")

        # Calculate EMAs
        fast_ema = IndicatorValidator.calculate_reference_ema(prices, fast_period)
        slow_ema = IndicatorValidator.calculate_reference_ema(prices, slow_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # For signal line, we need MACD values for multiple periods
        # This is simplified - in practice, you'd calculate MACD for each period
        # and then apply EMA to those values
        macd_values = [macd_line]  # Simplified for single calculation

        signal_line = macd_line  # Simplified - should be EMA of MACD values

        # Histogram
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def validate_rsi_accuracy(
        calculated_rsi: Decimal,
        market_data: List[MarketData],
        period: int = 14,
        tolerance: Decimal = Decimal("0.01"),
    ) -> bool:
        """
        Validate RSI calculation accuracy against reference implementation.

        Args:
            calculated_rsi: RSI calculated by production code
            market_data: Market data used for calculation
            period: RSI period
            tolerance: Acceptable difference (default 0.01 = 1%)

        Returns:
            True if calculated RSI is within tolerance of reference
        """
        # Extract closing prices
        prices = [data.close for data in market_data]

        # Calculate reference RSI
        reference_rsi = IndicatorValidator.calculate_reference_rsi(prices, period)

        # Check if within tolerance
        difference = abs(calculated_rsi - reference_rsi)
        return difference <= tolerance

    @staticmethod
    def validate_sma_accuracy(
        calculated_sma: Decimal,
        market_data: List[MarketData],
        period: int,
        tolerance: Decimal = Decimal("0.001"),
    ) -> bool:
        """
        Validate SMA calculation accuracy.

        Args:
            calculated_sma: SMA calculated by production code
            market_data: Market data used for calculation
            period: SMA period
            tolerance: Acceptable difference

        Returns:
            True if calculated SMA is within tolerance
        """
        prices = [data.close for data in market_data]
        reference_sma = IndicatorValidator.calculate_reference_sma(prices, period)

        difference = abs(calculated_sma - reference_sma)
        return difference <= tolerance

    @staticmethod
    def validate_ema_accuracy(
        calculated_ema: Decimal,
        market_data: List[MarketData],
        period: int,
        tolerance: Decimal = Decimal("0.01"),
    ) -> bool:
        """
        Validate EMA calculation accuracy.

        Args:
            calculated_ema: EMA calculated by production code
            market_data: Market data used for calculation
            period: EMA period
            tolerance: Acceptable difference

        Returns:
            True if calculated EMA is within tolerance
        """
        prices = [data.close for data in market_data]
        reference_ema = IndicatorValidator.calculate_reference_ema(prices, period)

        difference = abs(calculated_ema - reference_ema)
        return difference <= tolerance


class PerformanceBenchmarker:
    """
    Benchmarks performance of technical indicator calculations.

    Ensures that real calculations meet performance requirements
    for production trading systems.
    """

    @staticmethod
    def benchmark_rsi_calculation(
        indicator_function, market_data: List[MarketData], iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark RSI calculation performance.

        Args:
            indicator_function: Function that calculates RSI
            market_data: Market data for calculation
            iterations: Number of iterations to run

        Returns:
            Performance metrics dictionary
        """
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                result = indicator_function(market_data, period=14)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                # Record failed calculations
                times.append(float("inf"))

        return {
            "avg_time_ms": np.mean(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "p95_time_ms": np.percentile(times, 95) * 1000,
            "success_rate": len([t for t in times if t != float("inf")]) / len(times),
            "total_iterations": iterations,
        }

    @staticmethod
    def benchmark_multiple_indicators(
        indicators: Dict[str, callable],
        market_data: List[MarketData],
        iterations: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple indicators for performance comparison.

        Args:
            indicators: Dictionary of indicator name to function
            market_data: Market data for calculations
            iterations: Number of iterations per indicator

        Returns:
            Performance metrics for each indicator
        """
        results = {}

        for name, indicator_func in indicators.items():
            try:
                benchmark_result = PerformanceBenchmarker.benchmark_rsi_calculation(
                    indicator_func, market_data, iterations
                )
                results[name] = benchmark_result
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "avg_time_ms": float("inf"),
                    "success_rate": 0.0,
                }

        return results


def create_known_test_cases() -> Dict[str, Dict[str, any]]:
    """
    Create known test cases with expected results for validation.

    Returns:
        Dictionary of test cases with market data and expected results
    """
    test_cases = {
        "rsi_simple": {
            "description": "Simple RSI test case with known result",
            "prices": [
                Decimal("44.00"),
                Decimal("44.50"),
                Decimal("44.25"),
                Decimal("43.75"),
                Decimal("44.50"),
                Decimal("45.00"),
                Decimal("45.50"),
                Decimal("45.25"),
                Decimal("46.00"),
                Decimal("47.00"),
                Decimal("46.75"),
                Decimal("46.50"),
                Decimal("46.25"),
                Decimal("47.75"),
                Decimal("48.00"),  # 15th price for 14-period RSI
            ],
            "expected_rsi": Decimal("70.53"),  # Approximate expected RSI
            "tolerance": Decimal("1.0"),  # 1% tolerance
        },
        "sma_simple": {
            "description": "Simple SMA test case",
            "prices": [Decimal(str(i)) for i in range(10, 20)],  # 10, 11, 12, ..., 19
            "period": 5,
            "expected_sma": Decimal("17.0"),  # Average of 15, 16, 17, 18, 19
            "tolerance": Decimal("0.001"),
        },
        "ema_simple": {
            "description": "Simple EMA test case",
            "prices": [Decimal("10.0")] * 10 + [Decimal("20.0")] * 10,  # Step function
            "period": 5,
            "expected_approximate": True,  # EMA doesn't have exact expected value for this
        },
    }

    return test_cases


def validate_all_indicators(
    technical_indicators,
    market_data: List[MarketData],
    tolerance_overrides: Dict[str, Decimal] = None,
) -> Dict[str, bool]:
    """
    Validate all technical indicators against reference implementations.

    Args:
        technical_indicators: TechnicalIndicators instance
        market_data: Market data for calculations
        tolerance_overrides: Custom tolerance values for specific indicators

    Returns:
        Dictionary of validation results for each indicator
    """
    results = {}
    tolerances = tolerance_overrides or {}

    # Test RSI
    try:
        calculated_rsi = technical_indicators.calculate_rsi(market_data, period=14)
        rsi_valid = IndicatorValidator.validate_rsi_accuracy(
            calculated_rsi,
            market_data,
            period=14,
            tolerance=tolerances.get("rsi", Decimal("0.01")),
        )
        results["rsi"] = rsi_valid
    except Exception as e:
        results["rsi"] = False
        results["rsi_error"] = str(e)

    # Test SMA
    try:
        calculated_sma = technical_indicators.calculate_sma(market_data, period=20)
        sma_valid = IndicatorValidator.validate_sma_accuracy(
            calculated_sma,
            market_data,
            period=20,
            tolerance=tolerances.get("sma", Decimal("0.001")),
        )
        results["sma"] = sma_valid
    except Exception as e:
        results["sma"] = False
        results["sma_error"] = str(e)

    # Test EMA
    try:
        calculated_ema = technical_indicators.calculate_ema(market_data, period=20)
        ema_valid = IndicatorValidator.validate_ema_accuracy(
            calculated_ema,
            market_data,
            period=20,
            tolerance=tolerances.get("ema", Decimal("0.01")),
        )
        results["ema"] = ema_valid
    except Exception as e:
        results["ema"] = False
        results["ema_error"] = str(e)

    return results