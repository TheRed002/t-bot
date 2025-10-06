"""
Technical Indicator Validation Utilities.

This module provides mathematical validation functions for technical indicators
to ensure accuracy of real strategy calculations using Decimal precision.

CRITICAL: All calculations must use Decimal for financial accuracy.
"""

import math
from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 28


class IndicatorValidator:
    """Validator for technical indicator mathematical accuracy."""

    @staticmethod
    def calculate_sma(prices: list[Decimal], period: int) -> Decimal | None:
        """
        Calculate Simple Moving Average with Decimal precision.

        Args:
            prices: List of price values as Decimal
            period: Period for SMA calculation

        Returns:
            SMA value as Decimal or None if insufficient data
        """
        if len(prices) < period:
            return None

        return sum(prices[-period:]) / Decimal(str(period))

    @staticmethod
    def calculate_ema(prices: list[Decimal], period: int) -> Decimal | None:
        """
        Calculate Exponential Moving Average with Decimal precision.

        Args:
            prices: List of price values as Decimal
            period: Period for EMA calculation

        Returns:
            EMA value as Decimal or None if insufficient data
        """
        if len(prices) < period:
            return None

        # Calculate smoothing factor
        multiplier = Decimal("2") / (Decimal(str(period)) + Decimal("1"))

        # Start with SMA as initial EMA
        initial_sma = sum(prices[:period]) / Decimal(str(period))
        ema = initial_sma

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (Decimal("1") - multiplier))

        return ema

    @staticmethod
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None:
        """
        Calculate Relative Strength Index with Decimal precision using Wilder's smoothing.

        This implementation matches TA-Lib's RSI calculation which uses Wilder's
        smoothing method (exponential moving average) rather than simple moving average.

        Args:
            prices: List of price values as Decimal (chronological order)
            period: Period for RSI calculation (default 14)

        Returns:
            RSI value as Decimal (0-100) or None if insufficient data
        """
        if len(prices) < period + 1:
            return None

        # Calculate price changes
        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        if len(gains) < period:
            return None

        # Wilder's smoothing method (same as TA-Lib)
        # 1. Calculate initial average using SMA for first 'period' values
        avg_gain = sum(gains[:period]) / Decimal(str(period))
        avg_loss = sum(losses[:period]) / Decimal(str(period))

        # 2. Apply Wilder's smoothing for remaining values
        # Formula: New Average = ((Previous Average * (period - 1)) + Current Value) / period
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * Decimal(str(period - 1))) + gains[i]) / Decimal(str(period))
            avg_loss = ((avg_loss * Decimal(str(period - 1))) + losses[i]) / Decimal(str(period))

        # Handle division by zero
        if avg_loss == 0:
            return Decimal("100")

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        prices: list[Decimal], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> dict[str, Decimal] | None:
        """
        Calculate MACD (Moving Average Convergence Divergence) with Decimal precision.

        Args:
            prices: List of price values as Decimal
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)

        Returns:
            Dictionary with MACD, signal, and histogram values or None if insufficient data
        """
        if len(prices) < slow_period:
            return None

        # Calculate fast and slow EMAs
        fast_ema = IndicatorValidator.calculate_ema(prices, fast_period)
        slow_ema = IndicatorValidator.calculate_ema(prices, slow_period)

        if fast_ema is None or slow_ema is None:
            return None

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # For signal line, we need MACD history (simplified for testing)
        # In real implementation, this would use historical MACD values
        signal_line = macd_line  # Simplified for testing

        # Calculate histogram
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_bollinger_bands(
        prices: list[Decimal], period: int = 20, std_dev_multiplier: float = 2.0
    ) -> dict[str, Decimal] | None:
        """
        Calculate Bollinger Bands with Decimal precision.

        Args:
            prices: List of price values as Decimal
            period: Period for moving average (default 20)
            std_dev_multiplier: Standard deviation multiplier (default 2.0)

        Returns:
            Dictionary with upper, middle, and lower band values or None if insufficient data
        """
        if len(prices) < period:
            return None

        # Calculate middle band (SMA)
        middle_band = IndicatorValidator.calculate_sma(prices, period)
        if middle_band is None:
            return None

        # Calculate standard deviation
        recent_prices = prices[-period:]
        variance = sum([(price - middle_band) ** 2 for price in recent_prices]) / Decimal(
            str(period)
        )
        std_dev = Decimal(str(math.sqrt(float(variance))))

        # Calculate bands
        multiplier = Decimal(str(std_dev_multiplier))
        upper_band = middle_band + (std_dev * multiplier)
        lower_band = middle_band - (std_dev * multiplier)

        return {"upper": upper_band, "middle": middle_band, "lower": lower_band}

    @staticmethod
    def calculate_atr(
        high_prices: list[Decimal],
        low_prices: list[Decimal],
        close_prices: list[Decimal],
        period: int = 14,
    ) -> Decimal | None:
        """
        Calculate Average True Range with Decimal precision.

        Args:
            high_prices: List of high prices as Decimal
            low_prices: List of low prices as Decimal
            close_prices: List of close prices as Decimal
            period: Period for ATR calculation (default 14)

        Returns:
            ATR value as Decimal or None if insufficient data
        """
        if (
            len(high_prices) < period + 1
            or len(low_prices) < period + 1
            or len(close_prices) < period + 1
        ):
            return None

        true_ranges = []

        for i in range(1, len(close_prices)):
            # Calculate true range components
            hl = high_prices[i] - low_prices[i]
            hc = abs(high_prices[i] - close_prices[i - 1])
            lc = abs(low_prices[i] - close_prices[i - 1])

            # True range is the maximum of the three
            true_range = max(hl, hc, lc)
            true_ranges.append(true_range)

        if len(true_ranges) < period:
            return None

        # Calculate ATR as simple average of true ranges
        atr = sum(true_ranges[-period:]) / Decimal(str(period))
        return atr

    @staticmethod
    def calculate_stochastic(
        high_prices: list[Decimal],
        low_prices: list[Decimal],
        close_prices: list[Decimal],
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict[str, Decimal] | None:
        """
        Calculate Stochastic Oscillator with Decimal precision.

        Args:
            high_prices: List of high prices as Decimal
            low_prices: List of low prices as Decimal
            close_prices: List of close prices as Decimal
            k_period: Period for %K calculation (default 14)
            d_period: Period for %D calculation (default 3)

        Returns:
            Dictionary with %K and %D values or None if insufficient data
        """
        if (
            len(high_prices) < k_period
            or len(low_prices) < k_period
            or len(close_prices) < k_period
        ):
            return None

        # Calculate %K
        recent_highs = high_prices[-k_period:]
        recent_lows = low_prices[-k_period:]
        current_close = close_prices[-1]

        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)

        if highest_high == lowest_low:
            k_percent = Decimal("50")  # Neutral value when no range
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * Decimal(
                "100"
            )

        # For %D, we would need historical %K values (simplified for testing)
        d_percent = k_percent  # Simplified for testing

        return {"k_percent": k_percent, "d_percent": d_percent}

    @staticmethod
    def validate_indicator_range(indicator_name: str, value: Decimal) -> bool:
        """
        Validate that indicator value is within expected range.

        Args:
            indicator_name: Name of the indicator
            value: Calculated indicator value

        Returns:
            True if value is within valid range, False otherwise
        """
        if indicator_name.lower() == "rsi":
            return Decimal("0") <= value <= Decimal("100")
        elif indicator_name.lower() in ["stochastic_k", "stochastic_d"]:
            return Decimal("0") <= value <= Decimal("100")
        elif indicator_name.lower() in [
            "sma",
            "ema",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        ]:
            return value > Decimal("0")  # Prices should be positive
        elif indicator_name.lower() == "atr":
            return value >= Decimal("0")  # ATR should be non-negative
        else:
            return True  # Unknown indicator, assume valid

    @staticmethod
    def compare_indicators(
        calculated_value: Decimal, expected_value: Decimal, tolerance: Decimal = Decimal("0.01")
    ) -> tuple[bool, Decimal]:
        """
        Compare calculated indicator value with expected value.

        Args:
            calculated_value: Value calculated by strategy
            expected_value: Expected value from validation calculation
            tolerance: Acceptable difference tolerance

        Returns:
            Tuple of (is_within_tolerance, actual_difference)
        """
        difference = abs(calculated_value - expected_value)
        is_within_tolerance = difference <= tolerance

        return is_within_tolerance, difference

    @staticmethod
    def validate_decimal_precision(value: any) -> bool:
        """
        Validate that a value is using Decimal precision.

        Args:
            value: Value to check

        Returns:
            True if value is Decimal type, False otherwise
        """
        return isinstance(value, Decimal)

    @staticmethod
    def convert_to_decimal_if_needed(value: any) -> Decimal:
        """
        Convert value to Decimal if it's not already.

        Args:
            value: Value to convert

        Returns:
            Value as Decimal
        """
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        else:
            raise ValueError(f"Cannot convert {type(value)} to Decimal")


class IndicatorAccuracyTester:
    """Test harness for indicator accuracy validation."""

    def __init__(self):
        self.validator = IndicatorValidator()
        self.test_results = []

    def test_sma_accuracy(
        self, strategy_sma_func, test_prices: list[Decimal], period: int, symbol: str = "BTC/USDT"
    ) -> dict[str, any]:
        """
        Test SMA calculation accuracy.

        Args:
            strategy_sma_func: Strategy's SMA calculation function
            test_prices: Test price data
            period: SMA period
            symbol: Trading symbol

        Returns:
            Test result dictionary
        """
        # Calculate expected SMA
        expected_sma = self.validator.calculate_sma(test_prices, period)

        if expected_sma is None:
            return {"test": "SMA Accuracy", "status": "SKIPPED", "reason": "Insufficient data"}

        # Calculate strategy SMA (this would be async in real implementation)
        try:
            # Note: In real implementation, this would be await strategy_sma_func(symbol, period)
            calculated_sma = None  # Placeholder - actual implementation would call strategy

            if calculated_sma is None:
                return {
                    "test": "SMA Accuracy",
                    "status": "FAILED",
                    "reason": "Strategy returned None",
                }

            # Validate Decimal precision
            if not self.validator.validate_decimal_precision(calculated_sma):
                return {
                    "test": "SMA Accuracy",
                    "status": "FAILED",
                    "reason": f"Strategy returned {type(calculated_sma)}, expected Decimal",
                }

            # Compare values
            is_accurate, difference = self.validator.compare_indicators(
                calculated_sma, expected_sma, Decimal("0.01")
            )

            result = {
                "test": "SMA Accuracy",
                "status": "PASSED" if is_accurate else "FAILED",
                "calculated": calculated_sma,
                "expected": expected_sma,
                "difference": difference,
                "tolerance": Decimal("0.01"),
            }

            self.test_results.append(result)
            return result

        except Exception as e:
            return {"test": "SMA Accuracy", "status": "ERROR", "error": str(e)}

    def generate_comprehensive_test_report(self) -> dict[str, any]:
        """
        Generate comprehensive test report for all indicator tests.

        Returns:
            Complete test report with summary and details
        """
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        skipped_tests = len([r for r in self.test_results if r["status"] == "SKIPPED"])

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            },
            "details": self.test_results,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_tests = [r for r in self.test_results if r["status"] == "FAILED"]
        if failed_tests:
            recommendations.append("Review failed indicator calculations for mathematical accuracy")

        error_tests = [r for r in self.test_results if r["status"] == "ERROR"]
        if error_tests:
            recommendations.append("Fix error conditions in indicator implementation")

        precision_issues = [r for r in self.test_results if "Decimal" in r.get("reason", "")]
        if precision_issues:
            recommendations.append("Ensure all financial calculations use Decimal type")

        return recommendations
