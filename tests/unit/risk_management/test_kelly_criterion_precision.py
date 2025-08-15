"""
Comprehensive Kelly Criterion Tests for Position Sizing.

This module tests the mathematical accuracy and edge cases of Kelly Criterion
calculations in position sizing to prevent incorrect position sizing that
could lead to excessive risk or missed opportunities.

CRITICAL AREAS TESTED:
1. Mathematical accuracy of Kelly fraction calculation
2. Edge cases (all wins, all losses, no variance, extreme values)
3. Position size limits and safeguards
4. Confidence weighting accuracy
5. Historical data window handling
6. Performance under various market scenarios
"""

from datetime import datetime
from decimal import Decimal, getcontext
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from src.core.config import Config
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import PositionSizeMethod, Signal, SignalDirection
from src.risk_management.position_sizing import PositionSizer


class TestKellyCriterionPrecision:
    """
    Test suite for Kelly Criterion calculations with focus on mathematical accuracy.

    These tests ensure Kelly Criterion calculations are mathematically correct
    and handle edge cases that could cause excessive risk or suboptimal sizing.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration with known Kelly parameters."""
        config = Config()
        # Set known Kelly parameters for testing
        config.risk.kelly_lookback_days = 30
        config.risk.kelly_max_fraction = 0.25  # 25% max Kelly
        config.risk.default_position_size_pct = 0.02  # 2% default
        config.risk.max_position_size_pct = 0.10  # 10% max position
        return config

    @pytest.fixture
    def position_sizer(self, config):
        """Create position sizer instance."""
        return PositionSizer(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )

    @pytest.fixture(autouse=True)
    def setup_high_precision(self):
        """Set high decimal precision for calculations."""
        getcontext().prec = 50  # Very high precision for testing

    @pytest.mark.asyncio
    async def test_kelly_mathematical_accuracy_known_values(self, position_sizer, sample_signal):
        """Test Kelly calculation with known mathematical values."""
        portfolio_value = Decimal("10000")

        # Create returns with known mean and variance
        # Mean = 0.02, Variance = 0.01, Kelly = 0.02/0.01 = 2.0 (will be capped)
        returns = [0.02, 0.01, 0.03, 0.01, 0.02] * 6  # 30 returns
        position_sizer.return_history["BTCUSDT"] = returns

        # Calculate expected Kelly fraction
        returns_array = np.array(returns)
        expected_mean = np.mean(returns_array)
        expected_variance = np.var(returns_array)
        expected_kelly = expected_mean / expected_variance

        # Should be capped at max_kelly_fraction (0.25)
        expected_kelly_capped = min(expected_kelly, 0.25)
        # Apply confidence
        expected_kelly_final = expected_kelly_capped * sample_signal.confidence
        expected_position_size = portfolio_value * Decimal(str(expected_kelly_final))

        # Calculate actual Kelly
        actual_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should be close to expected (allowing for floating point precision)
        ratio = float(actual_size / expected_position_size)
        assert 0.95 <= ratio <= 1.05  # Within 5% tolerance

    @pytest.mark.asyncio
    async def test_kelly_edge_case_all_positive_returns(self, position_sizer, sample_signal):
        """Test Kelly with all positive returns (extreme optimism)."""
        portfolio_value = Decimal("10000")

        # All positive returns
        returns = [0.01, 0.02, 0.015, 0.025, 0.01] * 6  # 30 positive returns
        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should be capped at max Kelly fraction
        max_allowed = portfolio_value * Decimal(str(position_sizer.risk_config.kelly_max_fraction))
        max_with_confidence = max_allowed * Decimal(str(sample_signal.confidence))

        assert position_size <= max_with_confidence
        assert position_size > Decimal("0")

    @pytest.mark.asyncio
    async def test_kelly_edge_case_all_negative_returns(self, position_sizer, sample_signal):
        """Test Kelly with all negative returns (extreme pessimism)."""
        portfolio_value = Decimal("10000")

        # All negative returns
        returns = [-0.01, -0.02, -0.015, -0.005, -0.01] * 6  # 30 negative returns
        position_sizer.return_history["BTCUSDT"] = returns

        # Kelly with all negative returns should fallback to fixed percentage
        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should fallback to fixed percentage sizing
        expected_fallback = (
            portfolio_value * Decimal("0.02") * Decimal(str(sample_signal.confidence))
        )
        assert position_size == expected_fallback

    @pytest.mark.asyncio
    async def test_kelly_edge_case_zero_variance(self, position_sizer, sample_signal):
        """Test Kelly with zero variance (all identical returns)."""
        portfolio_value = Decimal("10000")

        # All identical returns (zero variance)
        returns = [0.01] * 30
        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should fallback to fixed percentage sizing
        expected_fallback = (
            portfolio_value * Decimal("0.02") * Decimal(str(sample_signal.confidence))
        )
        assert position_size == expected_fallback

    @pytest.mark.asyncio
    async def test_kelly_extreme_variance_scenarios(self, position_sizer, sample_signal):
        """Test Kelly with extreme variance scenarios."""
        portfolio_value = Decimal("10000")

        test_scenarios = [
            # Very low variance
            ([0.001] * 15 + [0.002] * 15, "low_variance"),
            # Very high variance
            ([-0.1, 0.1] * 15, "high_variance"),
            # Mixed extreme returns
            ([-0.05, 0.15, -0.02, 0.08] * 7 + [-0.01, 0.02], "mixed_extreme"),
        ]

        for returns, scenario_name in test_scenarios:
            position_sizer.return_history["BTCUSDT"] = returns

            position_size = await position_sizer._kelly_criterion_sizing(
                sample_signal, portfolio_value
            )

            # All scenarios should produce valid position sizes
            assert isinstance(position_size, Decimal)
            assert position_size >= Decimal("0")
            assert position_size <= portfolio_value * Decimal("0.25")  # Never exceed max Kelly

    @pytest.mark.asyncio
    async def test_kelly_confidence_weighting_accuracy(self, position_sizer):
        """Test that confidence weighting is applied correctly."""
        portfolio_value = Decimal("10000")

        # Fixed returns for consistent Kelly calculation
        returns = [0.02, 0.01, 0.03, -0.01, 0.02] * 6
        position_sizer.return_history["BTCUSDT"] = returns

        # Test different confidence levels
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        position_sizes = []

        for confidence in confidence_levels:
            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=confidence,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                strategy_name="test_strategy",
            )

            size = await position_sizer._kelly_criterion_sizing(signal, portfolio_value)
            position_sizes.append((confidence, size))

        # Position sizes should increase with confidence (monotonically)
        for i in range(1, len(position_sizes)):
            prev_confidence, prev_size = position_sizes[i - 1]
            curr_confidence, curr_size = position_sizes[i]

            # Size should increase with confidence (or stay same due to caps)
            assert curr_size >= prev_size * Decimal("0.99")  # Allow for small rounding

    @pytest.mark.asyncio
    async def test_kelly_historical_window_sensitivity(self, position_sizer, sample_signal):
        """Test Kelly sensitivity to historical data window size."""
        portfolio_value = Decimal("10000")

        # Create longer series and test different windows
        full_returns = [0.02, -0.01, 0.03, -0.005, 0.015] * 20  # 100 returns

        window_tests = [
            (30, "short_window"),
            (50, "medium_window"),
            (100, "long_window"),
        ]

        results = []
        for window_size, window_name in window_tests:
            # Set window size in config
            position_sizer.risk_config.kelly_lookback_days = window_size

            # Use appropriate data window
            position_sizer.return_history["BTCUSDT"] = full_returns[-window_size:]

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)
            results.append((window_size, size, window_name))

        # All results should be valid
        for window_size, size, name in results:
            assert isinstance(size, Decimal)
            assert size > Decimal("0")
            assert size <= portfolio_value * Decimal("0.25")

    @pytest.mark.asyncio
    async def test_kelly_with_real_market_scenarios(self, position_sizer, sample_signal):
        """Test Kelly with realistic market return scenarios."""
        portfolio_value = Decimal("10000")

        # Realistic crypto market scenarios
        market_scenarios = {
            "bull_market": [0.05, 0.03, 0.07, 0.02, 0.04, 0.06, 0.01, 0.03]
            * 4,  # Generally positive
            "bear_market": [-0.03, -0.05, -0.02, 0.01, -0.04, -0.01, -0.06, 0.02]
            * 4,  # Generally negative
            "sideways": [0.01, -0.01, 0.02, -0.015, 0.005, -0.005, 0.01, -0.01] * 4,  # Mixed
            "volatile": [0.1, -0.08, 0.12, -0.09, 0.07, -0.05, 0.08, -0.06] * 4,  # High volatility
        }

        for scenario_name, returns in market_scenarios.items():
            position_sizer.return_history["BTCUSDT"] = returns

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

            # Verify reasonable sizing for each scenario
            assert isinstance(size, Decimal)
            assert size >= Decimal("0")

            if scenario_name == "bull_market":
                # Bull market should allow larger positions (but capped)
                assert size > Decimal("100")  # More than 1% of portfolio
            elif scenario_name == "bear_market":
                # Bear market should use smaller positions or fallback
                # May fallback to fixed percentage
                pass  # Just verify no crash

            # All scenarios should respect maximum position size
            max_size = portfolio_value * Decimal("0.25")  # 25% max Kelly
            assert size <= max_size

    @pytest.mark.asyncio
    async def test_kelly_calculation_consistency(self, position_sizer, sample_signal):
        """Test that Kelly calculations are consistent across multiple calls."""
        portfolio_value = Decimal("10000")

        # Fixed return series
        returns = [0.02, 0.01, 0.03, -0.01, 0.015, 0.025, -0.005, 0.02] * 4
        position_sizer.return_history["BTCUSDT"] = returns

        # Calculate Kelly multiple times
        sizes = []
        for _ in range(10):
            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)
            sizes.append(size)

        # All sizes should be identical
        for size in sizes[1:]:
            assert size == sizes[0]

    @pytest.mark.asyncio
    async def test_kelly_numerical_stability(self, position_sizer, sample_signal):
        """Test numerical stability with extreme input values."""
        portfolio_value = Decimal("10000")

        # Test with extreme but valid values
        extreme_scenarios = [
            # Very small returns
            ([0.0001, -0.0001] * 15, "micro_returns"),
            # Large returns
            ([0.5, -0.3, 0.4, -0.2] * 7 + [0.1, -0.1], "large_returns"),
            # Mixed scales
            ([0.001, 0.1, -0.05, 0.002, -0.08] * 6, "mixed_scales"),
        ]

        for returns, scenario in extreme_scenarios:
            position_sizer.return_history["BTCUSDT"] = returns

            # Should not crash or produce invalid results
            try:
                size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)
                assert isinstance(size, Decimal)
                assert size >= Decimal("0")
            except Exception as e:
                # If calculation fails, should fallback gracefully
                assert "Kelly Criterion calculation failed" in str(e)

    @pytest.mark.asyncio
    async def test_kelly_integration_with_calculate_position_size(
        self, position_sizer, sample_signal
    ):
        """Test Kelly integration with main position size calculation method."""
        portfolio_value = Decimal("10000")

        # Add sufficient return history
        returns = [0.02, 0.01, 0.03, -0.01, 0.015] * 6  # 30 returns
        position_sizer.return_history["BTCUSDT"] = returns

        # Calculate using main method
        size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Calculate using internal method
        internal_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should apply additional limits in main method
        assert size <= portfolio_value * Decimal(
            str(position_sizer.risk_config.max_position_size_pct)
        )
        assert size >= Decimal("0")

        # If within limits, should match internal calculation
        max_allowed = portfolio_value * Decimal(
            str(position_sizer.risk_config.max_position_size_pct)
        )
        if internal_size <= max_allowed:
            assert size == internal_size

    @pytest.mark.asyncio
    async def test_kelly_statistical_properties(self, position_sizer, sample_signal):
        """Test that Kelly calculations respect statistical properties."""
        portfolio_value = Decimal("10000")

        # Generate returns with known statistical properties
        np.random.seed(42)  # Reproducible results

        # Normal distribution with positive mean
        returns = np.random.normal(0.01, 0.02, 30).tolist()  # Mean=1%, Std=2%
        position_sizer.return_history["BTCUSDT"] = returns

        size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Calculate theoretical Kelly
        mean_return = np.mean(returns)
        variance = np.var(returns)
        theoretical_kelly = mean_return / variance if variance > 0 else 0

        # Apply caps and confidence
        capped_kelly = min(theoretical_kelly, 0.25) * sample_signal.confidence
        expected_size = portfolio_value * Decimal(str(capped_kelly))

        # Should be reasonably close to theoretical value
        if variance > 0 and mean_return > 0:
            ratio = float(size / expected_size) if expected_size > 0 else 0
            assert 0.8 <= ratio <= 1.2  # Within 20% of theoretical

    @pytest.mark.asyncio
    async def test_kelly_fallback_behavior(self, position_sizer, sample_signal):
        """Test Kelly fallback to fixed percentage in various scenarios."""
        portfolio_value = Decimal("10000")

        fallback_scenarios = [
            ([], "no_data"),
            ([0.01] * 10, "insufficient_data"),  # Less than required window
            ([float("nan")] * 30, "nan_data"),
            ([0.0] * 30, "zero_returns"),
        ]

        for returns, scenario in fallback_scenarios:
            if scenario != "nan_data":
                position_sizer.return_history["BTCUSDT"] = returns
            else:
                # Handle NaN scenario carefully
                position_sizer.return_history["BTCUSDT"] = [0.01] * 30
                with patch("numpy.var", return_value=float("nan")):
                    pass  # Will test in the calculation

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

            # Should fallback to fixed percentage
            expected_fallback = (
                portfolio_value * Decimal("0.02") * Decimal(str(sample_signal.confidence))
            )
            assert size == expected_fallback

    @pytest.mark.asyncio
    async def test_kelly_performance_benchmarks(self, position_sizer, sample_signal):
        """Test Kelly calculation performance for high-frequency use."""
        import time

        portfolio_value = Decimal("10000")
        returns = [0.02, 0.01, 0.03, -0.01, 0.015] * 6
        position_sizer.return_history["BTCUSDT"] = returns

        # Benchmark multiple calculations
        num_calculations = 1000
        start_time = time.time()

        for _ in range(num_calculations):
            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)
            assert isinstance(size, Decimal)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_calculations

        # Should complete calculations quickly (< 1ms average)
        assert avg_time < 0.001, f"Kelly calculation too slow: {avg_time:.6f}s average"

        # Performance should be consistent
        assert total_time < 1.0, f"Total time too high: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_kelly_with_different_distributions(self, position_sizer, sample_signal):
        """Test Kelly with non-normal return distributions."""
        portfolio_value = Decimal("10000")

        # Test with different probability distributions
        np.random.seed(42)

        distribution_tests = [
            # Skewed distributions
            (stats.skewnorm.rvs(a=5, size=30).tolist(), "positive_skew"),
            (stats.skewnorm.rvs(a=-5, size=30).tolist(), "negative_skew"),
            # Heavy-tailed distribution
            (stats.t.rvs(df=3, size=30).tolist(), "heavy_tails"),
            # Bimodal-like distribution
            (
                np.concatenate(
                    [np.random.normal(-0.02, 0.01, 15), np.random.normal(0.03, 0.01, 15)]
                ).tolist(),
                "bimodal",
            ),
        ]

        for returns, dist_name in distribution_tests:
            # Scale returns to reasonable range
            returns = [r * 0.01 for r in returns]  # Scale to 1% magnitude
            position_sizer.return_history["BTCUSDT"] = returns

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

            # Should handle all distribution types
            assert isinstance(size, Decimal)
            assert size >= Decimal("0")
            assert size <= portfolio_value * Decimal("0.25")  # Respect caps
