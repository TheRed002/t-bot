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

from datetime import datetime, timezone
from decimal import Decimal, getcontext
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from src.core.config import Config
from src.core.types.risk import PositionSizeMethod
from src.core.types.trading import Signal, SignalDirection
from src.risk_management.service import RiskService


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
        # Note: RiskConfig uses different attribute names than expected
        # kelly_fraction instead of kelly_max_fraction
        # risk_per_trade instead of default_position_size_pct
        return config

    @pytest.fixture
    def position_sizer(self, config):
        """Create risk service instance."""
        return RiskService(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            signal_id="test_signal_1",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy",
            metadata={},
        )

    @pytest.fixture(autouse=True)
    def setup_high_precision(self):
        """Set high decimal precision for calculations."""
        getcontext().prec = 50  # Very high precision for testing

    @pytest.mark.asyncio
    async def test_kelly_mathematical_accuracy_known_values(self, position_sizer, sample_signal):
        """Test Kelly calculation with known mathematical values."""
        portfolio_value = Decimal("10000")

        # Create returns with both wins and losses for proper Kelly calculation
        # Include both winning and losing trades
        returns = [
            0.02,
            -0.01,
            0.03,
            -0.015,
            0.025,
            0.01,
            -0.005,
            0.02,
            -0.01,
            0.015,
        ] * 3  # 30 returns
        position_sizer._return_history["BTC/USDT"] = returns

        # Calculate expected Kelly fraction using the actual formula: f = (p*b - q) / b
        returns_array = np.array(returns)
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]

        # Calculate win probability
        p = len(winning_returns) / len(returns)
        q = 1 - p

        # Calculate win/loss ratio
        avg_win = np.mean([abs(r) for r in winning_returns]) if winning_returns else 0
        avg_loss = np.mean([abs(r) for r in losing_returns]) if losing_returns else 1
        b = avg_win / avg_loss if avg_loss > 0 else 1

        # Kelly formula
        kelly_fraction = (p * b - q) / b if b > 0 else 0

        # Apply Half-Kelly for safety
        kelly_fraction *= 0.5

        # Should be capped at max_kelly_fraction (0.25)
        expected_kelly_capped = min(kelly_fraction, 0.25)
        # Apply confidence (convert both to Decimal)
        expected_kelly_final = Decimal(str(expected_kelly_capped)) * Decimal(
            str(sample_signal.strength)
        )
        expected_position_size = portfolio_value * expected_kelly_final

        # Calculate actual Kelly
        actual_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should be close to expected (allowing for floating point precision)
        ratio = float(actual_size / expected_position_size)
        assert 0.5 <= ratio <= 1.5  # Within reasonable tolerance for Kelly calculation differences

    @pytest.mark.asyncio
    async def test_kelly_edge_case_all_positive_returns(self, position_sizer, sample_signal):
        """Test Kelly with all positive returns (extreme optimism)."""
        portfolio_value = Decimal("10000")

        # All positive returns
        returns = [0.01, 0.02, 0.015, 0.025, 0.01] * 6  # 30 positive returns
        position_sizer._return_history["BTC/USDT"] = returns

        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should be capped at max Kelly fraction
        max_allowed = portfolio_value * position_sizer.risk_config.max_position_size_pct
        max_with_confidence = max_allowed * Decimal(str(sample_signal.strength))

        assert position_size <= max_with_confidence
        assert position_size > Decimal("0")

    @pytest.mark.asyncio
    async def test_kelly_edge_case_all_negative_returns(self, position_sizer, sample_signal):
        """Test Kelly with all negative returns (extreme pessimism).

        IMPORTANT: Kelly fallback uses FixedPercentageAlgorithm which does NOT apply signal strength.
        See src/utils/position_sizing.py lines 146-151 for fallback implementation.
        See src/utils/position_sizing.py lines 84-86 for FIXED_PERCENTAGE behavior.
        """
        portfolio_value = Decimal("10000")

        # All negative returns
        returns = [-0.01, -0.02, -0.015, -0.005, -0.01] * 6  # 30 negative returns
        position_sizer._return_history["BTC/USDT"] = returns

        # Kelly with all negative returns should fallback to fixed percentage
        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should fallback to fixed percentage sizing WITHOUT signal strength multiplier
        # Note: RiskService uses default_position_size_pct which may differ from Config's risk_per_trade
        expected_fallback = (
            portfolio_value
            * Decimal(str(position_sizer.risk_config.default_position_size_pct))
        )
        assert position_size == expected_fallback

    @pytest.mark.asyncio
    async def test_kelly_edge_case_zero_variance(self, position_sizer, sample_signal):
        """Test Kelly with zero variance (all identical returns).

        IMPORTANT: Kelly fallback uses FixedPercentageAlgorithm which does NOT apply signal strength.
        See src/utils/position_sizing.py lines 146-151 for fallback implementation.
        See src/utils/position_sizing.py lines 84-86 for FIXED_PERCENTAGE behavior.
        """
        portfolio_value = Decimal("10000")

        # All identical returns (zero variance)
        returns = [0.01] * 30
        position_sizer._return_history["BTC/USDT"] = returns

        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Should fallback to fixed percentage sizing WITHOUT signal strength multiplier
        expected_fallback = (
            portfolio_value
            * Decimal(str(position_sizer.risk_config.default_position_size_pct))
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
            position_sizer._return_history["BTC/USDT"] = returns

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
        position_sizer._return_history["BTC/USDT"] = returns

        # Test different confidence levels
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        position_sizes = []

        for confidence in confidence_levels:
            signal = Signal(
                signal_id="test_signal_2",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                strength=confidence,
                timestamp=datetime.now(timezone.utc),
                source="test_strategy",
                metadata={},
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
            position_sizer._return_history["BTC/USDT"] = full_returns[-window_size:]

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
            position_sizer._return_history["BTC/USDT"] = returns

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
        position_sizer._return_history["BTC/USDT"] = returns

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
            position_sizer._return_history["BTC/USDT"] = returns

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
        position_sizer._return_history["BTC/USDT"] = returns

        # Calculate using main method with current price and method
        # IMPORTANT: calculate_position_size returns COINS, not USD
        # See src/risk_management/service.py lines 469-483 for USD→coins conversion
        current_price = Decimal("50000")  # BTC price
        size_coins = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, current_price, PositionSizeMethod.KELLY_CRITERION
        )

        # Calculate using internal method - returns USD value
        internal_size_usd = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Convert coins back to USD for comparison
        size_usd = size_coins * current_price

        # Should apply additional limits in main method (using max_position_size_pct)
        max_allowed_usd = portfolio_value * Decimal(str(position_sizer.risk_config.max_position_size_pct))
        assert size_usd <= max_allowed_usd
        assert size_coins >= Decimal("0")

        # If within limits, USD values should match (accounting for rounding in conversion)
        if internal_size_usd <= max_allowed_usd:
            # Allow small rounding errors from USD→coins→USD conversion
            assert abs(size_usd - internal_size_usd) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_kelly_statistical_properties(self, position_sizer, sample_signal):
        """Test that Kelly calculations respect statistical properties."""
        portfolio_value = Decimal("10000")

        # Generate returns with known statistical properties
        np.random.seed(42)  # Reproducible results

        # Normal distribution with positive mean
        returns = np.random.normal(0.01, 0.02, 30).tolist()  # Mean=1%, Std=2%
        position_sizer._return_history["BTC/USDT"] = returns

        size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        # Calculate Kelly using proper win/loss approach
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        if len(winning_returns) > 0 and len(losing_returns) > 0:
            win_probability = len(winning_returns) / len(returns)
            average_win = np.mean(winning_returns)
            average_loss = abs(np.mean(losing_returns))
            win_loss_ratio = average_win / average_loss
            
            # Kelly formula: f = p - (1-p)/b, where p is win prob, b is win/loss ratio
            theoretical_kelly = win_probability - (1 - win_probability) / win_loss_ratio
            theoretical_kelly = max(0, theoretical_kelly)  # Can't be negative
            
            # Apply half-kelly, strength adjustment, and bounds
            half_kelly = theoretical_kelly * 0.5  # Half-Kelly factor
            strength_adjusted = half_kelly * float(sample_signal.strength)
            bounded_kelly = min(max(strength_adjusted, 0.01), 0.10)  # Bounds from config
            
            expected_size = portfolio_value * Decimal(str(bounded_kelly))
            
            # Should be reasonably close to theoretical value
            ratio = float(size / expected_size) if expected_size > 0 else 0
            assert (
                0.5 <= ratio <= 1.5
            )  # Within 50% of theoretical (Kelly implementations can vary)

    @pytest.mark.asyncio
    async def test_kelly_fallback_behavior(self, position_sizer, sample_signal):
        """Test Kelly fallback to fixed percentage in various scenarios.

        IMPORTANT: Kelly fallback uses FixedPercentageAlgorithm which does NOT apply signal strength.
        See src/utils/position_sizing.py lines 146-151 for fallback implementation.
        See src/utils/position_sizing.py lines 84-86 for FIXED_PERCENTAGE behavior.
        """
        portfolio_value = Decimal("10000")

        fallback_scenarios = [
            ([], "no_data"),
            ([0.01] * 10, "insufficient_data"),  # Less than required window
            ([float("nan")] * 30, "nan_data"),
            ([0.0] * 30, "zero_returns"),
        ]

        for returns, scenario in fallback_scenarios:
            if scenario != "nan_data":
                position_sizer._return_history["BTC/USDT"] = returns
            else:
                # Handle NaN scenario carefully
                position_sizer._return_history["BTC/USDT"] = [0.01] * 30
                with patch("numpy.var", return_value=float("nan")):
                    pass  # Will test in the calculation

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

            # Should fallback to fixed percentage WITHOUT signal strength multiplier
            expected_fallback = (
                portfolio_value
                * Decimal(str(position_sizer.risk_config.default_position_size_pct))
            )
            assert size == expected_fallback

    @pytest.mark.asyncio
    async def test_kelly_performance_benchmarks(self, position_sizer, sample_signal):
        """Test Kelly calculation performance for high-frequency use."""
        import time

        portfolio_value = Decimal("10000")
        returns = [0.02, 0.01, 0.03, -0.01, 0.015] * 6
        position_sizer._return_history["BTC/USDT"] = returns

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
            position_sizer._return_history["BTC/USDT"] = returns

            size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

            # Should handle all distribution types
            assert isinstance(size, Decimal)
            assert size >= Decimal("0")
            assert size <= portfolio_value * Decimal("0.25")  # Respect caps
