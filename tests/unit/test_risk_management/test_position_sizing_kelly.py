"""
Comprehensive unit tests for Kelly Criterion position sizing implementation.

This module tests the improved Kelly Criterion implementation with:
- Half-Kelly for safety
- Proper win/loss probability calculation
- Decimal precision throughout
- Proper bounds (1% min, 25% max)
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.types.risk import PositionSizeMethod
from src.core.types.trading import Signal, SignalDirection
from src.risk_management.position_sizing import PositionSizer


class TestKellyCriterionImproved:
    """Test cases for improved Kelly Criterion implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def position_sizer(self, config):
        """Create position sizer instance."""
        return PositionSizer(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            symbol="BTCUSDT",
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy",
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_kelly_with_positive_edge(self, position_sizer, sample_signal):
        """Test Kelly Criterion with positive edge (profitable strategy)."""
        portfolio_value = Decimal("10000")

        # Create returns with positive edge: 60% win rate, 2:1 win/loss ratio
        # Wins: 0.02 (2%), Losses: -0.01 (1%)
        returns = []
        for _ in range(18):  # 60% wins
            returns.append(0.02)
        for _ in range(12):  # 40% losses
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # With 60% win rate and 2:1 ratio:
        # Full Kelly = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4 (40%)
        # Half Kelly = 0.4 * 0.5 = 0.2 (20%)
        # With 0.8 strength = 0.2 * 0.8 = 0.16 (16%)
        # But will be capped at 2% by config risk_per_trade

        assert position_size >= Decimal("100")  # At least 1%
        assert position_size <= Decimal("200")  # Max 2% (risk_per_trade limit)
        # Should be capped at 2% = 200
        assert position_size == Decimal("200")

    @pytest.mark.asyncio
    async def test_kelly_with_negative_edge(self, position_sizer, sample_signal):
        """Test Kelly Criterion with negative edge (losing strategy)."""
        portfolio_value = Decimal("10000")

        # Create returns with negative edge: 40% win rate, 1:1 ratio
        returns = []
        for _ in range(12):  # 40% wins
            returns.append(0.01)
        for _ in range(18):  # 60% losses
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # With negative edge, should return minimum position (1%)
        assert position_size == portfolio_value * Decimal("0.01")

    @pytest.mark.asyncio
    async def test_kelly_half_kelly_safety(self, position_sizer, sample_signal):
        """Test that Half-Kelly is properly applied for safety."""
        portfolio_value = Decimal("10000")

        # Create returns with very high edge to test Half-Kelly
        # 80% win rate, 3:1 ratio - would give very high full Kelly
        returns = []
        for _ in range(24):  # 80% wins
            returns.append(0.03)
        for _ in range(6):  # 20% losses
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Even with very high edge, should be capped at 25%
        assert position_size <= portfolio_value * Decimal("0.25")

    @pytest.mark.asyncio
    async def test_kelly_minimum_bound(self, position_sizer, sample_signal):
        """Test Kelly Criterion respects minimum 1% bound."""
        portfolio_value = Decimal("10000")

        # Create returns with very small positive edge
        returns = []
        for _ in range(16):  # Slightly positive
            returns.append(0.001)
        for _ in range(14):
            returns.append(-0.001)

        position_sizer.return_history["BTCUSDT"] = returns

        # Use very low confidence to trigger minimum
        sample_signal.strength = 0.01

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should be at least 1% or 0 (rejected)
        if position_size > 0:
            assert position_size >= portfolio_value * Decimal("0.01")

    @pytest.mark.asyncio
    async def test_kelly_maximum_bound(self, position_sizer, sample_signal):
        """Test Kelly Criterion respects maximum 25% bound."""
        portfolio_value = Decimal("10000")

        # Create returns with extremely high edge
        returns = []
        for _ in range(29):  # 97% win rate
            returns.append(0.05)
        for _ in range(1):  # 3% loss rate
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns
        sample_signal.strength = 1.0  # Maximum confidence

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should never exceed 2% (risk_per_trade config limit)
        assert position_size <= portfolio_value * Decimal("0.02")

    @pytest.mark.asyncio
    async def test_kelly_decimal_precision(self, position_sizer, sample_signal):
        """Test Kelly Criterion maintains Decimal precision throughout."""
        portfolio_value = Decimal("10000.12345678")

        returns = [Decimal("0.01234567") for _ in range(18)]
        returns.extend([Decimal("-0.00567890") for _ in range(12)])

        # Convert to float for storage (as it comes from market data)
        position_sizer.return_history["BTCUSDT"] = [float(r) for r in returns]

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Result should be a Decimal
        assert isinstance(position_size, Decimal)
        # Should have calculated with precision
        assert position_size > 0

    @pytest.mark.asyncio
    async def test_kelly_all_winning_trades(self, position_sizer, sample_signal):
        """Test Kelly Criterion with all winning trades (edge case)."""
        portfolio_value = Decimal("10000")

        # All winning trades - should fallback to fixed percentage
        returns = [0.01 for _ in range(30)]
        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should fallback to fixed percentage (2% * 0.8 confidence = 1.6%)
        expected = portfolio_value * Decimal("0.02") * Decimal("0.8")
        assert position_size == expected

    @pytest.mark.asyncio
    async def test_kelly_all_losing_trades(self, position_sizer, sample_signal):
        """Test Kelly Criterion with all losing trades (edge case)."""
        portfolio_value = Decimal("10000")

        # All losing trades - should fallback to fixed percentage
        returns = [-0.01 for _ in range(30)]
        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should fallback to fixed percentage
        expected = portfolio_value * Decimal("0.02") * Decimal("0.8")
        assert position_size == expected

    @pytest.mark.asyncio
    async def test_kelly_very_small_losses(self, position_sizer, sample_signal):
        """Test Kelly Criterion with very small losses (near-zero division)."""
        portfolio_value = Decimal("10000")

        # Very small losses that could cause division issues
        returns = []
        for _ in range(20):
            returns.append(0.01)
        for _ in range(10):
            returns.append(-0.00001)  # Very small loss

        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should handle gracefully and fallback if needed
        assert position_size > 0
        assert position_size <= portfolio_value * Decimal("0.25")

    @pytest.mark.asyncio
    async def test_kelly_confidence_adjustment(self, position_sizer):
        """Test Kelly Criterion properly adjusts for signal confidence."""
        portfolio_value = Decimal("10000")

        # Fixed returns for consistent Kelly fraction
        returns = []
        for _ in range(18):  # 60% wins
            returns.append(0.02)
        for _ in range(12):  # 40% losses
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns

        # Test with different confidence levels
        confidences = [0.2, 0.5, 0.8, 1.0]
        sizes = []

        for confidence in confidences:
            signal = Signal(
                symbol="BTCUSDT",
                direction=SignalDirection.BUY,
                strength=confidence,
                timestamp=datetime.now(timezone.utc),
                source="test_strategy",
                metadata={},
            )

            size = await position_sizer.calculate_position_size(
                signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
            )
            sizes.append(size)

        # Higher confidence should result in larger position sizes
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    @pytest.mark.asyncio
    async def test_kelly_win_loss_ratio_calculation(self, position_sizer, sample_signal):
        """Test accurate win/loss ratio calculation in Kelly formula."""
        portfolio_value = Decimal("10000")

        # Create specific win/loss pattern
        # Wins: 15 trades of +3% = average win of 3%
        # Losses: 15 trades of -1% = average loss of 1%
        # Win rate: 50%, Win/Loss ratio: 3:1
        returns = []
        for _ in range(15):
            returns.append(0.03)  # Win
        for _ in range(15):
            returns.append(-0.01)  # Loss

        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # With 50% win rate and 3:1 ratio:
        # Full Kelly = (0.5 * 3 - 0.5) / 3 = 1.0 / 3 = 0.333 (33.3%)
        # Half Kelly = 0.333 * 0.5 = 0.1665 (16.65%)
        # With 0.8 strength = 0.1665 * 0.8 = 0.1332 (13.32%)
        # But will be capped at 2% by config risk_per_trade

        # Should be capped at 2% = 200
        expected = portfolio_value * Decimal("0.02")  # Capped at 2%
        assert position_size == expected

    @pytest.mark.asyncio
    async def test_kelly_logging_detail(self, position_sizer, sample_signal, caplog, capsys):
        """Test that Kelly Criterion logs detailed calculation info."""
        portfolio_value = Decimal("10000")

        returns = []
        for _ in range(18):
            returns.append(0.02)
        for _ in range(12):
            returns.append(-0.01)

        position_sizer.return_history["BTCUSDT"] = returns

        with caplog.at_level("DEBUG"):
            await position_sizer.calculate_position_size(
                sample_signal,
                portfolio_value,
                PositionSizeMethod.KELLY_CRITERION,
            )

        # Check that detailed Kelly information is logged in structured output (stdout)
        captured = capsys.readouterr()
        stdout_text = captured.out
        assert "Kelly Criterion sizing (Half-Kelly)" in stdout_text
        assert "win_probability" in stdout_text
        assert "loss_probability" in stdout_text
        assert "win_loss_ratio" in stdout_text
        assert "half_kelly" in stdout_text
        assert "final_fraction" in stdout_text

    @pytest.mark.asyncio
    async def test_validate_position_with_new_bounds(self, position_sizer):
        """Test position validation with new 1% min, config max (2%) bounds."""
        portfolio_value = Decimal("10000")

        # Test below minimum (1%)
        assert not await position_sizer.validate_position_size(
            Decimal("99"),
            portfolio_value,  # 0.99%
        )

        # Test at minimum
        assert await position_sizer.validate_position_size(
            Decimal("100"),
            portfolio_value,  # 1%
        )

        # Test valid range
        assert await position_sizer.validate_position_size(
            Decimal("150"),
            portfolio_value,  # 1.5%
        )

        # Test at config maximum (2%)
        assert await position_sizer.validate_position_size(
            Decimal("200"),
            portfolio_value,  # 2%
        )

        # Test above config maximum
        assert not await position_sizer.validate_position_size(
            Decimal("201"),
            portfolio_value,  # 10.01%
        )
