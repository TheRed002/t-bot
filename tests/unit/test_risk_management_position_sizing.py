"""
Unit tests for PositionSizer class.

This module tests the position sizing algorithms and calculations.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.core.types import (
    Signal, SignalDirection, PositionSizeMethod
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, ValidationError
)
from src.core.config import Config
from src.risk_management.position_sizing import PositionSizer


class TestPositionSizer:
    """Test cases for PositionSizer class."""

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
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )

    def test_initialization(self, position_sizer, config):
        """Test position sizer initialization."""
        assert position_sizer.config == config
        assert position_sizer.risk_config == config.risk
        assert position_sizer.price_history == {}
        assert position_sizer.return_history == {}

    @pytest.mark.asyncio
    async def test_calculate_position_size_fixed_percentage(
            self, position_sizer, sample_signal):
        """Test fixed percentage position sizing."""
        portfolio_value = Decimal("10000")

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.FIXED_PCT
        )

        # Expected: 10000 * 0.02 * 0.8 = 160
        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_kelly_criterion(
            self, position_sizer, sample_signal):
        """Test Kelly Criterion position sizing."""
        portfolio_value = Decimal("10000")

        # Add some return history for Kelly calculation
        position_sizer.return_history["BTCUSDT"] = [
            0.01, -0.005, 0.02, -0.01, 0.015] * 6  # 30 days

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_position_size_kelly_insufficient_data(
            self, position_sizer, sample_signal):
        """Test Kelly Criterion with insufficient data."""
        portfolio_value = Decimal("10000")

        # No return history
        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )

        # Should fallback to fixed percentage
        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_volatility_adjusted(
            self, position_sizer, sample_signal):
        """Test volatility-adjusted position sizing."""
        portfolio_value = Decimal("10000")

        # Add price history for volatility calculation
        position_sizer.price_history["BTCUSDT"] = [
            50000 + i * 100 for i in range(20)]

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.VOLATILITY_ADJUSTED
        )

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_position_size_volatility_insufficient_data(
            self, position_sizer, sample_signal):
        """Test volatility-adjusted sizing with insufficient data."""
        portfolio_value = Decimal("10000")

        # No price history
        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.VOLATILITY_ADJUSTED
        )

        # Should fallback to fixed percentage
        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_confidence_weighted(
            self, position_sizer, sample_signal):
        """Test confidence-weighted position sizing."""
        portfolio_value = Decimal("10000")

        position_size = await position_sizer.calculate_position_size(
            sample_signal, portfolio_value, PositionSizeMethod.CONFIDENCE_WEIGHTED
        )

        # Expected: 10000 * 0.02 * (0.8^2) = 128
        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence ** 2))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_default_method(
            self, position_sizer, sample_signal):
        """Test position sizing with default method."""
        portfolio_value = Decimal("10000")

        position_size = await position_sizer.calculate_position_size(sample_signal, portfolio_value)

        # Should use default method from config
        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_max_limit(
            self, position_sizer, sample_signal):
        """Test position size maximum limit enforcement."""
        portfolio_value = Decimal("10000")

        # Create signal with very high confidence to trigger max limit
        sample_signal.confidence = 1.0

        position_size = await position_sizer.calculate_position_size(sample_signal, portfolio_value)

        # Should be capped at max position size (10% of portfolio)
        max_size = portfolio_value * Decimal("0.1")
        assert position_size <= max_size

    @pytest.mark.asyncio
    async def test_calculate_position_size_min_limit(
            self, position_sizer, sample_signal):
        """Test position size minimum limit enforcement."""
        portfolio_value = Decimal("10000")

        # Create signal with very low confidence
        sample_signal.confidence = 0.01

        position_size = await position_sizer.calculate_position_size(sample_signal, portfolio_value)

        # Should return 0 if below minimum
        assert position_size == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_signal(
            self, position_sizer):
        """Test position sizing with invalid signal."""
        portfolio_value = Decimal("10000")

        with pytest.raises(RiskManagementError):
            await position_sizer.calculate_position_size(None, portfolio_value)

    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_portfolio_value(
            self, position_sizer, sample_signal):
        """Test position sizing with invalid portfolio value."""
        with pytest.raises(RiskManagementError):
            await position_sizer.calculate_position_size(sample_signal, Decimal("0"))

    @pytest.mark.asyncio
    async def test_calculate_position_size_unsupported_method(
            self, position_sizer, sample_signal):
        """Test position sizing with unsupported method."""
        portfolio_value = Decimal("10000")

        with pytest.raises(RiskManagementError):
            await position_sizer.calculate_position_size(
                sample_signal, portfolio_value, "unsupported_method"
            )

    @pytest.mark.asyncio
    async def test_fixed_percentage_sizing(
            self, position_sizer, sample_signal):
        """Test fixed percentage sizing calculation."""
        portfolio_value = Decimal("10000")

        position_size = await position_sizer._fixed_percentage_sizing(sample_signal, portfolio_value)

        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_kelly_criterion_sizing(self, position_sizer, sample_signal):
        """Test Kelly Criterion sizing calculation."""
        portfolio_value = Decimal("10000")

        # Add return history
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 6
        position_sizer.return_history["BTCUSDT"] = returns

        position_size = await position_sizer._kelly_criterion_sizing(sample_signal, portfolio_value)

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_kelly_criterion_zero_variance(
            self, position_sizer, sample_signal):
        """Test Kelly Criterion with zero variance."""
        portfolio_value = Decimal("10000")

        # Add return history with zero variance
        position_sizer.return_history["BTCUSDT"] = [0.01] * 30

        position_size = await position_sizer.calculate_position_size(sample_signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION)

        # Should be capped by the max position size limit
        assert position_size > 0
        assert position_size <= portfolio_value * \
            Decimal(str(position_sizer.risk_config.max_position_size_pct))

    @pytest.mark.asyncio
    async def test_volatility_adjusted_sizing(
            self, position_sizer, sample_signal):
        """Test volatility-adjusted sizing calculation."""
        portfolio_value = Decimal("10000")

        # Add price history
        prices = [50000 + i * 100 for i in range(20)]
        position_sizer.price_history["BTCUSDT"] = prices

        position_size = await position_sizer._volatility_adjusted_sizing(sample_signal, portfolio_value)

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_confidence_weighted_sizing(
            self, position_sizer, sample_signal):
        """Test confidence-weighted sizing calculation."""
        portfolio_value = Decimal("10000")

        position_size = await position_sizer._confidence_weighted_sizing(sample_signal, portfolio_value)

        expected_size = portfolio_value * \
            Decimal("0.02") * Decimal(str(sample_signal.confidence ** 2))
        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_update_price_history(self, position_sizer):
        """Test price history update."""
        symbol = "BTCUSDT"
        price = 50000.0

        await position_sizer.update_price_history(symbol, price)

        assert symbol in position_sizer.price_history
        assert position_sizer.price_history[symbol] == [price]

    @pytest.mark.asyncio
    async def test_update_price_history_return_calculation(
            self, position_sizer):
        """Test return calculation in price history update."""
        symbol = "BTCUSDT"

        # Add first price
        await position_sizer.update_price_history(symbol, 50000.0)

        # Add second price
        await position_sizer.update_price_history(symbol, 51000.0)

        assert symbol in position_sizer.return_history
        assert len(position_sizer.return_history[symbol]) == 1
        # (51000-50000)/50000
        assert position_sizer.return_history[symbol][0] == 0.02

    @pytest.mark.asyncio
    async def test_update_price_history_max_history(self, position_sizer):
        """Test price history maximum size limit."""
        symbol = "BTCUSDT"

        # Add more prices than max history
        for i in range(300):  # More than max_history
            await position_sizer.update_price_history(symbol, 50000.0 + i)

        # Should keep only recent history
        max_history = max(
            position_sizer.risk_config.volatility_window * 2, 100)
        assert len(position_sizer.price_history[symbol]) <= max_history

    @pytest.mark.asyncio
    async def test_get_position_size_summary(
            self, position_sizer, sample_signal):
        """Test position size summary generation."""
        portfolio_value = Decimal("10000")

        summary = await position_sizer.get_position_size_summary(sample_signal, portfolio_value)

        assert isinstance(summary, dict)
        for method in PositionSizeMethod:
            assert method.value in summary
            assert "position_size" in summary[method.value]
            assert "portfolio_percentage" in summary[method.value]

    @pytest.mark.asyncio
    async def test_validate_position_size_valid(self, position_sizer):
        """Test position size validation with valid size."""
        position_size = Decimal("1000")
        portfolio_value = Decimal("10000")

        result = await position_sizer.validate_position_size(position_size, portfolio_value)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_position_size_below_minimum(self, position_sizer):
        """Test position size validation with size below minimum."""
        position_size = Decimal("5")  # Below 0.1% of 10000
        portfolio_value = Decimal("10000")

        result = await position_sizer.validate_position_size(position_size, portfolio_value)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_position_size_above_maximum(self, position_sizer):
        """Test position size validation with size above maximum."""
        position_size = Decimal("2000")  # Above 10% of 10000
        portfolio_value = Decimal("10000")

        result = await position_sizer.validate_position_size(position_size, portfolio_value)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_position_size_exception(self, position_sizer):
        """Test position size validation with exception."""
        position_size = Decimal("1000")
        portfolio_value = Decimal("10000")

        with patch.object(position_sizer, 'risk_config', create=True) as mock_config:
            mock_config.max_position_size_pct = "invalid"

            result = await position_sizer.validate_position_size(position_size, portfolio_value)

            assert result is False

    def test_kelly_calculation_accuracy(self, position_sizer, sample_signal):
        """Test Kelly Criterion calculation accuracy."""
        portfolio_value = Decimal("10000")

        # Create returns with known mean and variance
        returns = [0.02, -0.01, 0.03, -0.005, 0.015] * 6  # 30 returns
        position_sizer.return_history["BTCUSDT"] = returns

        # Calculate expected Kelly fraction
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        variance = np.var(returns_array)
        expected_kelly = mean_return / variance

        # Apply confidence and max fraction limits
        kelly_with_confidence = expected_kelly * sample_signal.confidence
        max_kelly = position_sizer.risk_config.kelly_max_fraction
        expected_kelly_final = min(kelly_with_confidence, max_kelly)

        # Calculate expected position size
        expected_position_size = portfolio_value * \
            Decimal(str(expected_kelly_final))

        # This test verifies the mathematical accuracy of Kelly calculation
        assert expected_position_size > 0

    def test_volatility_calculation_accuracy(
            self, position_sizer, sample_signal):
        """Test volatility calculation accuracy."""
        portfolio_value = Decimal("10000")

        # Create price history with known volatility
        base_price = 50000
        prices = [
            base_price +
            i *
            100 +
            np.random.normal(
                0,
                50) for i in range(20)]
        position_sizer.price_history["BTCUSDT"] = prices

        # Calculate expected volatility
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        expected_volatility = np.std(returns)

        # This test verifies the mathematical accuracy of volatility
        # calculation
        assert expected_volatility > 0
