"""
Optimized test suite for monitoring financial validation module.

Fast tests with minimal overhead and batch operations.
"""

import logging
from decimal import Decimal

import pytest

# Disable logging for maximum speed
logging.disable(logging.CRITICAL)

# Import actual financial validation functions for testing
from src.monitoring.financial_validation import (
    calculate_pnl_percentage,
    validate_drawdown_percent,
    validate_execution_time,
    validate_pnl_usd,
    validate_portfolio_value,
    validate_position_size_usd,
    validate_price,
    validate_quantity,
    validate_sharpe_ratio,
    validate_slippage_bps,
    validate_timeframe,
    validate_var,
    validate_volume_usd,
)


class TestBasicValidation:
    """Test basic validation functions."""

    def test_validate_price_success(self):
        """Test successful price validation."""
        result = validate_price(100.0, "BTC-USD")
        assert isinstance(result, Decimal)
        assert result > 0

    def test_validate_price_none(self):
        """Test price validation with None."""
        with pytest.raises(ValueError):
            validate_price(None, "BTC-USD")

    def test_validate_price_negative(self):
        """Test price validation with negative value."""
        with pytest.raises(ValueError):
            validate_price(-100, "BTC-USD")

    def test_validate_quantity_success(self):
        """Test successful quantity validation."""
        result = validate_quantity(1.0, "BTC")
        assert isinstance(result, Decimal)
        assert result > 0

    def test_validate_quantity_none(self):
        """Test quantity validation with None."""
        with pytest.raises(ValueError):
            validate_quantity(None, "BTC")

    def test_validate_pnl_usd_positive(self):
        """Test PnL validation with positive value."""
        result = validate_pnl_usd(1000.0)
        assert result == 1000.0

    def test_validate_pnl_usd_negative(self):
        """Test PnL validation with negative value."""
        result = validate_pnl_usd(-500.0)
        assert result == -500.0

    def test_validate_pnl_usd_invalid(self):
        """Test PnL validation with invalid type."""
        with pytest.raises(ValueError):
            validate_pnl_usd("invalid")

    def test_validate_volume_usd_success(self):
        """Test successful volume validation."""
        result = validate_volume_usd(5000.0)
        assert result == 5000.0

    def test_validate_volume_usd_zero(self):
        """Test volume validation with zero."""
        result = validate_volume_usd(0)
        assert result == 0.0

    def test_validate_volume_usd_negative(self):
        """Test volume validation with negative value."""
        with pytest.raises(ValueError):
            validate_volume_usd(-1000)

    def test_validate_slippage_bps_success(self):
        """Test successful slippage validation."""
        result = validate_slippage_bps(50.0)
        assert result == 50.0

    def test_validate_slippage_bps_negative(self):
        """Test slippage validation with negative value."""
        result = validate_slippage_bps(-25.0)
        assert result == -25.0

    def test_validate_execution_time_success(self):
        """Test successful execution time validation."""
        result = validate_execution_time(0.5)
        assert result == 0.5

    def test_validate_execution_time_zero(self):
        """Test execution time validation with zero."""
        result = validate_execution_time(0)
        assert result == 0.0

    def test_validate_execution_time_negative(self):
        """Test execution time validation with negative value."""
        with pytest.raises(ValueError):
            validate_execution_time(-1.0)

    def test_validate_var_success(self):
        """Test successful VaR validation."""
        result = validate_var(10000.0, 0.95)
        assert result == 10000.0

    def test_validate_var_invalid_confidence(self):
        """Test VaR validation with invalid confidence."""
        with pytest.raises(ValueError):
            validate_var(10000.0, 1.5)

    def test_validate_drawdown_percent_success(self):
        """Test successful drawdown validation."""
        result = validate_drawdown_percent(5.0)
        assert result == 5.0

    def test_validate_drawdown_percent_zero(self):
        """Test drawdown validation with zero."""
        result = validate_drawdown_percent(0)
        assert result == 0.0

    def test_validate_drawdown_percent_negative(self):
        """Test drawdown validation with negative value."""
        with pytest.raises(ValueError):
            validate_drawdown_percent(-5.0)

    def test_validate_sharpe_ratio_positive(self):
        """Test sharpe ratio validation with positive value."""
        result = validate_sharpe_ratio(1.5)
        assert result == 1.5

    def test_validate_sharpe_ratio_negative(self):
        """Test sharpe ratio validation with negative value."""
        result = validate_sharpe_ratio(-0.5)
        assert result == -0.5

    def test_validate_sharpe_ratio_zero(self):
        """Test sharpe ratio validation with zero."""
        result = validate_sharpe_ratio(0)
        assert result == 0.0

    def test_validate_portfolio_value_success(self):
        """Test successful portfolio value validation."""
        result = validate_portfolio_value(100000.0)
        assert result == 100000.0

    def test_validate_portfolio_value_zero(self):
        """Test portfolio value validation with zero."""
        result = validate_portfolio_value(0)
        assert result == 0.0

    def test_validate_portfolio_value_negative(self):
        """Test portfolio value validation with negative value."""
        with pytest.raises(ValueError):
            validate_portfolio_value(-50000)

    def test_validate_timeframe_success(self):
        """Test successful timeframe validation."""
        # Test single case for speed
        result = validate_timeframe("1h")
        assert result == "1h"

    def test_validate_timeframe_invalid(self):
        """Test timeframe validation with invalid value."""
        with pytest.raises(ValueError):
            validate_timeframe("invalid")

    def test_calculate_pnl_percentage_positive(self):
        """Test PnL percentage calculation with positive gain."""
        result = calculate_pnl_percentage(1000, 10000)
        assert result == 10.0

    def test_calculate_pnl_percentage_negative(self):
        """Test PnL percentage calculation with loss."""
        result = calculate_pnl_percentage(-500, 10000)
        assert result == -5.0

    def test_calculate_pnl_percentage_zero_portfolio(self):
        """Test PnL percentage calculation with zero portfolio."""
        with pytest.raises(ValueError):
            calculate_pnl_percentage(1000, 0)

    def test_validate_position_size_usd_success(self):
        """Test successful position size validation."""
        result = validate_position_size_usd(5000.0)
        assert result == 5000.0

    def test_validate_position_size_usd_zero(self):
        """Test position size validation with zero."""
        result = validate_position_size_usd(0)
        assert result == 0.0

    def test_validate_position_size_usd_negative(self):
        """Test position size validation with negative value."""
        with pytest.raises(ValueError):
            validate_position_size_usd(-1000)
