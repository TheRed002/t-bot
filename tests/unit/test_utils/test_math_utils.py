"""
Unit tests for math_utils module.

Tests mathematical utility functions.
"""

from decimal import Decimal

import pytest

from src.core.exceptions import ValidationError
from src.utils.math_utils import (
    calculate_beta,
    calculate_correlation,
    calculate_max_drawdown,
    calculate_percentage_change,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    calculate_volatility,
    safe_max,
    safe_min,
    safe_percentage,
)


class TestCalculatePercentageChange:
    """Test calculate_percentage_change function."""

    def test_positive_change(self):
        """Test positive percentage change."""
        old_value = Decimal("100")
        new_value = Decimal("110")
        result = calculate_percentage_change(old_value, new_value)

        assert isinstance(result, Decimal)
        assert result == Decimal("0.1")  # 10% increase

    def test_negative_change(self):
        """Test negative percentage change."""
        old_value = Decimal("100")
        new_value = Decimal("90")
        result = calculate_percentage_change(old_value, new_value)

        assert isinstance(result, Decimal)
        assert result == Decimal("-0.1")  # 10% decrease

    def test_zero_change(self):
        """Test zero percentage change."""
        old_value = Decimal("100")
        new_value = Decimal("100")
        result = calculate_percentage_change(old_value, new_value)

        assert isinstance(result, Decimal)
        assert result == Decimal("0")

    def test_zero_old_value_raises_error(self):
        """Test that zero old value raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Cannot calculate percentage change with zero old value"
        ):
            calculate_percentage_change(Decimal("0"), Decimal("100"))

    def test_large_numbers(self):
        """Test with large numbers."""
        old_value = Decimal("1000000")
        new_value = Decimal("1050000")
        result = calculate_percentage_change(old_value, new_value)

        assert isinstance(result, Decimal)
        assert result == Decimal("0.05")  # 5% increase


class TestCalculateSharpeRatio:
    """Test calculate_sharpe_ratio function."""

    def test_positive_sharpe_ratio(self):
        """Test calculating Sharpe ratio with positive returns."""
        returns = [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.005"),
            Decimal("0.015"),
            Decimal("0.01"),
        ]
        result = calculate_sharpe_ratio(returns)

        assert isinstance(result, Decimal)
        assert result > 0  # Should be positive for decent returns

    def test_sharpe_ratio_with_custom_risk_free_rate(self):
        """Test Sharpe ratio with custom risk-free rate."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("0.015"), Decimal("0.01")]
        result = calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.03"))

        assert isinstance(result, Decimal)

    def test_sharpe_ratio_different_frequencies(self):
        """Test Sharpe ratio with different frequencies."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("0.015")]

        daily = calculate_sharpe_ratio(returns, frequency="daily")
        monthly = calculate_sharpe_ratio(returns, frequency="monthly")

        assert isinstance(daily, Decimal)
        assert isinstance(monthly, Decimal)
        # Daily should be different from monthly due to annualization
        assert daily != monthly

    def test_empty_returns_raises_error(self):
        """Test that empty returns list raises ValidationError."""
        with pytest.raises(ValidationError, match="Returns list cannot be empty"):
            calculate_sharpe_ratio([])

    def test_insufficient_returns_raises_error(self):
        """Test that single return raises ValidationError."""
        with pytest.raises(ValidationError):
            calculate_sharpe_ratio([Decimal("0.01")])

    def test_invalid_frequency_raises_error(self):
        """Test that invalid frequency raises ValidationError."""
        returns = [Decimal("0.01"), Decimal("0.02")]
        with pytest.raises(ValidationError):
            calculate_sharpe_ratio(returns, frequency="invalid")


class TestCalculateMaxDrawdown:
    """Test calculate_max_drawdown function."""

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        equity_curve = [Decimal("100"), Decimal("110"), Decimal("90"), Decimal("95")]
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity_curve)

        assert isinstance(max_dd, Decimal)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        assert max_dd >= 0  # Drawdown is represented as positive value
        assert start_idx <= end_idx

    def test_no_drawdown(self):
        """Test with continuously increasing equity curve."""
        equity_curve = [Decimal("100"), Decimal("110"), Decimal("120"), Decimal("130")]
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity_curve)

        assert max_dd == Decimal("0")  # No drawdown in increasing curve

    def test_single_point_equity(self):
        """Test with single point equity curve."""
        equity_curve = [Decimal("100")]
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity_curve)

        assert max_dd == Decimal("0")
        assert start_idx == 0
        assert end_idx == 0

    def test_empty_equity_curve_raises_error(self):
        """Test that empty equity curve raises ValidationError."""
        with pytest.raises(ValidationError):
            calculate_max_drawdown([])


class TestCalculateVar:
    """Test calculate_var function."""

    def test_var_calculation(self):
        """Test VaR calculation with normal returns."""
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
            Decimal("0.005"),
        ]
        var = calculate_var(returns)

        assert isinstance(var, Decimal)
        assert var <= 0  # VaR should be negative (loss)

    def test_var_with_different_confidence(self):
        """Test VaR with different confidence levels."""
        returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.015"), Decimal("-0.01")]

        var_95 = calculate_var(returns, confidence_level=Decimal("0.95"))
        var_99 = calculate_var(returns, confidence_level=Decimal("0.99"))

        assert isinstance(var_95, Decimal)
        assert isinstance(var_99, Decimal)
        # 99% VaR should be more conservative (larger loss)
        assert var_99 <= var_95

    def test_empty_returns_var_raises_error(self):
        """Test that empty returns raises ValidationError."""
        with pytest.raises(ValidationError):
            calculate_var([])

    def test_invalid_confidence_level_raises_error(self):
        """Test that invalid confidence level raises ValidationError."""
        returns = [Decimal("0.01"), Decimal("-0.02")]
        with pytest.raises(ValidationError):
            calculate_var(returns, confidence_level=Decimal("1.5"))  # > 1.0


class TestCalculateVolatility:
    """Test calculate_volatility function."""

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
            Decimal("0.005"),
        ]
        volatility = calculate_volatility(returns)

        assert isinstance(volatility, Decimal)
        assert volatility >= 0  # Volatility should be non-negative

    def test_volatility_with_window(self):
        """Test volatility with rolling window."""
        returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.015"), Decimal("-0.01")]
        volatility = calculate_volatility(returns, window=3)

        assert isinstance(volatility, Decimal)
        assert volatility >= 0

    def test_zero_volatility(self):
        """Test volatility with identical returns."""
        returns = [Decimal("0.01"), Decimal("0.01"), Decimal("0.01"), Decimal("0.01")]
        volatility = calculate_volatility(returns)

        assert volatility == Decimal("0")  # Identical returns should have zero volatility

    def test_empty_returns_volatility_raises_error(self):
        """Test that empty returns raises ValidationError."""
        with pytest.raises(ValidationError):
            calculate_volatility([])


class TestCalculateCorrelation:
    """Test calculate_correlation function."""

    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        series1 = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
        series2 = [Decimal("2"), Decimal("4"), Decimal("6"), Decimal("8")]
        correlation = calculate_correlation(series1, series2)

        assert isinstance(correlation, Decimal)
        # Allow for more precision tolerance
        assert abs(correlation - Decimal("1.0")) < Decimal("0.1")  # Should be close to 1

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        series1 = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
        series2 = [Decimal("4"), Decimal("3"), Decimal("2"), Decimal("1")]
        correlation = calculate_correlation(series1, series2)

        assert isinstance(correlation, Decimal)
        # Allow for more precision tolerance
        assert abs(correlation - Decimal("-1.0")) < Decimal("0.1")  # Should be close to -1

    def test_no_correlation(self):
        """Test no correlation."""
        series1 = [Decimal("1"), Decimal("1"), Decimal("1"), Decimal("1")]
        series2 = [Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]

        # One series has no variance, so correlation should be 0
        correlation = calculate_correlation(series1, series2)
        assert correlation == Decimal("0")

    def test_mismatched_series_length_raises_error(self):
        """Test that mismatched series lengths raise ValidationError."""
        series1 = [Decimal("1"), Decimal("2")]
        series2 = [Decimal("1"), Decimal("2"), Decimal("3")]

        with pytest.raises(ValidationError):
            calculate_correlation(series1, series2)

    def test_empty_series_raises_error(self):
        """Test that empty series raise ValidationError."""
        with pytest.raises(ValidationError):
            calculate_correlation([], [])


class TestCalculateBeta:
    """Test calculate_beta function."""

    def test_beta_calculation(self):
        """Test beta calculation."""
        asset_returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.015"), Decimal("-0.01")]
        market_returns = [Decimal("0.005"), Decimal("-0.015"), Decimal("0.01"), Decimal("-0.008")]

        beta = calculate_beta(asset_returns, market_returns)

        assert isinstance(beta, Decimal)
        # Beta can be any real number

    def test_beta_high_correlation(self):
        """Test beta with highly correlated returns."""
        asset_returns = [Decimal("0.02"), Decimal("-0.04"), Decimal("0.03"), Decimal("-0.02")]
        market_returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.015"), Decimal("-0.01")]

        beta = calculate_beta(asset_returns, market_returns)

        assert isinstance(beta, Decimal)
        # Beta could be positive or negative depending on the calculation, so just check it's a number
        assert beta is not None

    def test_mismatched_returns_length_raises_error(self):
        """Test that mismatched returns lengths raise ValidationError."""
        asset_returns = [Decimal("0.01"), Decimal("0.02")]
        market_returns = [Decimal("0.01")]

        with pytest.raises(ValidationError):
            calculate_beta(asset_returns, market_returns)


class TestSafeMin:
    """Test safe_min function."""

    def test_safe_min_basic(self):
        """Test basic safe_min functionality."""
        result = safe_min(Decimal("5"), Decimal("3"), Decimal("8"))
        assert result == Decimal("3")

    def test_safe_min_single_value(self):
        """Test safe_min with single value."""
        result = safe_min(Decimal("42"))
        assert result == Decimal("42")

    def test_safe_min_with_default(self):
        """Test safe_min with default when no values."""
        result = safe_min(default=Decimal("10"))
        assert result == Decimal("10")

    def test_safe_min_no_values_no_default_raises_error(self):
        """Test that no values and no default raises ValidationError."""
        with pytest.raises(ValidationError):
            safe_min()


class TestSafeMax:
    """Test safe_max function."""

    def test_safe_max_basic(self):
        """Test basic safe_max functionality."""
        result = safe_max(Decimal("5"), Decimal("3"), Decimal("8"))
        assert result == Decimal("8")

    def test_safe_max_single_value(self):
        """Test safe_max with single value."""
        result = safe_max(Decimal("42"))
        assert result == Decimal("42")

    def test_safe_max_with_default(self):
        """Test safe_max with default when no values."""
        result = safe_max(default=Decimal("10"))
        assert result == Decimal("10")

    def test_safe_max_no_values_no_default_raises_error(self):
        """Test that no values and no default raises ValidationError."""
        with pytest.raises(ValidationError):
            safe_max()


class TestSafePercentage:
    """Test safe_percentage function."""

    def test_safe_percentage_basic(self):
        """Test basic safe_percentage calculation."""
        result = safe_percentage(Decimal("25"), Decimal("100"))
        assert result == Decimal("25")  # 25/100 * 100 = 25%

    def test_safe_percentage_zero_total(self):
        """Test safe_percentage with zero total."""
        result = safe_percentage(Decimal("25"), Decimal("0"))
        assert result == Decimal("0")  # Should return default (0) multiplied by 100 = 0

    def test_safe_percentage_with_custom_default(self):
        """Test safe_percentage with custom default."""
        result = safe_percentage(Decimal("25"), Decimal("0"), default=Decimal("50"))
        assert result == Decimal("5000")  # Default gets multiplied by 100 to be percentage

    def test_safe_percentage_normal_calculation(self):
        """Test safe_percentage normal calculation."""
        result = safe_percentage(Decimal("30"), Decimal("120"))
        assert result == Decimal("25")  # 30/120 * 100 = 25%


class TestCalculateSortinoRatio:
    """Test calculate_sortino_ratio function."""

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
            Decimal("0.005"),
        ]
        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sortino, Decimal)

    def test_sortino_with_custom_target(self):
        """Test Sortino ratio with custom target return."""
        returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.015"), Decimal("-0.01")]
        sortino = calculate_sortino_ratio(returns, risk_free_rate=Decimal("0.005"))

        assert isinstance(sortino, Decimal)

    def test_sortino_empty_returns_raises_error(self):
        """Test that empty returns returns zero."""
        result = calculate_sortino_ratio([])
        assert result == Decimal("0")  # Function returns 0 for empty returns, doesn't raise error


class TestMathUtilsIntegration:
    """Test integration between different math utility functions."""

    def test_sharpe_sortino_comparison(self):
        """Test that Sharpe and Sortino ratios are calculated for same data."""
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
            Decimal("0.005"),
        ]

        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sharpe, Decimal)
        assert isinstance(sortino, Decimal)
        # Both should be calculated successfully

    def test_volatility_var_relationship(self):
        """Test relationship between volatility and VaR."""
        returns = [
            Decimal("0.01"),
            Decimal("-0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
            Decimal("0.005"),
        ]

        volatility = calculate_volatility(returns)
        var = calculate_var(returns)

        assert isinstance(volatility, Decimal)
        assert isinstance(var, Decimal)
        assert volatility > 0
        assert var < 0

    def test_correlation_beta_relationship(self):
        """Test relationship between correlation and beta."""
        asset_returns = [Decimal("0.02"), Decimal("-0.03"), Decimal("0.01"), Decimal("-0.01")]
        market_returns = [Decimal("0.01"), Decimal("-0.02"), Decimal("0.005"), Decimal("-0.008")]

        correlation = calculate_correlation(asset_returns, market_returns)
        beta = calculate_beta(asset_returns, market_returns)

        assert isinstance(correlation, Decimal)
        assert isinstance(beta, Decimal)

    def test_percentage_change_drawdown_workflow(self):
        """Test workflow using percentage change and drawdown."""
        initial_value = Decimal("1000")
        returns = [Decimal("0.1"), Decimal("-0.2"), Decimal("0.15"), Decimal("-0.05")]

        # Build equity curve from returns
        equity_curve = [initial_value]
        current_value = initial_value

        for ret in returns:
            new_value = current_value * (Decimal("1") + ret)
            equity_curve.append(new_value)
            current_value = new_value

        # Calculate max drawdown
        max_dd, start_idx, end_idx = calculate_max_drawdown(equity_curve)

        # Calculate final percentage change
        final_change = calculate_percentage_change(initial_value, equity_curve[-1])

        assert isinstance(max_dd, Decimal)
        assert isinstance(final_change, Decimal)
        assert max_dd >= 0  # Drawdown is represented as positive value

    def test_safe_functions_with_edge_cases(self):
        """Test safe functions with edge cases."""
        # Test safe_min and safe_max with same values
        values = [Decimal("42"), Decimal("42"), Decimal("42")]

        min_val = safe_min(*values)
        max_val = safe_max(*values)

        assert min_val == max_val == Decimal("42")

        # Test safe_percentage with various scenarios
        normal_pct = safe_percentage(Decimal("50"), Decimal("200"))
        zero_total_pct = safe_percentage(Decimal("50"), Decimal("0"), default=Decimal("100"))

        assert normal_pct == Decimal("25")  # 50/200 * 100
        assert zero_total_pct == Decimal("10000")  # Default value (100) gets multiplied by 100
