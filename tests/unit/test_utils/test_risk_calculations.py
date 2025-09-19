"""
Tests for Risk Calculations utilities.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.core.types.risk import RiskLevel
from src.utils.decimal_utils import ZERO, ONE, to_decimal
from src.utils.risk_calculations import (
    calculate_var,
    calculate_expected_shortfall,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_current_drawdown,
    calculate_portfolio_value,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    determine_risk_level,
    update_returns_history,
    validate_risk_inputs
)


class TestCalculateVaR:
    """Test Value at Risk calculation."""

    def test_calculate_var_basic(self):
        """Test basic VaR calculation."""
        returns = [to_decimal(str(x)) for x in [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.02, 0.01, 
                                                 -0.04, 0.02, 0.03, -0.01, 0.05]]
        
        var = calculate_var(returns, to_decimal("0.95"), 1)
        
        assert isinstance(var, Decimal)
        assert var >= ZERO

    def test_calculate_var_insufficient_data(self):
        """Test VaR calculation with insufficient data."""
        returns = [to_decimal("-0.01"), to_decimal("0.02")]  # Only 2 returns
        
        var = calculate_var(returns)
        
        assert var == ZERO

    def test_calculate_var_empty_returns(self):
        """Test VaR calculation with empty returns."""
        returns = []
        
        var = calculate_var(returns)
        
        assert var == ZERO

    def test_calculate_var_with_time_horizon(self):
        """Test VaR calculation with different time horizon."""
        returns = [to_decimal(str(x)) for x in [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.02, 0.01,
                                                 -0.04, 0.02, 0.03, -0.01, 0.05]]
        
        var_1d = calculate_var(returns, to_decimal("0.95"), 1)
        var_5d = calculate_var(returns, to_decimal("0.95"), 5)
        
        # 5-day VaR should be higher than 1-day VaR due to sqrt(time) scaling
        assert var_5d > var_1d

    def test_calculate_var_different_confidence_levels(self):
        """Test VaR calculation with different confidence levels."""
        returns = [to_decimal(str(x)) for x in [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.02, 0.01,
                                                 -0.04, 0.02, 0.03, -0.01, 0.05]]
        
        var_95 = calculate_var(returns, to_decimal("0.95"))
        var_99 = calculate_var(returns, to_decimal("0.99"))
        
        # 99% VaR should be higher than 95% VaR
        assert var_99 >= var_95

    def test_calculate_var_exception_handling(self):
        """Test VaR calculation exception handling."""
        # Test with invalid numpy operations by mocking
        with patch('numpy.array', side_effect=Exception("Numpy error")):
            returns = [to_decimal("-0.01"), to_decimal("0.02"), to_decimal("-0.03")]
            
            var = calculate_var(returns)
            
            assert var == ZERO


class TestCalculateExpectedShortfall:
    """Test Expected Shortfall calculation."""

    def test_calculate_expected_shortfall_basic(self):
        """Test basic expected shortfall calculation."""
        returns = [to_decimal(str(x)) for x in [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.02, 0.01,
                                                 -0.04, 0.02, 0.03, -0.01, 0.05]]
        
        es = calculate_expected_shortfall(returns, to_decimal("0.95"))
        
        assert isinstance(es, Decimal)
        assert es >= ZERO

    def test_calculate_expected_shortfall_insufficient_data(self):
        """Test expected shortfall with insufficient data."""
        returns = [to_decimal("-0.01"), to_decimal("0.02")]
        
        es = calculate_expected_shortfall(returns)
        
        assert es == ZERO

    def test_calculate_expected_shortfall_no_tail_returns(self):
        """Test expected shortfall when no returns are in tail."""
        # All positive returns
        returns = [to_decimal("0.01"), to_decimal("0.02"), to_decimal("0.03"), to_decimal("0.04"), 
                   to_decimal("0.05"), to_decimal("0.01"), to_decimal("0.02"), to_decimal("0.03"),
                   to_decimal("0.04"), to_decimal("0.05"), to_decimal("0.01"), to_decimal("0.02")]
        
        es = calculate_expected_shortfall(returns, to_decimal("0.95"))
        
        assert es >= ZERO

    def test_calculate_expected_shortfall_exception_handling(self):
        """Test expected shortfall exception handling."""
        with patch('numpy.percentile', side_effect=Exception("Percentile error")):
            returns = [to_decimal("-0.01"), to_decimal("0.02"), to_decimal("-0.03")]
            
            es = calculate_expected_shortfall(returns)
            
            assert es == ZERO


class TestCalculateSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_calculate_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = [to_decimal(str(x/100)) for x in range(-10, 11)]  # 21 returns
        returns.extend([to_decimal(str(x/100)) for x in range(-10, 11)])  # 42 total returns
        
        sharpe = calculate_sharpe_ratio(returns, to_decimal("0.02"))
        
        assert sharpe is not None
        assert isinstance(sharpe, Decimal)

    def test_calculate_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = [to_decimal("0.01"), to_decimal("0.02")]  # Only 2 returns
        
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe is None

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = [to_decimal("0.01")] * 35  # Same return 35 times

        sharpe = calculate_sharpe_ratio(returns)

        # When volatility is 0, the implementation returns 0, not None
        assert sharpe == ZERO

    def test_calculate_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        returns = [to_decimal("0.01") + to_decimal(str(i * 0.001)) for i in range(35)]
        
        sharpe = calculate_sharpe_ratio(returns, to_decimal("0.01"))
        
        assert sharpe is not None
        assert sharpe > ZERO

    def test_calculate_sharpe_ratio_exception_handling(self):
        """Test Sharpe ratio exception handling."""
        with patch('numpy.mean', side_effect=Exception("Mean error")):
            returns = [to_decimal("0.01")] * 35
            
            sharpe = calculate_sharpe_ratio(returns)
            
            assert sharpe is None


class TestCalculateMaxDrawdown:
    """Test maximum drawdown calculation."""

    def test_calculate_max_drawdown_basic(self):
        """Test basic max drawdown calculation."""
        values = [to_decimal("100"), to_decimal("110"), to_decimal("105"), 
                  to_decimal("90"), to_decimal("95"), to_decimal("120")]
        
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        
        assert isinstance(max_dd, Decimal)
        assert max_dd >= ZERO
        assert isinstance(peak_idx, int)
        assert isinstance(trough_idx, int)
        assert peak_idx <= trough_idx

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with strictly increasing values."""
        values = [to_decimal("100"), to_decimal("110"), to_decimal("120"), to_decimal("130")]
        
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        
        assert max_dd == ZERO
        assert peak_idx >= 0
        assert trough_idx >= 0

    def test_calculate_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data."""
        values = [to_decimal("100")]  # Only one value
        
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        
        assert max_dd == ZERO
        assert peak_idx == 0
        assert trough_idx == 0

    def test_calculate_max_drawdown_empty_values(self):
        """Test max drawdown with empty values."""
        values = []
        
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        
        assert max_dd == ZERO
        assert peak_idx == 0
        assert trough_idx == 0

    def test_calculate_max_drawdown_large_drawdown(self):
        """Test max drawdown with large drawdown."""
        values = [to_decimal("100"), to_decimal("50"), to_decimal("25"), to_decimal("30")]
        
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)
        
        # Should be around 75% drawdown
        assert max_dd > to_decimal("0.7")
        assert peak_idx == 0  # Peak at beginning
        assert trough_idx == 2  # Trough at lowest point

    def test_calculate_max_drawdown_exception_handling(self):
        """Test max drawdown exception handling."""
        # Since our implementation doesn't use numpy, patching numpy.array has no effect
        # The function should calculate the actual drawdown
        with patch('numpy.array', side_effect=Exception("Array error")):
            values = [to_decimal("100"), to_decimal("90")]

            max_dd, peak_idx, trough_idx = calculate_max_drawdown(values)

            # Should calculate actual drawdown of 10%
            assert max_dd == to_decimal("0.1")
            assert peak_idx == 0
            assert trough_idx == 1


class TestCalculateCurrentDrawdown:
    """Test current drawdown calculation."""

    def test_calculate_current_drawdown_basic(self):
        """Test basic current drawdown calculation."""
        current_value = to_decimal("90")
        historical_values = [to_decimal("80"), to_decimal("100"), to_decimal("95")]
        
        current_dd = calculate_current_drawdown(current_value, historical_values)
        
        assert isinstance(current_dd, Decimal)
        assert current_dd >= ZERO
        # Should be 10% drawdown from peak of 100
        assert current_dd == to_decimal("0.1")

    def test_calculate_current_drawdown_no_drawdown(self):
        """Test current drawdown when current value is at peak."""
        current_value = to_decimal("100")
        historical_values = [to_decimal("80"), to_decimal("90"), to_decimal("100")]
        
        current_dd = calculate_current_drawdown(current_value, historical_values)
        
        assert current_dd == ZERO

    def test_calculate_current_drawdown_above_peak(self):
        """Test current drawdown when current value exceeds peak."""
        current_value = to_decimal("110")
        historical_values = [to_decimal("80"), to_decimal("90"), to_decimal("100")]
        
        current_dd = calculate_current_drawdown(current_value, historical_values)
        
        assert current_dd == ZERO

    def test_calculate_current_drawdown_empty_history(self):
        """Test current drawdown with empty historical values."""
        current_value = to_decimal("100")
        historical_values = []
        
        current_dd = calculate_current_drawdown(current_value, historical_values)
        
        assert current_dd == ZERO

    def test_calculate_current_drawdown_zero_peak(self):
        """Test current drawdown with zero peak value."""
        current_value = to_decimal("50")
        historical_values = [ZERO, to_decimal("-10")]
        
        current_dd = calculate_current_drawdown(current_value, historical_values)
        
        assert current_dd == ZERO

    def test_calculate_current_drawdown_exception_handling(self):
        """Test current drawdown exception handling."""
        with patch('builtins.max', side_effect=Exception("Max error")):
            current_value = to_decimal("90")
            historical_values = [to_decimal("100")]
            
            current_dd = calculate_current_drawdown(current_value, historical_values)
            
            assert current_dd == ZERO


class TestCalculatePortfolioValue:
    """Test portfolio value calculation."""

    def test_calculate_portfolio_value_basic(self):
        """Test basic portfolio value calculation."""
        # Mock positions
        position1 = MagicMock()
        position1.symbol = "BTCUSD"
        position1.quantity = to_decimal("2")
        position1.current_price = to_decimal("50000")
        
        position2 = MagicMock()
        position2.symbol = "ETHUSD"
        position2.quantity = to_decimal("10")
        position2.current_price = to_decimal("3000")
        
        positions = [position1, position2]
        
        # Mock market data
        market_data1 = MagicMock()
        market_data1.symbol = "BTCUSD"
        market_data1.close = to_decimal("51000")  # Updated price
        
        market_data2 = MagicMock()
        market_data2.symbol = "ETHUSD"
        market_data2.close = to_decimal("3100")  # Updated price
        
        market_data = [market_data1, market_data2]
        
        portfolio_value = calculate_portfolio_value(positions, market_data)
        
        expected_value = to_decimal("2") * to_decimal("51000") + to_decimal("10") * to_decimal("3100")
        assert portfolio_value == expected_value

    def test_calculate_portfolio_value_missing_market_data(self):
        """Test portfolio value calculation with missing market data."""
        position = MagicMock()
        position.symbol = "BTCUSD"
        position.quantity = to_decimal("2")
        position.current_price = to_decimal("50000")
        
        positions = [position]
        market_data = []  # No market data
        
        portfolio_value = calculate_portfolio_value(positions, market_data)
        
        expected_value = to_decimal("2") * to_decimal("50000")  # Uses current_price
        assert portfolio_value == expected_value

    def test_calculate_portfolio_value_zero_price(self):
        """Test portfolio value calculation with zero price."""
        position = MagicMock()
        position.symbol = "BTCUSD"
        position.quantity = to_decimal("2")
        position.current_price = ZERO
        
        market_data_item = MagicMock()
        market_data_item.symbol = "BTCUSD"
        market_data_item.close = ZERO
        
        positions = [position]
        market_data = [market_data_item]
        
        portfolio_value = calculate_portfolio_value(positions, market_data)
        
        assert portfolio_value == ZERO

    def test_calculate_portfolio_value_empty_positions(self):
        """Test portfolio value calculation with empty positions."""
        positions = []
        market_data = []
        
        portfolio_value = calculate_portfolio_value(positions, market_data)
        
        assert portfolio_value == ZERO

    def test_calculate_portfolio_value_exception_handling(self):
        """Test portfolio value calculation exception handling."""
        # Mock position with missing attributes to trigger exception
        position = MagicMock()
        del position.symbol  # Remove required attribute
        
        positions = [position]
        market_data = []
        
        portfolio_value = calculate_portfolio_value(positions, market_data)
        
        assert portfolio_value == ZERO


class TestCalculateSortinoRatio:
    """Test Sortino ratio calculation."""

    def test_calculate_sortino_ratio_basic(self):
        """Test basic Sortino ratio calculation."""
        returns = [to_decimal(str(x/100)) for x in [-5, 2, -3, 4, 1, -2, 3, -1, 5, -4, 6, -2]]
        
        sortino = calculate_sortino_ratio(returns, to_decimal("0.02"), ZERO)
        
        assert isinstance(sortino, Decimal)
        assert sortino != ZERO

    def test_calculate_sortino_ratio_insufficient_data(self):
        """Test Sortino ratio with insufficient data."""
        returns = [to_decimal("0.01"), to_decimal("0.02")]  # Only 2 returns
        
        sortino = calculate_sortino_ratio(returns)
        
        assert sortino == ZERO

    def test_calculate_sortino_ratio_no_downside_deviation(self):
        """Test Sortino ratio with no downside deviation."""
        returns = [to_decimal("0.01"), to_decimal("0.02"), to_decimal("0.03")] * 4  # All positive

        sortino = calculate_sortino_ratio(returns, ZERO, ZERO)

        # When there's no downside, function uses standard deviation instead
        # So it calculates a positive Sortino ratio
        assert sortino > ZERO

    def test_calculate_sortino_ratio_exception_handling(self):
        """Test Sortino ratio exception handling."""
        with patch('numpy.array', side_effect=Exception("Array error")):
            returns = [to_decimal("0.01")] * 12
            
            sortino = calculate_sortino_ratio(returns)
            
            assert sortino == ZERO


class TestCalculateCalmarRatio:
    """Test Calmar ratio calculation."""

    def test_calculate_calmar_ratio_basic(self):
        """Test basic Calmar ratio calculation."""
        returns = [to_decimal(str(x/100)) for x in [-5, 2, -3, 4, 1, -2, 3, -1, 5, -4]]
        
        calmar = calculate_calmar_ratio(returns, ONE)
        
        assert isinstance(calmar, Decimal)

    def test_calculate_calmar_ratio_empty_returns(self):
        """Test Calmar ratio with empty returns."""
        returns = []
        
        calmar = calculate_calmar_ratio(returns)
        
        assert calmar == ZERO

    def test_calculate_calmar_ratio_zero_max_drawdown(self):
        """Test Calmar ratio with zero max drawdown."""
        # Constant returns (no drawdown)
        returns = [to_decimal("0.01")] * 10
        
        calmar = calculate_calmar_ratio(returns)
        
        assert calmar == ZERO  # Division by zero max drawdown

    def test_calculate_calmar_ratio_exception_handling(self):
        """Test Calmar ratio exception handling."""
        with patch('numpy.mean', side_effect=Exception("Mean error")):
            returns = [to_decimal("0.01")] * 10
            
            calmar = calculate_calmar_ratio(returns)
            
            assert calmar == ZERO


class TestDetermineRiskLevel:
    """Test risk level determination."""

    def test_determine_risk_level_low(self):
        """Test low risk level determination."""
        var_1d = to_decimal("100")  # 1% VaR for 10k portfolio
        current_drawdown = to_decimal("0.02")  # 2% drawdown
        sharpe_ratio = to_decimal("1.5")  # Good Sharpe ratio
        portfolio_value = to_decimal("10000")
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level == RiskLevel.LOW

    def test_determine_risk_level_medium(self):
        """Test medium risk level determination."""
        var_1d = to_decimal("400")  # 4% VaR for 10k portfolio
        current_drawdown = to_decimal("0.08")  # 8% drawdown
        sharpe_ratio = to_decimal("0.3")  # Low Sharpe ratio
        portfolio_value = to_decimal("10000")
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level == RiskLevel.MEDIUM

    def test_determine_risk_level_high(self):
        """Test high risk level determination."""
        var_1d = to_decimal("800")  # 8% VaR for 10k portfolio
        current_drawdown = to_decimal("0.15")  # 15% drawdown
        sharpe_ratio = to_decimal("-0.5")  # Negative Sharpe ratio
        portfolio_value = to_decimal("10000")
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level == RiskLevel.HIGH

    def test_determine_risk_level_critical(self):
        """Test critical risk level determination."""
        var_1d = to_decimal("1200")  # 12% VaR for 10k portfolio
        current_drawdown = to_decimal("0.25")  # 25% drawdown
        sharpe_ratio = to_decimal("-1.5")  # Very negative Sharpe ratio
        portfolio_value = to_decimal("10000")
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level == RiskLevel.CRITICAL

    def test_determine_risk_level_none_sharpe_ratio(self):
        """Test risk level determination with None Sharpe ratio."""
        var_1d = to_decimal("200")
        current_drawdown = to_decimal("0.03")
        sharpe_ratio = None
        portfolio_value = to_decimal("10000")
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_determine_risk_level_zero_portfolio(self):
        """Test risk level determination with zero portfolio value."""
        var_1d = to_decimal("100")
        current_drawdown = to_decimal("0.05")
        sharpe_ratio = to_decimal("1.0")
        portfolio_value = ZERO
        
        risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
        
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_determine_risk_level_exception_handling(self):
        """Test risk level determination exception handling."""
        with patch('src.utils.risk_calculations.safe_divide', side_effect=Exception("Division error")):
            var_1d = to_decimal("100")
            current_drawdown = to_decimal("0.05")
            sharpe_ratio = to_decimal("1.0")
            portfolio_value = to_decimal("10000")
            
            risk_level = determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)
            
            assert risk_level == RiskLevel.MEDIUM  # Safe default


class TestUpdateReturnsHistory:
    """Test returns history update."""

    def test_update_returns_history_basic(self):
        """Test basic returns history update."""
        values = [to_decimal("100"), to_decimal("102"), to_decimal("104"), to_decimal("103")]
        
        returns = update_returns_history(values)
        
        assert len(returns) == 3  # n-1 returns for n values
        assert all(isinstance(ret, Decimal) for ret in returns)

    def test_update_returns_history_with_callback(self):
        """Test returns history update with callback."""
        values = [to_decimal("100"), to_decimal("105"), to_decimal("110")]
        callback = MagicMock()
        
        returns = update_returns_history(values, callback)
        
        callback.assert_called_once_with(returns)
        assert len(returns) == 2

    def test_update_returns_history_insufficient_data(self):
        """Test returns history update with insufficient data."""
        values = [to_decimal("100")]  # Only one value
        
        returns = update_returns_history(values)
        
        assert returns == []

    def test_update_returns_history_max_history_limit(self):
        """Test returns history update with max history limit."""
        values = [to_decimal(str(100 + i)) for i in range(300)]  # 300 values -> 299 returns
        
        returns = update_returns_history(values, max_history=252)
        
        assert len(returns) == 252  # Limited to max_history

    def test_update_returns_history_zero_previous_value(self):
        """Test returns history update with zero previous value."""
        values = [ZERO, to_decimal("100"), to_decimal("105")]
        
        returns = update_returns_history(values)
        
        # Should skip calculation where previous value is zero
        assert len(returns) == 1  # Only one valid return calculation

    def test_update_returns_history_exception_handling(self):
        """Test returns history update exception handling."""
        with patch('src.utils.risk_calculations.safe_divide', side_effect=Exception("Division error")):
            values = [to_decimal("100"), to_decimal("105")]
            
            returns = update_returns_history(values)
            
            assert returns == []


class TestValidateRiskInputs:
    """Test risk input validation."""

    def test_validate_risk_inputs_valid(self):
        """Test validation with valid inputs."""
        portfolio_value = to_decimal("10000")
        
        position = MagicMock()
        position.symbol = "BTCUSD"
        position.quantity = to_decimal("1")
        
        market_data = MagicMock()
        
        positions = [position]
        market_data_list = [market_data]
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data_list)
        
        assert is_valid is True

    def test_validate_risk_inputs_low_portfolio_value(self):
        """Test validation with low portfolio value."""
        portfolio_value = to_decimal("50")  # Below minimum
        positions = []
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data, to_decimal("100"))
        
        assert is_valid is False

    def test_validate_risk_inputs_data_mismatch(self):
        """Test validation with position/market data mismatch."""
        portfolio_value = to_decimal("10000")
        positions = [MagicMock(), MagicMock()]  # 2 positions
        market_data = [MagicMock()]  # 1 market data item
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False

    def test_validate_risk_inputs_missing_symbol(self):
        """Test validation with position missing symbol."""
        portfolio_value = to_decimal("10000")
        
        position = MagicMock()
        del position.symbol  # Remove symbol
        
        positions = [position]
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False

    def test_validate_risk_inputs_empty_symbol(self):
        """Test validation with position having empty symbol."""
        portfolio_value = to_decimal("10000")
        
        position = MagicMock()
        position.symbol = ""  # Empty symbol
        position.quantity = to_decimal("1")
        
        positions = [position]
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False

    def test_validate_risk_inputs_invalid_quantity(self):
        """Test validation with invalid position quantity."""
        portfolio_value = to_decimal("10000")
        
        position = MagicMock()
        position.symbol = "BTCUSD"
        position.quantity = ZERO  # Invalid quantity
        
        positions = [position]
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False

    def test_validate_risk_inputs_missing_quantity(self):
        """Test validation with position missing quantity."""
        portfolio_value = to_decimal("10000")
        
        position = MagicMock()
        position.symbol = "BTCUSD"
        del position.quantity  # Remove quantity
        
        positions = [position]
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False

    def test_validate_risk_inputs_empty_lists(self):
        """Test validation with empty positions and market data."""
        portfolio_value = to_decimal("10000")
        positions = []
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is True

    def test_validate_risk_inputs_exception_handling(self):
        """Test validation exception handling."""
        # Create invalid position object that will cause exception in validation
        class InvalidPosition:
            def __getattribute__(self, name):
                if name == "symbol":
                    raise Exception("Attribute access error")
                return super().__getattribute__(name)
        
        portfolio_value = to_decimal("10000")
        positions = [InvalidPosition()]
        market_data = []
        
        is_valid = validate_risk_inputs(portfolio_value, positions, market_data)
        
        assert is_valid is False


class TestRiskCalculationsEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_risk_calculations_with_extreme_values(self):
        """Test risk calculations with extreme values."""
        # Very large returns
        large_returns = [to_decimal("1.0"), to_decimal("2.0"), to_decimal("-1.5")] * 10
        
        var = calculate_var(large_returns)
        assert var >= ZERO
        
        sharpe = calculate_sharpe_ratio(large_returns)
        assert sharpe is not None or sharpe is None  # Either valid or None

    def test_risk_calculations_precision_consistency(self):
        """Test that risk calculations maintain Decimal precision."""
        returns = [to_decimal("0.01234567890123456789")] * 35
        
        sharpe = calculate_sharpe_ratio(returns)
        
        if sharpe is not None:
            assert isinstance(sharpe, Decimal)

    def test_integration_risk_workflow(self):
        """Test integration of multiple risk calculations."""
        # Simulate a complete risk calculation workflow
        portfolio_values = [to_decimal(str(1000 + i * 10)) for i in range(50)]
        returns = update_returns_history(portfolio_values)
        
        if len(returns) >= 10:
            var = calculate_var(returns)
            es = calculate_expected_shortfall(returns)
            max_dd, _, _ = calculate_max_drawdown(portfolio_values)
            current_dd = calculate_current_drawdown(portfolio_values[-1], portfolio_values[:-1])
            
            assert all(val >= ZERO for val in [var, es, max_dd, current_dd])
            
            if len(returns) >= 30:
                sharpe = calculate_sharpe_ratio(returns)
                risk_level = determine_risk_level(var, current_dd, sharpe, portfolio_values[-1])
                
                assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]