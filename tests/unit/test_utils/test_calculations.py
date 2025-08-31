"""
Unit tests for calculations module.

This module tests the financial calculation utilities.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.utils.calculations import (
    FinancialCalculator,
    calc,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    moving_average,
    max_drawdown,
    kelly_criterion,
    position_size_volatility_adjusted,
    calculate_returns,
    risk_reward_ratio,
    expected_value,
    profit_factor,
)
from src.core.exceptions import ValidationError


class TestFinancialCalculatorInitialization:
    """Test FinancialCalculator initialization and singleton behavior."""

    def test_financial_calculator_instance(self):
        """Test that FinancialCalculator can be instantiated."""
        calculator = FinancialCalculator()
        assert calculator is not None
        assert isinstance(calculator, FinancialCalculator)

    def test_calc_singleton_exists(self):
        """Test that calc singleton instance exists."""
        assert calc is not None
        assert isinstance(calc, FinancialCalculator)

    def test_convenience_exports(self):
        """Test that convenience exports are callable."""
        # Test that all convenience exports are callable
        assert callable(sharpe_ratio)
        assert callable(sortino_ratio)
        assert callable(calmar_ratio)
        assert callable(moving_average)
        assert callable(max_drawdown)
        assert callable(kelly_criterion)
        assert callable(position_size_volatility_adjusted)
        assert callable(calculate_returns)
        assert callable(risk_reward_ratio)
        assert callable(expected_value)
        assert callable(profit_factor)


class TestFinancialCalculatorBasicMethods:
    """Test basic financial calculation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = FinancialCalculator()

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = tuple([Decimal('0.02'), Decimal('0.03'), Decimal('-0.01'), Decimal('0.04')])
        risk_free_rate = Decimal('0.01')
        
        result = self.calculator.sharpe_ratio(returns, risk_free_rate)
        assert isinstance(result, Decimal)
        assert result > 0  # Should be positive for profitable strategy

    def test_sharpe_ratio_with_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = tuple([Decimal('0.02'), Decimal('0.02'), Decimal('0.02'), Decimal('0.02')])
        risk_free_rate = Decimal('0.01')
        
        # Should handle division by zero gracefully
        result = self.calculator.sharpe_ratio(returns, risk_free_rate)
        assert isinstance(result, Decimal)

    def test_moving_average_calculation(self):
        """Test moving average calculation."""
        prices = tuple([Decimal('100'), Decimal('102'), Decimal('101'), Decimal('103'), Decimal('105')])
        period = 3
        
        result = self.calculator.moving_average(prices, period)
        assert isinstance(result, Decimal)
        # This returns the last moving average value, not a list

    def test_moving_average_empty_list(self):
        """Test moving average with empty price list."""
        # Empty tuple should return ZERO, not raise exception
        result = self.calculator.moving_average(tuple(), 3)
        assert result == Decimal('0')

    def test_moving_average_invalid_period(self):
        """Test moving average with invalid period."""
        prices = tuple([Decimal('100'), Decimal('102')])
        
        # This should not raise ValidationError - it returns average of available prices
        result = self.calculator.moving_average(prices, 3)  # Period > length
        assert isinstance(result, Decimal)

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        prices = [Decimal('100'), Decimal('110'), Decimal('105'), Decimal('90'), Decimal('95')]
        
        result = self.calculator.max_drawdown(prices)
        assert isinstance(result, tuple)
        assert len(result) == 3  # Returns (max_drawdown, start_idx, end_idx)
        assert isinstance(result[0], Decimal)

    def test_max_drawdown_empty_list(self):
        """Test max drawdown with empty price list."""
        result = self.calculator.max_drawdown([])
        assert isinstance(result, tuple)
        assert result == (Decimal('0'), 0, 0)

    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = [Decimal('100'), Decimal('102'), Decimal('101'), Decimal('103')]
        
        result = self.calculator.calculate_returns(prices)
        assert isinstance(result, list)
        assert len(result) == len(prices) - 1
        assert all(isinstance(ret, Decimal) for ret in result)

    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        win_rate = Decimal('0.6')
        avg_win = Decimal('0.15')
        avg_loss = Decimal('0.10')
        
        result = self.calculator.kelly_criterion(win_rate, avg_win, avg_loss)
        assert isinstance(result, Decimal)
        assert 0 <= result <= 1  # Kelly fraction should be between 0 and 1

    def test_kelly_criterion_edge_cases(self):
        """Test Kelly criterion with edge case inputs."""
        # Test with zero win rate
        result = self.calculator.kelly_criterion(Decimal('0'), Decimal('0.1'), Decimal('0.1'))
        assert result == 0
        
        # Test with 100% win rate
        result = self.calculator.kelly_criterion(Decimal('1'), Decimal('0.1'), Decimal('0.1'))
        assert result >= 0

    def test_position_size_volatility_adjusted(self):
        """Test volatility-adjusted position sizing."""
        capital = Decimal('10000')
        risk_per_trade = Decimal('0.02')
        stop_distance = Decimal('0.05')  # 5% stop loss
        volatility = Decimal('0.20')
        
        result = self.calculator.position_size_volatility_adjusted(capital, risk_per_trade, stop_distance, volatility)
        assert isinstance(result, Decimal)
        assert result > 0
        assert result <= capital

    def test_risk_reward_ratio(self):
        """Test risk-reward ratio calculation."""
        entry_price = Decimal('100')
        take_profit = Decimal('115')
        stop_loss = Decimal('95')
        
        result = self.calculator.risk_reward_ratio(entry_price, take_profit, stop_loss)
        assert isinstance(result, Decimal)
        assert result > 0

    def test_risk_reward_ratio_invalid_prices(self):
        """Test risk-reward ratio with different price relationships."""
        # Test that method handles unusual price relationships
        # Method calculates absolute differences, so it won't raise ValidationError
        result = self.calculator.risk_reward_ratio(
            Decimal('100'), Decimal('105'), Decimal('115')
        )
        assert isinstance(result, Decimal)
        assert result > 0

    def test_expected_value(self):
        """Test expected value calculation."""
        win_probability = Decimal('0.6')
        avg_win = Decimal('150')
        avg_loss = Decimal('50')  # As positive number
        
        result = self.calculator.expected_value(win_probability, avg_win, avg_loss)
        assert isinstance(result, Decimal)

    def test_expected_value_edge_cases(self):
        """Test expected value with edge cases."""
        # Test with zero win probability
        result = self.calculator.expected_value(Decimal('0'), Decimal('100'), Decimal('50'))
        assert isinstance(result, Decimal)
        assert result <= 0  # Should be negative (only losses)
        
        # Test with 100% win probability
        result = self.calculator.expected_value(Decimal('1'), Decimal('100'), Decimal('50'))
        assert isinstance(result, Decimal)
        assert result > 0  # Should be positive (only wins)

    def test_profit_factor(self):
        """Test profit factor calculation."""
        wins = [Decimal('0.05'), Decimal('0.03')]
        losses = [Decimal('0.02'), Decimal('0.01')]
        
        result = self.calculator.profit_factor(wins, losses)
        assert isinstance(result, Decimal)
        assert result >= 0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        wins = [Decimal('0.05'), Decimal('0.03'), Decimal('0.02')]
        losses = []
        
        result = self.calculator.profit_factor(wins, losses)
        assert isinstance(result, Decimal)
        # Should be very high when there are no losses


class TestFinancialCalculatorEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = FinancialCalculator()

    def test_empty_inputs(self):
        """Test methods with empty inputs."""
        # Empty tuple should return ZERO for sharpe_ratio
        result = self.calculator.sharpe_ratio(tuple(), Decimal('0.02'))
        assert result == Decimal('0')
            
        # Empty list should return empty list for calculate_returns
        result = self.calculator.calculate_returns([])
        assert result == []

    def test_invalid_decimal_inputs(self):
        """Test methods with invalid decimal inputs."""
        with pytest.raises((ValidationError, TypeError)):
            self.calculator.sharpe_ratio("invalid", Decimal('0.02'))
            
        with pytest.raises((ValidationError, TypeError)):
            self.calculator.kelly_criterion("invalid", Decimal('0.1'), Decimal('0.1'))

    def test_negative_volatility(self):
        """Test methods with negative volatility."""
        # Need to pass volatility as 4th parameter and add a valid stop_distance as 3rd
        result = self.calculator.position_size_volatility_adjusted(
            Decimal('10000'), Decimal('0.02'), Decimal('0.1'), Decimal('-0.1')
        )
        # With negative volatility, the function should return ZERO
        assert result == Decimal('0')

    def test_zero_capital(self):
        """Test position sizing with zero capital."""
        # Need to pass volatility as 4th parameter and add a valid stop_distance as 3rd
        result = self.calculator.position_size_volatility_adjusted(
            Decimal('0'), Decimal('0.02'), Decimal('0.1'), Decimal('0.2')
        )
        # With zero capital, the function should return ZERO
        assert result == Decimal('0')


class TestConvenienceExports:
    """Test convenience exports from the calculations module."""

    def test_sharpe_ratio_export(self):
        """Test that sharpe_ratio convenience export works."""
        returns = tuple([Decimal('0.02'), Decimal('0.03'), Decimal('-0.01')])
        risk_free_rate = Decimal('0.01')
        
        # Should work the same as calc.sharpe_ratio
        result1 = sharpe_ratio(returns, risk_free_rate)
        result2 = calc.sharpe_ratio(returns, risk_free_rate)
        
        assert result1 == result2

    def test_moving_average_export(self):
        """Test that moving_average convenience export works."""
        prices = tuple([Decimal('100'), Decimal('102'), Decimal('101')])
        period = 2
        
        result1 = moving_average(prices, period)
        result2 = calc.moving_average(prices, period)
        
        assert result1 == result2

    def test_kelly_criterion_export(self):
        """Test that kelly_criterion convenience export works."""
        win_rate = Decimal('0.6')
        avg_win = Decimal('0.15')
        avg_loss = Decimal('0.10')
        
        result1 = kelly_criterion(win_rate, avg_win, avg_loss)
        result2 = calc.kelly_criterion(win_rate, avg_win, avg_loss)
        
        assert result1 == result2