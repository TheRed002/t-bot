"""Calculation utilities for the T-Bot trading system."""

from .financial import FinancialCalculator, calc

# Export singleton instance and class
__all__ = ['FinancialCalculator', 'calc']

# Convenience exports for common calculations
sharpe_ratio = calc.sharpe_ratio
sortino_ratio = calc.sortino_ratio
calmar_ratio = calc.calmar_ratio
moving_average = calc.moving_average
max_drawdown = calc.max_drawdown
kelly_criterion = calc.kelly_criterion
position_size_volatility_adjusted = calc.position_size_volatility_adjusted
calculate_returns = calc.calculate_returns
risk_reward_ratio = calc.risk_reward_ratio
expected_value = calc.expected_value
profit_factor = calc.profit_factor