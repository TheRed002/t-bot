"""Validation module for the T-Bot trading system."""

from .core import ValidationFramework

# Global validator instance - single source of truth
validator = ValidationFramework()

# Export for convenience
validate_order = validator.validate_order
validate_strategy_params = validator.validate_strategy_params
validate_price = validator.validate_price
validate_quantity = validator.validate_quantity
validate_symbol = validator.validate_symbol
validate_exchange_credentials = validator.validate_exchange_credentials
validate_risk_parameters = validator.validate_risk_parameters
validate_timeframe = validator.validate_timeframe
validate_batch = validator.validate_batch

__all__ = [
    'validator',
    'ValidationFramework',
    'validate_order',
    'validate_strategy_params',
    'validate_price',
    'validate_quantity',
    'validate_symbol',
    'validate_exchange_credentials',
    'validate_risk_parameters',
    'validate_timeframe',
    'validate_batch'
]