"""Validation module for the T-Bot trading system.

This module provides both legacy validation framework and modern
service-based validation with dependency injection.

Modern Usage (Recommended):
    ```python
    # In service constructors
    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service


    # Validating data
    result = await self.validation_service.validate_order(order_data)
    if not result.is_valid:
        raise ValidationError(result.get_error_summary())
    ```

Legacy Usage (Backward Compatibility):
    ```python
    from src.utils.validation import validate_order

    validate_order(order_data)  # Raises ValueError if invalid
    ```
"""

from .core import ValidationFramework
from .service import (
    NumericValidationRule,
    StringValidationRule,
    ValidationCache,
    ValidationContext,
    ValidationDetail,
    ValidationResult,
    ValidationRule,
    ValidationService,
    ValidationType,
    ValidatorRegistry,
    get_validation_service,
    shutdown_validation_service,
)

# Global validator instance - single source of truth (backward compatibility)
# Import directly from the consolidated ValidationFramework in core.py
validator = ValidationFramework()

# Legacy convenience exports - use ValidationFramework as single source of truth
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
    "NumericValidationRule",
    "StringValidationRule",
    "ValidationCache",
    "ValidationContext",
    "ValidationDetail",
    "ValidationFramework",
    "ValidationResult",
    "ValidationRule",
    # Modern service-based validation
    "ValidationService",
    "ValidationType",
    "ValidatorRegistry",
    "get_validation_service",
    "shutdown_validation_service",
    "validate_batch",
    "validate_exchange_credentials",
    "validate_order",
    "validate_price",
    "validate_quantity",
    "validate_risk_parameters",
    "validate_strategy_params",
    "validate_symbol",
    "validate_timeframe",
    # Legacy validation framework
    "validator",
]
