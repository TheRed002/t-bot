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


# Get validator instance from DI container for backward compatibility
def _get_validator() -> ValidationFramework:
    """Get validator instance from DI container with lazy initialization."""
    from src.core.dependency_injection import injector

    # Try to resolve, if not found, register via service registry
    try:
        return injector.resolve("ValidationFramework")
    except Exception as e:
        # Use service registry to properly register all services
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(f"Failed to resolve ValidationFramework, using service registry: {e}")

        # Import and call service registry to register all util services
        from ..service_registry import register_util_services

        register_util_services()

        return injector.resolve("ValidationFramework")


# Use lazy property to avoid circular dependencies during module import
_validator_instance = None


def _get_validator_cached() -> ValidationFramework:
    """Get cached validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = _get_validator()
    return _validator_instance


# Legacy convenience exports - use lazy evaluation to avoid circular dependencies
def validate_order(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_order(*args, **kwargs)


def validate_strategy_params(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_strategy_params(*args, **kwargs)


def validate_price(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_price(*args, **kwargs)


def validate_quantity(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_quantity(*args, **kwargs)


def validate_symbol(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_symbol(*args, **kwargs)


def validate_exchange_credentials(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_exchange_credentials(*args, **kwargs)


def validate_risk_parameters(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_risk_parameters(*args, **kwargs)


def validate_timeframe(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_timeframe(*args, **kwargs)


def validate_batch(*args, **kwargs):
    """Legacy validation function."""
    return _get_validator_cached().validate_batch(*args, **kwargs)


# Expose cached validator for other uses
def get_validator() -> ValidationFramework:
    """Get the validation framework instance."""
    return _get_validator_cached()


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
    # Legacy validation framework accessor
    "get_validator",
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
]
