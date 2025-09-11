"""
Data Validation Module

Provides comprehensive data validation capabilities using consolidated utilities.
Consolidated validation logic is now available in src.utils.validation.

Key Components:
- DataValidator: Main validation class using consolidated utilities
- MarketDataValidator: Specific market data validator using consolidated utilities  
- DataValidationPipeline: Pipeline validation orchestration
"""

# Import consolidated validation types for backward compatibility
from src.utils.validation.validation_types import (
    ValidationCategory,
    ValidationIssue,
    ValidationLevel,
)

from .core import DataValidationPipeline
from .data_validator import DataValidator, MarketDataValidationResult
from .market_data_validator import MarketDataValidator
from .validators import PriceValidator, VolumeValidator

__all__ = [
    # Main validators using consolidated utilities
    "DataValidator",
    "MarketDataValidator",
    "DataValidationPipeline",
    # Legacy validators
    "PriceValidator",
    "VolumeValidator",
    # Validation result types
    "MarketDataValidationResult",
    # Validation types (from consolidated utilities)
    "ValidationCategory",
    "ValidationIssue",
    "ValidationLevel",
]
