"""
Data Validation Module

Provides comprehensive data validation capabilities including:
- Market data validation with quality scoring
- Statistical validation and outlier detection
- Temporal validation for data freshness
- Regulatory compliance checks

Key Components:
- DataValidator: Main validation class with comprehensive checks
- ValidationRule: Configurable validation rule system
- QualityScore: Data quality scoring metrics
"""

from .core import DataValidationPipeline
from .data_validator import (
    DataValidator,
    MarketDataValidationResult,
    QualityDimension,
    QualityScore,
    ValidationCategory,
    ValidationIssue,
    ValidationRule,
    ValidationSeverity,
)
from .market_data_validator import MarketDataValidator
from .validators import PriceValidator, VolumeValidator

__all__ = [
    # Main validator
    "DataValidator",
    # Validation result types
    "QualityScore",
    "ValidationRule",
    "ValidationSeverity",
    "ValidationCategory",
    "QualityDimension",
    "ValidationIssue",
    "MarketDataValidationResult",
    # Specific validators
    "DataValidationPipeline",
    "MarketDataValidator",
    "PriceValidator",
    "VolumeValidator",
]
