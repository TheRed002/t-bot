"""
Data Flow Integrity Module for T-Bot Trading System.

This module provides comprehensive solutions to prevent precision loss cascades
and ensure data integrity throughout the entire data flow from monitoring to
utils modules and beyond.

Key Features:
- Decimal precision preservation across module boundaries
- Type consistency validation
- Null handling standardization
- Financial range validation with adaptive bounds
- Precision loss detection and reporting

Critical Fix Areas:
1. Financial Precision Loss: Prevents conversion chain from Decimal to float
2. Parameter Validation Gaps: Comprehensive input validation
3. Type Conversion Inconsistencies: Standardized Decimal/float/int handling
4. Hard-coded Validation Ranges: Dynamic limits for market conditions
5. Null Handling Problems: Consistent patterns for None values
"""

import warnings
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.monitoring.financial_precision import FinancialPrecisionWarning, safe_decimal_to_float
from src.utils.validators import (
    validate_financial_range,
    validate_null_handling,
    validate_type_conversion,
)

logger = get_logger(__name__)


class DataFlowIntegrityError(Exception):
    """Raised when data flow integrity is compromised."""

    pass


class PrecisionTracker:
    """
    Tracks precision loss across data transformations.

    This class monitors Decimal-to-float conversions and maintains
    a record of precision loss events for audit and optimization purposes.
    """

    def __init__(self) -> None:
        self.precision_events: list[dict[str, Any]] = []
        self.warning_count: int = 0
        self.error_count: int = 0

    def track_conversion(
        self, original: Decimal, converted: float, context: str, precision_loss: bool = False
    ) -> None:
        """
        Track a Decimal-to-float conversion event.

        Args:
            original: Original Decimal value
            converted: Converted float value
            context: Context/location of conversion
            precision_loss: Whether precision loss was detected
        """
        event = {
            "original": str(original),
            "converted": converted,
            "context": context,
            "precision_loss": precision_loss,
            "relative_error": self._calculate_relative_error(original, converted),
        }

        self.precision_events.append(event)

        if precision_loss:
            self.warning_count += 1

        # Log significant precision loss
        relative_error = event["relative_error"]
        if isinstance(relative_error, (int, float)) and relative_error > 0.0001:  # > 0.01%
            logger.warning(
                f"Significant precision loss in {context}: "
                f"{original} -> {converted} (error: {event['relative_error']:.6%})"
            )

    def _calculate_relative_error(self, original: Decimal, converted: float) -> float:
        """Calculate relative error between original and converted values."""
        if original == 0:
            return 0.0

        try:
            converted_decimal = Decimal(str(converted))
            difference = abs(original - converted_decimal)
            return float(difference / abs(original))
        except (InvalidOperation, ZeroDivisionError):
            return 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get summary of precision tracking."""
        return {
            "total_conversions": len(self.precision_events),
            "precision_warnings": self.warning_count,
            "error_count": self.error_count,
            "avg_relative_error": self._get_avg_relative_error(),
        }

    def _get_avg_relative_error(self) -> float:
        """Calculate average relative error across all conversions."""
        if not self.precision_events:
            return 0.0

        total_error = sum(event["relative_error"] for event in self.precision_events)
        return total_error / len(self.precision_events)


def get_precision_tracker() -> PrecisionTracker:
    """Get the global precision tracker instance from DI container with lazy initialization."""
    from src.core.dependency_injection import injector

    try:
        return injector.resolve("PrecisionTracker")
    except Exception:
        # Register if not found
        injector.register_service("PrecisionTracker", PrecisionTracker(), singleton=True)
        return injector.resolve("PrecisionTracker")


# PrecisionTracker registration is handled by service_registry.py


class DataFlowValidator:
    """
    Comprehensive data flow validation system.

    Provides end-to-end validation for data flowing between modules,
    ensuring type consistency, precision preservation, and range validation.
    """

    def __init__(self) -> None:
        self.validation_rules: dict[str, dict[str, Any]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default validation rules for common data types."""
        self.validation_rules = {
            "price": {
                "type": Decimal,
                "min_value": Decimal("0.00000001"),
                "max_value": Decimal("10000000"),
                "precision_digits": 8,
                "allow_null": False,
            },
            "volume": {
                "type": Decimal,
                "min_value": Decimal("0"),
                "max_value": Decimal("1000000000"),
                "precision_digits": 8,
                "allow_null": False,
            },
            "pnl": {
                "type": Decimal,
                "min_value": Decimal("-1000000"),
                "max_value": Decimal("1000000"),
                "precision_digits": 8,
                "allow_null": False,
            },
            "percentage": {
                "type": Decimal,
                "min_value": Decimal("0"),
                "max_value": Decimal("100"),
                "precision_digits": 4,
                "allow_null": False,
            },
            "ttl": {"type": int, "min_value": 1, "max_value": 86400, "allow_null": False},
        }

    def validate_data_flow(self, data: dict[str, Any], context: str = "unknown") -> dict[str, Any]:
        """
        Validate data flow ensuring type consistency and precision.

        Args:
            data: Data dictionary to validate
            context: Context for error reporting

        Returns:
            Dict: Validated and normalized data

        Raises:
            DataFlowIntegrityError: If validation fails
        """
        validated_data = {}

        for field_name, value in data.items():
            try:
                validated_value = self._validate_field(field_name, value, context)
                validated_data[field_name] = validated_value
            except Exception as e:
                raise DataFlowIntegrityError(
                    f"Data flow validation failed for {field_name} in {context}: {e}"
                ) from e

        return validated_data

    def _validate_field(self, field_name: str, value: Any, context: str) -> Any:
        """Validate individual field based on rules."""
        # Find applicable rule
        rule = self._get_rule_for_field(field_name)
        if not rule:
            # No specific rule, apply basic validation
            return validate_null_handling(value, allow_null=True, field_name=field_name)

        # Apply null handling
        validated_value = validate_null_handling(
            value, allow_null=rule.get("allow_null", True), field_name=field_name
        )

        if validated_value is None:
            return None

        # Type conversion
        target_type = rule.get("type")
        if target_type:
            validated_value = validate_type_conversion(
                validated_value, target_type, field_name, strict=False
            )

        # Range validation for numerical types
        if isinstance(validated_value, (Decimal, int, float)):
            min_val = rule.get("min_value")
            max_val = rule.get("max_value")

            if min_val is not None or max_val is not None:
                if isinstance(validated_value, Decimal):
                    validate_financial_range(validated_value, min_val, max_val, field_name)
                elif min_val is not None and validated_value < min_val:
                    raise ValidationError(
                        f"{field_name} below minimum: {validated_value} < {min_val}"
                    )
                elif max_val is not None and validated_value > max_val:
                    raise ValidationError(
                        f"{field_name} above maximum: {validated_value} > {max_val}"
                    )

        return validated_value

    def _get_rule_for_field(self, field_name: str) -> dict[str, Any] | None:
        """Get validation rule for a field name."""
        field_lower = field_name.lower()

        # Exact match
        if field_lower in self.validation_rules:
            return self.validation_rules[field_lower]

        # Partial match
        for rule_name, rule in self.validation_rules.items():
            if rule_name in field_lower or field_lower.endswith(rule_name):
                return rule

        return None

    def add_validation_rule(self, field_pattern: str, rule: dict[str, Any]) -> None:
        """Add custom validation rule."""
        self.validation_rules[field_pattern] = rule


class IntegrityPreservingConverter:
    """
    Converter that preserves data integrity across module boundaries.

    This class provides safe conversion methods that maintain precision
    and provide fallback strategies when precision loss is unavoidable.
    """

    def __init__(
        self, track_precision: bool = True, precision_tracker: PrecisionTracker | None = None
    ):
        self.track_precision = track_precision
        if precision_tracker is not None:
            self.tracker = precision_tracker
        elif track_precision:
            self.tracker = get_precision_tracker()
        else:
            self.tracker = None

    def safe_convert_for_metrics(
        self,
        value: Decimal | float | int,
        metric_name: str,
        precision_digits: int = 8,
        fallback_strategy: str = "warn",
    ) -> float:
        """
        Safely convert financial values for Prometheus metrics.

        Args:
            value: Value to convert
            metric_name: Metric name for context
            precision_digits: Decimal places to preserve
            fallback_strategy: Strategy when precision loss occurs ("warn", "error", "silent")

        Returns:
            float: Converted value suitable for metrics

        Raises:
            DataFlowIntegrityError: If fallback_strategy is "error" and precision loss occurs
        """
        try:
            # Use the existing safe_decimal_to_float with tracking
            result = safe_decimal_to_float(value, metric_name, precision_digits, warn_on_loss=True)

            # Track the conversion if enabled
            if self.tracker and isinstance(value, Decimal):
                self.tracker.track_conversion(
                    original=value,
                    converted=result,
                    context=f"metrics.{metric_name}",
                    precision_loss=False,  # Let the precision function determine this
                )

            return result

        except Exception as e:
            if fallback_strategy == "error":
                raise DataFlowIntegrityError(f"Failed to convert {metric_name}: {e}") from e
            elif fallback_strategy == "warn":
                warnings.warn(
                    f"Conversion failed for {metric_name}: {e}", FinancialPrecisionWarning
                )
                return 0.0
            else:  # silent
                return 0.0

    def batch_convert_with_integrity(
        self,
        data: dict[str, Decimal | float | int],
        precision_map: dict[str, int] | None = None,
        context: str = "batch_conversion",
    ) -> dict[str, float]:
        """
        Convert a batch of values while maintaining integrity tracking.

        Args:
            data: Dictionary of field_name -> value
            precision_map: Optional precision requirements per field
            context: Context for tracking

        Returns:
            Dict: Converted values as floats
        """
        precision_map = precision_map or {}
        results = {}

        for field_name, value in data.items():
            precision = precision_map.get(field_name, 8)
            results[field_name] = self.safe_convert_for_metrics(
                value, f"{context}.{field_name}", precision
            )

        return results


def get_data_flow_validator() -> DataFlowValidator:
    """Get the global data flow validator from DI container with lazy initialization."""
    from src.core.dependency_injection import injector

    try:
        return injector.resolve("DataFlowValidator")
    except Exception:
        # Register if not found
        injector.register_service("DataFlowValidator", DataFlowValidator(), singleton=True)
        return injector.resolve("DataFlowValidator")


def get_integrity_converter() -> IntegrityPreservingConverter:
    """Get the global integrity-preserving converter from DI container with lazy initialization."""
    from src.core.dependency_injection import injector

    try:
        return injector.resolve("IntegrityPreservingConverter")
    except Exception:
        # Register if not found
        injector.register_service(
            "IntegrityPreservingConverter", IntegrityPreservingConverter(), singleton=True
        )
        return injector.resolve("IntegrityPreservingConverter")


# Data flow services registration is handled by service_registry.py


def validate_cross_module_data(
    data: dict[str, Any], source_module: str, target_module: str, operation: str = "transfer"
) -> dict[str, Any]:
    """
    Validate data being transferred between modules.

    This is the main entry point for ensuring data integrity
    when data flows between monitoring and utils modules.

    Args:
        data: Data being transferred
        source_module: Source module name
        target_module: Target module name
        operation: Type of operation

    Returns:
        Dict: Validated data

    Raises:
        DataFlowIntegrityError: If validation fails
    """
    context = f"{source_module}->{target_module}:{operation}"

    try:
        # Use the global validator
        validated_data = _data_flow_validator.validate_data_flow(data, context)

        logger.debug(
            f"Data flow validation successful: {context}",
            extra={
                "source_module": source_module,
                "target_module": target_module,
                "operation": operation,
                "field_count": len(data),
            },
        )

        return validated_data

    except Exception as e:
        logger.error(
            f"Data flow validation failed: {context} - {e}",
            extra={
                "source_module": source_module,
                "target_module": target_module,
                "operation": operation,
                "error": str(e),
            },
        )
        raise


def fix_precision_cascade(
    data: dict[str, Any], target_formats: dict[str, str] | None = None
) -> dict[str, Any]:
    """
    Fix precision loss cascade by ensuring proper type handling.

    This function addresses the core issue of precision loss by:
    1. Converting all float financial values to Decimals
    2. Validating ranges before any conversions
    3. Using proper precision-preserving conversions when needed

    Args:
        data: Data to fix
        target_formats: Optional target format specifications

    Returns:
        Dict: Fixed data with preserved precision
    """
    target_formats = target_formats or {}
    fixed_data: dict[str, Any] = {}

    for field_name, value in data.items():
        if value is None:
            fixed_data[field_name] = None
            continue

        # Determine if this is a financial field
        is_financial = any(
            term in field_name.lower()
            for term in ["price", "amount", "value", "cost", "fee", "balance", "pnl", "volume"]
        )

        if is_financial and isinstance(value, float):
            # Convert float to Decimal to prevent precision loss
            try:
                decimal_value = Decimal(str(value))
                fixed_data[field_name] = decimal_value

                logger.debug(
                    f"Fixed precision cascade for {field_name}: {value} -> {decimal_value}"
                )

            except (InvalidOperation, ValueError) as e:
                logger.warning(f"Could not convert {field_name} to Decimal: {e}")
                fixed_data[field_name] = value
        else:
            fixed_data[field_name] = value

    return fixed_data


# Export the main functions that address the critical issues
__all__ = [
    "DataFlowIntegrityError",
    "DataFlowValidator",
    "IntegrityPreservingConverter",
    "PrecisionTracker",
    "fix_precision_cascade",
    "get_data_flow_validator",
    "get_integrity_converter",
    "get_precision_tracker",
    "validate_cross_module_data",
]
