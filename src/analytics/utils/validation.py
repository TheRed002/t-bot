"""Validation utilities for analytics module."""

import re
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.base.component import BaseComponent
from src.core.exceptions import ValidationError


class ValidationHelper(BaseComponent):
    """Centralized validation utilities for analytics."""

    def __init__(self):
        super().__init__()

    def validate_export_format(self, format_name: str, supported_formats: list[str]) -> str:
        """Validate export format against supported formats.

        Args:
            format_name: Format to validate
            supported_formats: List of supported format names

        Returns:
            Normalized format name

        Raises:
            ValidationError: If format is not supported
        """
        normalized_format = format_name.lower().strip()
        supported_lower = [fmt.lower() for fmt in supported_formats]

        if normalized_format not in supported_lower:
            raise ValidationError(
                f"Unsupported export format: {format_name}",
                details={"provided_format": format_name, "supported_formats": supported_formats},
            )

        # Return the original case format
        format_index = supported_lower.index(normalized_format)
        return supported_formats[format_index]

    def validate_date_range(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        allow_none: bool = True,
        max_range_days: int | None = None,
    ) -> None:
        """Validate date range parameters.

        Args:
            start_date: Start date
            end_date: End date
            allow_none: Whether None dates are allowed
            max_range_days: Maximum allowed range in days

        Raises:
            ValidationError: If date range is invalid
        """
        if not allow_none and (start_date is None or end_date is None):
            raise ValidationError("Start date and end date are required")

        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise ValidationError(
                    "Start date must be before or equal to end date",
                    details={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                )

            if max_range_days is not None:
                range_days = (end_date - start_date).days
                if range_days > max_range_days:
                    raise ValidationError(
                        f"Date range exceeds maximum allowed range of {max_range_days} days",
                        details={
                            "requested_range_days": range_days,
                            "max_allowed_days": max_range_days,
                        },
                    )

    def validate_numeric_range(
        self,
        value: int | float | Decimal,
        min_value: int | float | Decimal | None = None,
        max_value: int | float | Decimal | None = None,
        field_name: str = "value",
    ) -> None:
        """Validate numeric value is within specified range.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Name of the field for error messages

        Raises:
            ValidationError: If value is outside allowed range
        """
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{field_name} must be greater than or equal to {min_value}",
                details={
                    "field_name": field_name,
                    "value": str(value),
                    "min_value": str(min_value),
                },
            )

        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} must be less than or equal to {max_value}",
                details={
                    "field_name": field_name,
                    "value": str(value),
                    "max_value": str(max_value),
                },
            )

    def validate_required_fields(self, data: dict[str, Any], required_fields: list[str]) -> None:
        """Validate that required fields are present in data.

        Args:
            data: Data dictionary to validate
            required_fields: List of required field names

        Raises:
            ValidationError: If any required fields are missing
        """
        missing_fields = [
            field for field in required_fields if field not in data or data[field] is None
        ]

        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                details={
                    "missing_fields": missing_fields,
                    "required_fields": required_fields,
                    "provided_fields": list(data.keys()),
                },
            )

    def validate_list_not_empty(self, data_list: list[Any], field_name: str = "list") -> None:
        """Validate that a list is not empty.

        Args:
            data_list: List to validate
            field_name: Name of the field for error messages

        Raises:
            ValidationError: If list is empty
        """
        if not data_list:
            raise ValidationError(
                f"{field_name} cannot be empty",
                details={"field_name": field_name, "list_length": len(data_list)},
            )

    def validate_string_pattern(
        self,
        value: str,
        pattern: str,
        field_name: str = "value",
        pattern_description: str | None = None,
    ) -> None:
        """Validate string matches a regex pattern.

        Args:
            value: String to validate
            pattern: Regex pattern
            field_name: Name of the field for error messages
            pattern_description: Human-readable pattern description

        Raises:
            ValidationError: If string doesn't match pattern
        """
        if not re.match(pattern, value):
            description = pattern_description or f"pattern {pattern}"
            raise ValidationError(
                f"{field_name} must match {description}",
                details={"field_name": field_name, "value": value, "pattern": pattern},
            )

    def validate_choice(
        self, value: Any, choices: list[Any], field_name: str = "value", case_sensitive: bool = True
    ) -> Any:
        """Validate value is one of allowed choices.

        Args:
            value: Value to validate
            choices: List of allowed choices
            field_name: Name of the field for error messages
            case_sensitive: Whether string comparison is case sensitive

        Returns:
            The matched choice (useful for case-insensitive matching)

        Raises:
            ValidationError: If value is not in choices
        """
        if case_sensitive or not isinstance(value, str):
            if value in choices:
                return value
        else:
            # Case-insensitive string matching
            value_lower = value.lower()
            for choice in choices:
                if isinstance(choice, str) and choice.lower() == value_lower:
                    return choice

        raise ValidationError(
            f"{field_name} must be one of: {', '.join(map(str, choices))}",
            details={"field_name": field_name, "value": value, "allowed_choices": choices},
        )

    def validate_data_structure(
        self,
        data: Any,
        validator_func: Callable[[Any], bool],
        field_name: str = "data",
        error_message: str | None = None,
    ) -> None:
        """Validate data using a custom validator function.

        Args:
            data: Data to validate
            validator_func: Function that returns True if data is valid
            field_name: Name of the field for error messages
            error_message: Custom error message

        Raises:
            ValidationError: If validator function returns False
        """
        try:
            is_valid = validator_func(data)
        except Exception as e:
            raise ValidationError(
                f"Validation failed for {field_name}: {e!s}",
                details={"field_name": field_name, "validation_error": str(e)},
            ) from e

        if not is_valid:
            message = error_message or f"Invalid {field_name} structure"
            raise ValidationError(message, context={"field_name": field_name})

    def validate_alert_severity(self, severity: str) -> str:
        """Validate alert severity level.

        Args:
            severity: Severity level to validate

        Returns:
            Normalized severity level

        Raises:
            ValidationError: If severity is invalid
        """
        valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return self.validate_choice(severity, valid_severities, "severity", case_sensitive=False)

    def validate_time_window(self, window_str: str) -> int:
        """Validate and parse time window string to seconds.

        Args:
            window_str: Time window string like "5m", "1h", "1d"

        Returns:
            Time window in seconds

        Raises:
            ValidationError: If time window format is invalid
        """
        pattern = r"^(\d+)([smhd])$"
        match = re.match(pattern, window_str.lower())

        if not match:
            raise ValidationError(
                "Invalid time window format. Use format like '5m', '1h', '1d'",
                details={"provided_window": window_str, "expected_format": "NUMBER + (s|m|h|d)"},
            )

        value, unit = match.groups()
        value = int(value)

        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return value * multipliers[unit]

    def validate_analytics_boundary(self, data: dict[str, Any], target_module: str) -> None:
        """Validate data at analytics module boundary with core consistency patterns.
        
        Args:
            data: Data crossing module boundary
            target_module: Target module name
            
        Raises:
            ValidationError: If boundary validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(
                "Analytics boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Required fields for analytics module boundaries
        required_fields = ["processing_mode", "data_format", "message_pattern", "source"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Missing required boundary field '{field}' in analytics data",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

        # Validate processing mode consistency
        valid_modes = ["stream", "batch", "request_reply"]
        if data["processing_mode"] not in valid_modes:
            raise ValidationError(
                f"Invalid processing_mode for analytics boundary: {data['processing_mode']}",
                field_name="processing_mode",
                field_value=data["processing_mode"],
                expected_type=f"one of {valid_modes}",
            )

        # Validate message pattern alignment
        valid_patterns = ["pub_sub", "req_reply", "batch", "stream"]
        if data["message_pattern"] not in valid_patterns:
            raise ValidationError(
                f"Invalid message_pattern for analytics boundary: {data['message_pattern']}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                expected_type=f"one of {valid_patterns}",
            )

        # Validate data format consistency
        if not data["data_format"].startswith("analytics_") or not data["data_format"].endswith("_v1"):
            raise ValidationError(
                f"Invalid analytics data_format: {data['data_format']}. Must start with 'analytics_' and end with '_v1'",
                field_name="data_format",
                field_value=data["data_format"],
                expected_type="analytics_*_v1",
            )

        # Apply financial field validation if present
        financial_fields = ["price", "quantity", "volume", "value", "amount"]
        for field in financial_fields:
            if field in data and data[field] is not None:
                try:
                    from decimal import Decimal
                    value = Decimal(str(data[field]))
                    if value < 0:
                        raise ValidationError(
                            f"Financial field {field} cannot be negative at analytics boundary",
                            field_name=field,
                            field_value=value,
                            validation_rule="must be non-negative",
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Financial field {field} must be numeric at analytics boundary",
                        field_name=field,
                        field_value=data[field],
                        expected_type="numeric",
                    )

    def validate_cross_module_data(self, data: dict[str, Any], source_module: str, target_module: str) -> None:
        """Validate data for cross-module consistency aligned with core patterns.
        
        Args:
            data: Data to validate
            source_module: Source module name
            target_module: Target module name
            
        Raises:
            ValidationError: If cross-module validation fails
        """
        # Use existing boundary validation utilities for consistency
        try:
            from src.utils.messaging_patterns import BoundaryValidator

            if source_module == "analytics" and target_module == "core":
                # Validate analytics -> core boundary using consistent patterns
                boundary_data = {
                    "component": "analytics",
                    "operation": data.get("operation", "analytics_operation"),
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "processing_mode": data.get("processing_mode", "stream"),
                    "data_format": data.get("data_format", "analytics_data_v1"),
                    "message_pattern": data.get("message_pattern", "pub_sub"),
                    "boundary_crossed": True,
                }
                BoundaryValidator.validate_error_to_monitoring_boundary(boundary_data)

            elif source_module == "core" and target_module == "analytics":
                # Validate core -> analytics boundary
                self.validate_analytics_boundary(data, target_module)

        except Exception as e:
            raise ValidationError(
                f"Cross-module validation failed for {source_module} -> {target_module}: {e}",
                field_name="cross_module_data",
                field_value=str(data),
                expected_type="valid cross-module data",
            ) from e
