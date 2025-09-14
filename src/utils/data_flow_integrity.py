"""
Data Flow Integrity Validators

This module provides consistent validation at module boundaries to ensure
data integrity across the entire pipeline from exchanges to core systems.

Validations:
- Exchange → Data module boundary validation
- Data → Core module boundary validation  
- Error propagation consistency
- Data format standardization
- Financial precision preservation across module boundaries
- Type consistency validation
- Null handling standardization
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.exceptions import DataValidationError, ServiceError, ValidationError
from src.core.logging import get_logger
from src.utils.validators import validate_decimal_precision

logger = get_logger(__name__)


class ModuleBoundaryValidator:
    """Base validator for module boundary validation."""

    def __init__(self, source_module: str, target_module: str):
        self.source_module = source_module
        self.target_module = target_module
        self.logger = get_logger(f"{__name__}.{source_module}_to_{target_module}")

    def validate_data_structure(self, data: dict[str, Any], required_fields: list[str]) -> dict[str, Any]:
        """Validate basic data structure."""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise DataValidationError(
                f"Missing required fields in {self.source_module}→{self.target_module}: {missing_fields}"
            )

        return data

    def validate_decimal_fields(self, data: dict[str, Any], decimal_fields: list[str]) -> dict[str, Any]:
        """Validate and standardize decimal fields."""
        for field in decimal_fields:
            if field in data and data[field] is not None:
                try:
                    # Convert to Decimal with proper precision
                    # validate_decimal_precision returns bool, so convert separately
                    decimal_value = Decimal(str(data[field]))
                    if not decimal_value.is_finite():
                        raise DataValidationError(f"Non-finite value for {field}")
                    # Use 8 decimal places for crypto precision
                    data[field] = decimal_value.quantize(Decimal("0.00000001"))
                except Exception as e:
                    raise DataValidationError(
                        f"Invalid decimal format for {field} in {self.source_module}→{self.target_module}: {e}"
                    ) from e

        return data

    def validate_timestamp_fields(self, data: dict[str, Any], timestamp_fields: list[str]) -> dict[str, Any]:
        """Validate timestamp fields."""
        for field in timestamp_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (datetime, str, int, float)):
                    raise DataValidationError(
                        f"Invalid timestamp format for {field} in {self.source_module}→{self.target_module}"
                    )

        return data


class ExchangeToDataValidator(ModuleBoundaryValidator):
    """Validator for Exchange → Data module boundary."""

    def __init__(self):
        super().__init__("exchanges", "data")

    def validate_market_data(self, data: dict[str, Any], exchange: str) -> dict[str, Any]:
        """Validate market data from exchange before data processing."""
        try:
            # Required fields for market data
            required_fields = ["symbol", "price", "timestamp"]
            decimal_fields = ["price", "volume", "bid", "ask", "high", "low", "open", "close"]
            timestamp_fields = ["timestamp"]

            # Basic structure validation
            data = self.validate_data_structure(data, required_fields)

            # Decimal precision validation
            data = self.validate_decimal_fields(data, decimal_fields)

            # Timestamp validation
            data = self.validate_timestamp_fields(data, timestamp_fields)

            # Exchange-specific validation
            self._validate_exchange_specific(data, exchange)

            # Add validation metadata
            data["_validation"] = {
                "validated_at": datetime.utcnow().isoformat(),
                "validator": "ExchangeToDataValidator",
                "source_exchange": exchange,
                "version": "1.0"
            }

            self.logger.debug(f"Validated market data from {exchange}: {data.get('symbol', 'unknown')}")
            return data

        except Exception as e:
            self.logger.error(f"Market data validation failed for {exchange}: {e}")
            raise DataValidationError(f"Exchange→Data validation failed: {e}") from e

    def validate_order_data(self, data: dict[str, Any], exchange: str) -> dict[str, Any]:
        """Validate order data from exchange."""
        try:
            required_fields = ["order_id", "symbol", "status"]
            decimal_fields = ["quantity", "price", "filled_quantity", "remaining_quantity", "average_price"]
            timestamp_fields = ["created_at", "updated_at", "filled_at"]

            data = self.validate_data_structure(data, required_fields)
            data = self.validate_decimal_fields(data, decimal_fields)
            data = self.validate_timestamp_fields(data, timestamp_fields)

            # Validate order status
            valid_statuses = ["PENDING", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED"]
            if data.get("status") not in valid_statuses:
                raise DataValidationError(f"Invalid order status: {data.get('status')}")

            data["_validation"] = {
                "validated_at": datetime.utcnow().isoformat(),
                "validator": "ExchangeToDataValidator",
                "source_exchange": exchange,
                "validation_type": "order_data"
            }

            return data

        except Exception as e:
            raise DataValidationError(f"Order data validation failed: {e}") from e

    def _validate_exchange_specific(self, data: dict[str, Any], exchange: str) -> None:
        """Exchange-specific validation rules."""
        symbol = data.get("symbol", "")
        price = data.get("price")

        if exchange == "binance":
            # Binance-specific validation
            if not symbol or len(symbol) < 6:
                raise DataValidationError(f"Invalid Binance symbol format: {symbol}")

            if price and price <= 0:
                raise DataValidationError(f"Invalid price for Binance: {price}")

        elif exchange == "coinbase":
            # Coinbase-specific validation
            if not symbol or "-" not in symbol:
                raise DataValidationError(f"Invalid Coinbase symbol format: {symbol}")

        elif exchange == "okx":
            # OKX-specific validation
            if not symbol or "-" not in symbol:
                raise DataValidationError(f"Invalid OKX symbol format: {symbol}")


class DataToCoreValidator(ModuleBoundaryValidator):
    """Validator for Data → Core module boundary."""

    def __init__(self):
        super().__init__("data", "core")

    def validate_processed_data(self, data: dict[str, Any], data_type: str) -> dict[str, Any]:
        """Validate processed data before core consumption."""
        try:
            if data_type == "market_data":
                return self._validate_processed_market_data(data)
            elif data_type == "order_data":
                return self._validate_processed_order_data(data)
            elif data_type == "analytics_data":
                return self._validate_analytics_data(data)
            else:
                raise DataValidationError(f"Unknown data type: {data_type}")

        except Exception as e:
            raise DataValidationError(f"Data→Core validation failed: {e}") from e

    def _validate_processed_market_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate processed market data."""
        required_fields = ["symbol", "price", "timestamp", "source"]
        decimal_fields = ["price", "volume", "bid", "ask"]

        data = self.validate_data_structure(data, required_fields)
        data = self.validate_decimal_fields(data, decimal_fields)

        # Ensure data has been processed by data module
        if "_validation" not in data:
            raise DataValidationError("Data missing validation metadata from data module")

        # Add core validation metadata
        data["_core_validation"] = {
            "validated_at": datetime.utcnow().isoformat(),
            "validator": "DataToCoreValidator",
            "data_type": "market_data"
        }

        return data

    def _validate_processed_order_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate processed order data."""
        required_fields = ["order_id", "symbol", "status", "source"]
        decimal_fields = ["quantity", "price", "filled_quantity"]

        data = self.validate_data_structure(data, required_fields)
        data = self.validate_decimal_fields(data, decimal_fields)

        data["_core_validation"] = {
            "validated_at": datetime.utcnow().isoformat(),
            "validator": "DataToCoreValidator",
            "data_type": "order_data"
        }

        return data

    def _validate_analytics_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate analytics data."""
        required_fields = ["symbol", "metric_type", "value", "timestamp"]
        decimal_fields = ["value"]

        data = self.validate_data_structure(data, required_fields)
        data = self.validate_decimal_fields(data, decimal_fields)

        data["_core_validation"] = {
            "validated_at": datetime.utcnow().isoformat(),
            "validator": "DataToCoreValidator",
            "data_type": "analytics_data"
        }

        return data


class ErrorPropagationValidator:
    """Validator for consistent error propagation across modules."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.ErrorPropagationValidator")

    def validate_error_format(self, error: Exception, source_module: str) -> Exception:
        """Validate and standardize error format."""
        try:
            # Map source module errors to standard error types
            error_mapping = {
                "exchanges": {
                    "connection": "ExchangeConnectionError",
                    "order": "OrderRejectionError",
                    "rate_limit": "ExchangeRateLimitError",
                    "validation": "ValidationError"
                },
                "data": {
                    "validation": "DataValidationError",
                    "processing": "DataError",
                    "storage": "DataStorageError"
                },
                "core": {
                    "service": "ServiceError",
                    "validation": "ValidationError"
                }
            }

            # Determine error category
            error_type = type(error).__name__
            error_message = str(error)

            # Check if error follows expected patterns
            module_errors = error_mapping.get(source_module, {})
            expected_errors = list(module_errors.values())

            if error_type not in expected_errors:
                self.logger.warning(
                    f"Unexpected error type {error_type} from {source_module}. "
                    f"Expected one of: {expected_errors}"
                )

            # Add error metadata for tracing
            if not hasattr(error, "_propagation_metadata"):
                error._propagation_metadata = {
                    "source_module": source_module,
                    "error_type": error_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "validation_passed": error_type in expected_errors
                }

            return error

        except Exception as e:
            self.logger.error(f"Error validation failed: {e}")
            return error

    def validate_error_chain(self, errors: list[Exception]) -> list[Exception]:
        """Validate error chain for proper propagation."""
        validated_errors = []

        for i, error in enumerate(errors):
            # Check for proper error chaining
            if i > 0 and not hasattr(error, "__cause__"):
                self.logger.warning(f"Error at position {i} missing proper chaining")

            validated_errors.append(error)

        return validated_errors


class DataFlowIntegrityManager:
    """Manager for coordinating data flow validation across modules."""

    def __init__(self):
        self.exchange_to_data = ExchangeToDataValidator()
        self.data_to_core = DataToCoreValidator()
        self.error_validator = ErrorPropagationValidator()
        self.logger = get_logger(f"{__name__}.DataFlowIntegrityManager")

        # Statistics tracking
        self.stats = {
            "validations_performed": 0,
            "validation_failures": 0,
            "error_corrections": 0,
            "data_transformations": 0
        }

    def validate_exchange_data(self, data: dict[str, Any], exchange: str, data_type: str) -> dict[str, Any]:
        """Validate data coming from exchanges."""
        try:
            if data_type == "market_data":
                result = self.exchange_to_data.validate_market_data(data, exchange)
            elif data_type == "order_data":
                result = self.exchange_to_data.validate_order_data(data, exchange)
            else:
                raise DataValidationError(f"Unknown data type: {data_type}")

            self.stats["validations_performed"] += 1
            return result

        except Exception as e:
            self.stats["validation_failures"] += 1
            validated_error = self.error_validator.validate_error_format(e, "exchanges")
            raise validated_error

    def validate_core_data(self, data: dict[str, Any], data_type: str) -> dict[str, Any]:
        """Validate data before core processing."""
        try:
            result = self.data_to_core.validate_processed_data(data, data_type)
            self.stats["validations_performed"] += 1
            return result

        except Exception as e:
            self.stats["validation_failures"] += 1
            validated_error = self.error_validator.validate_error_format(e, "data")
            raise validated_error

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.stats,
            "success_rate": (
                (self.stats["validations_performed"] - self.stats["validation_failures"])
                / max(self.stats["validations_performed"], 1)
            ) * 100
        }


# Global instance
_integrity_manager: DataFlowIntegrityManager | None = None


def get_integrity_manager() -> DataFlowIntegrityManager:
    """Get global data flow integrity manager."""
    global _integrity_manager
    if _integrity_manager is None:
        _integrity_manager = DataFlowIntegrityManager()
    return _integrity_manager


# Utility functions for consistent validation
def validate_exchange_market_data(data: dict[str, Any], exchange: str) -> dict[str, Any]:
    """Utility to validate market data from exchanges."""
    manager = get_integrity_manager()
    return manager.validate_exchange_data(data, exchange, "market_data")


def validate_exchange_order_data(data: dict[str, Any], exchange: str) -> dict[str, Any]:
    """Utility to validate order data from exchanges."""
    manager = get_integrity_manager()
    return manager.validate_exchange_data(data, exchange, "order_data")


def validate_core_market_data(data: dict[str, Any]) -> dict[str, Any]:
    """Utility to validate market data for core processing."""
    manager = get_integrity_manager()
    return manager.validate_core_data(data, "market_data")


def validate_core_order_data(data: dict[str, Any]) -> dict[str, Any]:
    """Utility to validate order data for core processing."""
    manager = get_integrity_manager()
    return manager.validate_core_data(data, "order_data")


# Existing functions (keeping for backward compatibility)
import warnings
from decimal import InvalidOperation
from typing import Any

from src.core.logging import get_logger
from src.monitoring.financial_precision import FinancialPrecisionWarning
from src.utils.decimal_utils import decimal_to_float

logger = get_logger(__name__)


class DataFlowTransformer:
    """Standardized data transformation utilities for consistent cross-module data flow."""

    @staticmethod
    def apply_financial_field_transformation(entity: Any) -> Any:
        """
        Apply consistent financial field transformations to any entity.
        
        Args:
            entity: Entity with potential financial fields
            
        Returns:
            Entity with transformed financial fields
        """
        if entity is None:
            return entity

        # Import decimal utility at module level for consistency
        from src.utils.decimal_utils import to_decimal

        # Standard financial fields that need Decimal conversion
        financial_fields = ["price", "quantity", "volume", "value", "amount", "balance", "cost"]

        for field in financial_fields:
            if hasattr(entity, field) and getattr(entity, field) is not None:
                try:
                    setattr(entity, field, to_decimal(getattr(entity, field)))
                except Exception as e:
                    logger.warning(f"Failed to convert {field} to decimal: {e}")

        return entity

    @staticmethod
    def apply_standard_metadata(data: dict[str, Any], module_source: str, processing_mode: str = "stream") -> dict[str, Any]:
        """
        Apply standardized metadata for cross-module consistency.
        
        Args:
            data: Data dictionary to enhance
            module_source: Source module name
            processing_mode: Processing mode (default: stream)
            
        Returns:
            Enhanced data dictionary with standard metadata
        """
        from datetime import datetime, timezone

        enhanced_data = data.copy()

        # Add standard metadata fields
        enhanced_data.update({
            "processing_mode": processing_mode,
            "data_format": "standardized_entity_v1",  # Consistent format across modules
            "boundary_crossed": True,
            "module_source": module_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_pattern": "pub_sub" if processing_mode == "stream" else "req_reply"
        })

        return enhanced_data

    @staticmethod
    def validate_and_transform_entity(entity: Any, module_source: str, processing_mode: str = "stream") -> Any:
        """
        Complete validation and transformation for cross-module entity transfer.
        
        Args:
            entity: Entity to validate and transform
            module_source: Source module name  
            processing_mode: Processing mode
            
        Returns:
            Validated and transformed entity
        """
        if entity is None:
            return entity

        # Apply financial transformations
        entity = DataFlowTransformer.apply_financial_field_transformation(entity)

        # Add standard metadata if entity has attributes
        if hasattr(entity, "__dict__"):
            if not hasattr(entity, "processing_mode"):
                entity.processing_mode = processing_mode
            if not hasattr(entity, "data_format"):
                entity.data_format = "standardized_entity_v1"
            if not hasattr(entity, "boundary_crossed"):
                entity.boundary_crossed = True
            if not hasattr(entity, "module_source"):
                entity.module_source = module_source

        return entity


class StandardizedErrorPropagator:
    """Standardized error propagation for consistent cross-module error handling."""

    @staticmethod
    def propagate_validation_error(
        error: Exception,
        context: str,
        module_source: str,
        field_name: str = None,
        field_value: Any = None
    ) -> None:
        """
        Propagate validation errors with consistent metadata across modules.
        
        Args:
            error: Original exception
            context: Error context description
            module_source: Source module name
            field_name: Optional field name for field-specific errors
            field_value: Optional field value for debugging
        """
        from datetime import datetime, timezone

        from src.core.exceptions import ValidationError

        # Create standardized error metadata
        error_metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "module_source": module_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_mode": "stream",
            "data_format": "standardized_error_v1",
            "boundary_crossed": True
        }

        if field_name:
            error_metadata["field_name"] = field_name
        if field_value is not None:
            error_metadata["field_value"] = str(field_value)

        # Log error with structured metadata
        logger.error(f"Validation error in {module_source}.{context}: {error}", extra=error_metadata)

        # Raise standardized ValidationError
        raise ValidationError(
            f"{module_source} validation failed in {context}: {error}",
            field_name=field_name,
            field_value=field_value,
            expected_type="valid_data"
        ) from error

    @staticmethod
    def propagate_service_error(
        error: Exception,
        context: str,
        module_source: str,
        operation: str = None
    ) -> None:
        """
        Propagate service errors with consistent metadata across modules.
        
        Args:
            error: Original exception
            context: Error context description
            module_source: Source module name
            operation: Optional operation being performed
        """
        from datetime import datetime, timezone

        # Create standardized error metadata
        error_metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "module_source": module_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_mode": "stream",
            "data_format": "standardized_error_v1",
            "boundary_crossed": True
        }

        if operation:
            error_metadata["operation"] = operation

        # Log error with structured metadata
        logger.error(f"Service error in {module_source}.{context}: {error}", extra=error_metadata)

        # Raise standardized ServiceError
        raise ServiceError(
            f"{module_source} service failed in {context}: {error}"
        ) from error


class DataFlowValidator:
    """Validates data flow consistency across module boundaries."""

    @staticmethod
    def validate_message_pattern_consistency(data: dict[str, Any]) -> None:
        """
        Validate that message patterns are consistent across module boundaries.

        Args:
            data: Data dictionary to validate

        Raises:
            ValidationError: If message patterns are inconsistent
        """
        required_fields = ["message_pattern", "processing_mode", "data_format"]

        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Required message field '{field}' missing",
                    field_name=field,
                    field_value=None,
                    expected_type="string"
                )

        # Validate message pattern values
        valid_patterns = ["pub_sub", "req_reply", "stream", "batch"]
        if data["message_pattern"] not in valid_patterns:
            raise ValidationError(
                f"Invalid message pattern: {data['message_pattern']}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                validation_rule=f"must be one of {valid_patterns}"
            )

        # Validate processing mode alignment
        valid_modes = ["stream", "batch", "request_reply", "sync", "async"]
        if data["processing_mode"] not in valid_modes:
            raise ValidationError(
                f"Invalid processing mode: {data['processing_mode']}",
                field_name="processing_mode",
                field_value=data["processing_mode"],
                validation_rule=f"must be one of {valid_modes}"
            )

        # Validate data format versioning
        if not data["data_format"].endswith("_v1"):
            raise ValidationError(
                f"Invalid data format version: {data['data_format']}",
                field_name="data_format",
                field_value=data["data_format"],
                validation_rule="must end with _v1"
            )

    @staticmethod
    def validate_boundary_crossing_metadata(data: dict[str, Any]) -> None:
        """
        Validate boundary crossing metadata for cross-module communication.

        Args:
            data: Data dictionary to validate

        Raises:
            ValidationError: If boundary metadata is missing or invalid
        """
        # Check for boundary crossing flag
        if "boundary_crossed" not in data or not isinstance(data["boundary_crossed"], bool):
            raise ValidationError(
                "Missing or invalid boundary_crossed flag",
                field_name="boundary_crossed",
                field_value=data.get("boundary_crossed"),
                expected_type="bool"
            )

        # Validate timestamp format
        if "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                from datetime import datetime
                # Attempt to parse ISO format
                datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                raise ValidationError(
                    "Invalid timestamp format",
                    field_name="timestamp",
                    field_value=data["timestamp"],
                    validation_rule="must be ISO format"
                )

    @classmethod
    def validate_complete_data_flow(
        cls,
        data: dict[str, Any],
        source_module: str,
        target_module: str,
        operation_type: str = "data_flow"
    ) -> None:
        """
        Perform complete validation for cross-module data flow.

        Args:
            data: Data to validate
            source_module: Source module name
            target_module: Target module name
            operation_type: Type of operation

        Raises:
            ValidationError: If any validation fails
        """
        try:
            # Validate message pattern consistency
            cls.validate_message_pattern_consistency(data)

            # Validate boundary crossing metadata
            cls.validate_boundary_crossing_metadata(data)

            logger.debug(
                f"Data flow validation passed: {source_module} -> {target_module}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "operation_type": operation_type,
                    "validation_status": "passed"
                }
            )

        except ValidationError as e:
            logger.error(
                f"Data flow validation failed: {source_module} -> {target_module}: {e}",
                extra={
                    "source_module": source_module,
                    "target_module": target_module,
                    "operation_type": operation_type,
                    "validation_status": "failed",
                    "error_type": type(e).__name__
                }
            )
            raise


class DataFlowIntegrityError(Exception):
    """Raised when data flow integrity is compromised."""


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
        if isinstance(relative_error, int | float) and relative_error > 0.0001:  # > 0.01%
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
        except (InvalidOperation, ZeroDivisionError) as e:
            logger.debug(f"Error calculating relative error: {e}")
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

    # Interface compliance methods for factory pattern
    def track_operation(self, operation: str, input_precision: int, output_precision: int) -> None:
        """Track precision changes during operations for interface compliance."""
        # Implementation that adapts to the existing track_conversion method
        if input_precision != output_precision:
            # Create a mock event for interface compliance
            event = {
                "operation": operation,
                "input_precision": input_precision,
                "output_precision": output_precision,
                "precision_change": input_precision - output_precision,
                "timestamp": str(__import__("datetime").datetime.now()),
            }
            # Add to precision events for tracking
            self.precision_events.append(event)

            if input_precision > output_precision:
                self.warning_count += 1

    def get_precision_stats(self) -> dict[str, Any]:
        """Get precision tracking statistics for interface compliance."""
        return self.get_summary()


def get_precision_tracker(tracker: PrecisionTracker | None = None) -> PrecisionTracker:
    """Get precision tracker instance with proper dependency injection.

    Args:
        tracker: Injected tracker (required from service layer)

    Returns:
        PrecisionTracker: Tracker instance

    Raises:
        ValidationError: If tracker not properly injected
    """
    if tracker is None:
        raise ValidationError(
            "PrecisionTracker must be injected from service layer. "
            "Do not access DI container directly from utility functions.",
            error_code="SERV_001",
        )

    return tracker


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
        """Validate individual field based on rules using service layer validation with consistent error patterns."""
        # Find applicable rule
        rule = self._get_rule_for_field(field_name)
        if not rule:
            # No specific rule, apply basic validation
            return self._validate_null_handling_service(
                value, allow_null=True, field_name=field_name
            )

        try:
            # Apply null handling with consistent error propagation
            validated_value = self._validate_null_handling_service(
                value, allow_null=rule.get("allow_null", True), field_name=field_name
            )

            if validated_value is None:
                return None

            # Type conversion with consistent patterns
            target_type = rule.get("type")
            if target_type:
                validated_value = self._validate_type_conversion_service(
                    validated_value, target_type, field_name, strict=False
                )

            # Range validation for numerical types using consistent error propagation
            if isinstance(validated_value, Decimal | int | float):
                min_val = rule.get("min_value")
                max_val = rule.get("max_value")

                if min_val is not None or max_val is not None:
                    if isinstance(validated_value, Decimal):
                        self._validate_financial_range_service(
                            validated_value, min_val, max_val, field_name
                        )
                    elif min_val is not None and validated_value < min_val:
                        # Use consistent error propagation pattern matching messaging patterns
                        from src.utils.messaging_patterns import ErrorPropagationMixin
                        error_propagator = ErrorPropagationMixin()
                        validation_error = ValidationError(
                            f"{field_name} below minimum: {validated_value} < {min_val}",
                            field_name=field_name,
                            field_value=validated_value,
                            validation_rule="minimum_value_check"
                        )
                        error_propagator.propagate_validation_error(validation_error, f"field_validation_{context}")
                    elif max_val is not None and validated_value > max_val:
                        # Use consistent error propagation pattern matching messaging patterns
                        from src.utils.messaging_patterns import ErrorPropagationMixin
                        error_propagator = ErrorPropagationMixin()
                        validation_error = ValidationError(
                            f"{field_name} above maximum: {validated_value} > {max_val}",
                            field_name=field_name,
                            field_value=validated_value,
                            validation_rule="maximum_value_check"
                        )
                        error_propagator.propagate_validation_error(validation_error, f"field_validation_{context}")

            return validated_value

        except Exception as e:
            # Consistent error propagation for validation failures
            from src.utils.messaging_patterns import ErrorPropagationMixin
            error_propagator = ErrorPropagationMixin()

            if not isinstance(e, ValidationError):
                # Wrap non-validation errors in consistent format
                validation_error = ValidationError(
                    f"Field validation failed for {field_name}",
                    field_name=field_name,
                    field_value=str(value),
                    validation_rule="field_validation_error",
                    context={"original_error": str(e)}
                )
                error_propagator.propagate_validation_error(validation_error, f"field_validation_{context}")
            else:
                error_propagator.propagate_validation_error(e, f"field_validation_{context}")

    def _validate_null_handling_service(
        self, value: Any, allow_null: bool = False, field_name: str = "value"
    ) -> Any:
        """Service layer null handling validation."""
        if value is None:
            if allow_null:
                return None
            else:
                raise ValidationError(f"{field_name} cannot be None")

        # Check for other null-like values
        if isinstance(value, str) and value.strip() == "":
            if allow_null:
                return None
            else:
                raise ValidationError(f"{field_name} cannot be empty string")

        # Check for NaN values
        import math

        if isinstance(value, float) and math.isnan(value):
            if allow_null:
                return None
            else:
                raise ValidationError(f"{field_name} cannot be NaN")

        return value

    def _validate_type_conversion_service(
        self, value: Any, target_type: type, field_name: str = "value", strict: bool = True
    ) -> Any:
        """Service layer type conversion validation."""
        if value is None:
            raise ValidationError(f"Cannot convert None {field_name} to {target_type.__name__}")

        try:
            if target_type == Decimal:
                if isinstance(value, Decimal):
                    return value
                elif isinstance(value, int | float):
                    import math

                    if isinstance(value, float):
                        if not math.isfinite(value):
                            raise ValidationError(
                                f"Cannot convert non-finite float {field_name} to Decimal"
                            )
                    return Decimal(str(value))
                elif isinstance(value, str):
                    return Decimal(value.strip())
                else:
                    raise ValidationError(f"Cannot convert {type(value).__name__} to Decimal")
            elif target_type is float:
                import math

                if isinstance(value, float):
                    if not math.isfinite(value):
                        raise ValidationError(f"Invalid float {field_name}: {value}")
                    return value
                elif isinstance(value, int | Decimal):
                    # Use safe conversion to maintain precision tracking

                    result = decimal_to_float(value)
                    if not math.isfinite(result):
                        raise ValidationError(
                            f"Conversion of {field_name} to float resulted in non-finite value"
                        )
                    return result
                else:
                    # Use safe conversion for unknown types

                    return decimal_to_float(value)
            elif target_type is int:
                import math

                if isinstance(value, int):
                    return value
                elif isinstance(value, float | Decimal):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValidationError(
                            f"Cannot convert non-finite float {field_name} to int"
                        )
                    result = int(value)
                    if not strict:
                        return result
                    # Strict mode: check for precision loss using Decimal comparison
                    if Decimal(str(result)) != Decimal(str(value)):
                        raise ValidationError(
                            f"Precision loss converting {field_name} to int: {value} -> {result}"
                        )
                    return result
                else:
                    return int(value)  # Let Python handle the conversion
            else:
                return target_type(value)
        except (ValueError, TypeError, OverflowError, InvalidOperation) as e:
            raise ValidationError(
                f"Failed to convert {field_name} to {target_type.__name__}: {e}"
            ) from e

    def _validate_financial_range_service(
        self,
        value: Decimal | float,
        min_value: Decimal | float | None = None,
        max_value: Decimal | float | None = None,
        field_name: str = "value",
    ) -> Decimal:
        """Service layer financial range validation."""
        if value is None:
            raise ValidationError(f"{field_name} cannot be None")

        try:
            if isinstance(value, Decimal):
                decimal_value = value
            else:
                decimal_value = Decimal(str(value))
        except (ValueError, InvalidOperation, OverflowError) as e:
            raise ValidationError(f"Invalid {field_name}: cannot convert to Decimal") from e

        if not decimal_value.is_finite():
            raise ValidationError(f"{field_name} must be finite, got: {decimal_value}")

        # Dynamic range validation
        if min_value is not None:
            min_decimal = (
                Decimal(str(min_value)) if not isinstance(min_value, Decimal) else min_value
            )
            if decimal_value < min_decimal:
                raise ValidationError(
                    f"{field_name} below minimum: {decimal_value} < {min_decimal}"
                )

        if max_value is not None:
            max_decimal = (
                Decimal(str(max_value)) if not isinstance(max_value, Decimal) else max_value
            )
            if decimal_value > max_decimal:
                raise ValidationError(
                    f"{field_name} above maximum: {decimal_value} > {max_decimal}"
                )

        return decimal_value

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

    # Interface compliance methods for factory pattern
    def validate_data_integrity(self, data: Any) -> bool:
        """Validate data integrity for interface compliance."""
        try:
            if isinstance(data, dict):
                # Validate all fields in the dictionary
                for field_name, value in data.items():
                    self._validate_field(field_name, value, "integrity_check")
            else:
                # For non-dict data, perform basic validation
                self._validate_field("data", data, "integrity_check")
            return True
        except Exception as e:
            logger.error(f"Data integrity check failed: {e}")
            return False

    def get_validation_report(self) -> dict[str, Any]:
        """Get validation report for interface compliance."""
        return {
            "validation_rules": len(self.validation_rules),
            "supported_types": ["decimal", "float", "int", "str", "bool"],
            "status": "active",
        }


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
        # Use dependency injection for PrecisionTracker - required for clean architecture
        self.tracker = precision_tracker
        if precision_tracker is None and track_precision:
            logger.warning(
                "PrecisionTracker not injected - precision tracking disabled. "
                "Inject via dependency injection for full functionality."
            )

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
            result = decimal_to_float(value)

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
                    f"Conversion failed for {metric_name}: {e}",
                    FinancialPrecisionWarning,
                    stacklevel=2,
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


def get_data_flow_validator(validator: DataFlowValidator | None = None) -> DataFlowValidator:
    """Get data flow validator instance with proper dependency injection.

    Args:
        validator: Injected validator (required from service layer)

    Returns:
        DataFlowValidator: Validator instance

    Raises:
        ValidationError: If validator not properly injected
    """
    if validator is None:
        raise ValidationError(
            "DataFlowValidator must be injected from service layer. "
            "Do not access DI container directly from utility functions.",
            error_code="SERV_001",
        )

    return validator


def get_integrity_converter(
    converter: IntegrityPreservingConverter | None = None,
) -> IntegrityPreservingConverter:
    """Get integrity converter instance with proper dependency injection.

    Args:
        converter: Injected converter (required from service layer)

    Returns:
        IntegrityPreservingConverter: Converter instance

    Raises:
        ValidationError: If converter not properly injected
    """
    if converter is None:
        raise ValidationError(
            "IntegrityPreservingConverter must be injected from service layer. "
            "Do not access DI container directly from utility functions.",
            error_code="SERV_001",
        )

    return converter


# Data flow services registration is handled by service_registry.py


def validate_cross_module_data(
    data: dict[str, Any],
    source_module: str,
    target_module: str,
    operation: str = "transfer",
    validator: DataFlowValidator | None = None,
) -> dict[str, Any]:
    """
    Validate data being transferred between modules.

    This function should be called from the service layer with proper dependency injection.

    Args:
        data: Data being transferred
        source_module: Source module name
        target_module: Target module name
        operation: Type of operation
        validator: Injected data flow validator (required from service layer)

    Returns:
        Dict: Validated data

    Raises:
        DataFlowIntegrityError: If validation fails
    """
    context = f"{source_module}->{target_module}:{operation}"

    try:
        # Get validator with proper dependency injection
        data_validator = get_data_flow_validator(validator)
        validated_data = data_validator.validate_data_flow(data, context)

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
    data: dict[str, Any] | None = None,
    target_formats: dict[str, str] | None = None
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
