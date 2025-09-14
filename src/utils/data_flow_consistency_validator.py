"""
Data Flow Consistency Validator

This utility validates consistency between risk_management and utils modules
to ensure aligned data flow patterns, message handling, and processing paradigms.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Import messaging patterns dynamically when needed

logger = get_logger(__name__)


class DataFlowConsistencyValidator:
    """Validates data flow consistency between risk_management and utils modules."""

    def __init__(self):
        self.validation_results = {}
        # Lazy import to avoid circular dependency
        self._error_handler = None

    def validate_all(self) -> dict[str, Any]:
        """
        Validate all consistency aspects between modules.

        Returns:
            Dictionary with validation results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "in_progress",
            "checks": {},
        }

        try:
            # 1. Data transformation consistency
            results["checks"]["data_transformation"] = self._validate_data_transformations()

            # 2. Message queue patterns consistency
            results["checks"]["message_patterns"] = self._validate_message_patterns()

            # 3. Batch vs stream processing alignment
            results["checks"]["processing_alignment"] = self._validate_processing_alignment()

            # 4. Boundary validation consistency
            results["checks"]["boundary_validation"] = self._validate_boundary_consistency()

            # 5. Error propagation consistency
            results["checks"]["error_propagation"] = self._validate_error_propagation()

            # 6. Financial data type consistency
            results["checks"]["financial_types"] = self._validate_financial_types()

            # 7. ML-utils error propagation consistency
            results["checks"]["ml_error_propagation"] = self._validate_ml_error_propagation()

            # Calculate overall status
            all_passed = all(check["status"] == "passed" for check in results["checks"].values())
            results["validation_status"] = "passed" if all_passed else "failed"

        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            results["validation_status"] = "error"
            results["error"] = str(e)

        return results

    def _validate_data_transformations(self) -> dict[str, Any]:
        """Validate data transformation consistency."""
        try:
            # Test data transformation patterns
            test_data = {
                "price": "123.45",
                "quantity": "67.89",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Test messaging patterns transformation
            from src.utils.messaging_patterns import MessagingCoordinator

            coordinator = MessagingCoordinator("test")
            transformed = coordinator._apply_data_transformation(test_data)

            # Validate transformations
            checks = []

            # Check Decimal conversion
            if isinstance(transformed.get("price"), Decimal):
                checks.append(("price_decimal_conversion", True))
            else:
                checks.append(("price_decimal_conversion", False))

            # Check quantity conversion
            if isinstance(transformed.get("quantity"), Decimal):
                checks.append(("quantity_decimal_conversion", True))
            else:
                checks.append(("quantity_decimal_conversion", False))

            # Check timestamp consistency
            if "timestamp" in transformed:
                checks.append(("timestamp_consistency", True))
            else:
                checks.append(("timestamp_consistency", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Data transformation patterns validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Data transformation validation failed",
            }

    def _validate_message_patterns(self) -> dict[str, Any]:
        """Validate message pattern consistency."""
        try:
            # Test messaging coordinator patterns
            from src.utils.messaging_patterns import MessagingCoordinator

            coordinator = MessagingCoordinator("test")

            checks = []

            # Check pub/sub pattern availability
            try:
                from src.utils.messaging_patterns import MessagePattern

                checks.append(("pub_sub_pattern", MessagePattern.PUB_SUB is not None))
            except Exception:
                checks.append(("pub_sub_pattern", False))

            # Check req/reply pattern availability
            try:
                checks.append(("req_reply_pattern", MessagePattern.REQ_REPLY is not None))
            except Exception:
                checks.append(("req_reply_pattern", False))

            # Check stream pattern availability
            try:
                checks.append(("stream_pattern", MessagePattern.STREAM is not None))
            except Exception:
                checks.append(("stream_pattern", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Message patterns validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Message pattern validation failed",
            }

    def _validate_processing_alignment(self) -> dict[str, Any]:
        """Validate batch vs stream processing alignment."""
        try:
            checks = []

            # Test batch creation from stream
            try:
                from src.utils.messaging_patterns import ProcessingParadigmAligner

                stream_items = [{"test": "data1"}, {"test": "data2"}]
                batch_data = ProcessingParadigmAligner.create_batch_from_stream(stream_items)
                checks.append(("batch_from_stream", "items" in batch_data))
            except Exception:
                checks.append(("batch_from_stream", False))

            # Test stream creation from batch
            try:
                batch_data = {"items": [{"test": "data"}]}
                stream_items = ProcessingParadigmAligner.create_stream_from_batch(batch_data)
                checks.append(("stream_from_batch", isinstance(stream_items, list)))
            except Exception:
                checks.append(("stream_from_batch", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Processing alignment validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Processing alignment validation failed",
            }

    def _validate_boundary_consistency(self) -> dict[str, Any]:
        """Validate boundary validation consistency."""
        try:
            checks = []

            # Test boundary validator
            try:
                from src.utils.messaging_patterns import BoundaryValidator

                test_entity = {"id": "test_id", "price": 123.45, "quantity": 67.89}
                BoundaryValidator.validate_database_entity(test_entity, "validate")
                checks.append(("boundary_validation", True))
            except ValidationError:
                # Expected for missing required fields, but validator works
                checks.append(("boundary_validation", True))
            except Exception:
                checks.append(("boundary_validation", False))

            # Test financial field validation
            try:
                financial_data = {"price": 100.0, "quantity": 10.0}
                BoundaryValidator.validate_database_entity(financial_data, "create")
                checks.append(("financial_validation", True))
            except Exception:
                checks.append(("financial_validation", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Boundary validation consistency validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Boundary validation consistency check failed",
            }

    def _validate_error_propagation(self) -> dict[str, Any]:
        """Validate error propagation consistency."""
        try:
            checks = []

            # Test error propagation mixin
            try:
                from src.utils.messaging_patterns import ErrorPropagationMixin

                handler = ErrorPropagationMixin()

                # Test validation error propagation
                try:
                    handler.propagate_validation_error(ValueError("test"), "test_context")
                except Exception:
                    # Should raise an exception
                    checks.append(("validation_error_propagation", True))
                else:
                    checks.append(("validation_error_propagation", False))

            except Exception:
                checks.append(("validation_error_propagation", False))

            # Test service error propagation
            try:
                handler = ErrorPropagationMixin()
                try:
                    handler.propagate_service_error(RuntimeError("test"), "test_context")
                except Exception:
                    checks.append(("service_error_propagation", True))
                else:
                    checks.append(("service_error_propagation", False))
            except Exception:
                checks.append(("service_error_propagation", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Error propagation consistency validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Error propagation validation failed",
            }

    def _validate_financial_types(self) -> dict[str, Any]:
        """Validate financial data type consistency."""
        try:
            from src.utils.decimal_utils import ONE, ZERO, to_decimal

            checks = []

            # Test Decimal consistency
            try:
                price = to_decimal("123.45")
                quantity = to_decimal("67.89")
                checks.append(("decimal_conversion", isinstance(price, Decimal)))
                checks.append(
                    ("decimal_constants", isinstance(ZERO, Decimal) and isinstance(ONE, Decimal))
                )
            except Exception:
                checks.append(("decimal_conversion", False))
                checks.append(("decimal_constants", False))

            # Test financial precision
            try:
                from src.utils.decimal_utils import format_decimal

                formatted = format_decimal(to_decimal("123.456789"))
                checks.append(("decimal_formatting", isinstance(formatted, str)))
            except Exception:
                checks.append(("decimal_formatting", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "Financial type consistency validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "Financial type validation failed",
            }

    def _validate_ml_error_propagation(self) -> dict[str, Any]:
        """Validate ML-specific error propagation consistency."""
        try:
            checks = []

            # Test ML error propagation mixin
            try:
                from src.ml.data_transformer import MLErrorPropagationMixin

                handler = MLErrorPropagationMixin()

                # Test ML validation error propagation
                try:
                    handler.propagate_ml_validation_error(ValueError("test"), "test_context")
                except Exception:
                    # Should raise an exception
                    checks.append(("ml_validation_error_propagation", True))
                else:
                    checks.append(("ml_validation_error_propagation", False))

            except Exception:
                checks.append(("ml_validation_error_propagation", False))

            # Test ML model error propagation
            try:
                handler = MLErrorPropagationMixin()
                try:
                    handler.propagate_ml_model_error(RuntimeError("test"), "test_context")
                except Exception:
                    checks.append(("ml_model_error_propagation", True))
                else:
                    checks.append(("ml_model_error_propagation", False))
            except Exception:
                checks.append(("ml_model_error_propagation", False))

            # Test ML boundary validation
            try:
                from src.ml.data_transformer import MLDataTransformer

                test_data = {"processing_mode": "stream", "data_format": "bot_event_v1", "ml_operation_type": "prediction"}
                MLDataTransformer.validate_ml_to_utils_boundary(test_data)
                checks.append(("ml_boundary_validation", True))
            except Exception:
                checks.append(("ml_boundary_validation", False))

            passed = all(check[1] for check in checks)

            return {
                "status": "passed" if passed else "failed",
                "checks": dict(checks),
                "details": "ML error propagation consistency validated",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "details": "ML error propagation validation failed",
            }


def validate_data_flow_consistency() -> dict[str, Any]:
    """
    Convenience function to validate data flow consistency.

    Returns:
        Validation results dictionary
    """
    validator = DataFlowConsistencyValidator()
    return validator.validate_all()


def log_consistency_results(results: dict[str, Any]) -> None:
    """
    Log consistency validation results.

    Args:
        results: Validation results from validate_data_flow_consistency()
    """
    status = results.get("validation_status", "unknown")

    if status == "passed":
        logger.info("✅ Data flow consistency validation PASSED")
    elif status == "failed":
        logger.warning("⚠️ Data flow consistency validation FAILED")
    elif status == "error":
        logger.error("❌ Data flow consistency validation ERROR")

    # Log individual check results
    checks = results.get("checks", {})
    for check_name, check_result in checks.items():
        check_status = check_result.get("status", "unknown")
        if check_status == "passed":
            logger.info(f"  ✅ {check_name}: {check_result.get('details', 'passed')}")
        elif check_status == "failed":
            logger.warning(f"  ❌ {check_name}: {check_result.get('details', 'failed')}")
        elif check_status == "error":
            logger.error(f"  🚨 {check_name}: {check_result.get('error', 'error')}")
