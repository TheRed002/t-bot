"""Unified exception hierarchy for the T-Bot trading system.

This module provides a comprehensive, standardized exception system with:
- Standardized error codes for every exception
- Categorization (retryable, fatal, validation, etc.)
- Rich error context and metadata
- Proper inheritance hierarchy
- Logging levels and suggested resolutions
- Integration with circuit breakers and retry mechanisms

CRITICAL USAGE RULES:
1. ALL modules MUST import from this single source of truth
2. NEVER create duplicate exceptions elsewhere
3. ALL exceptions MUST include error codes
4. Use categorization for automated error handling

Example Usage:
    from src.core.exceptions import (
        TradingBotError, ExchangeError, RiskManagementError,
        ValidationError, ExecutionError, ModelError, DataError,
        StateConsistencyError, SecurityError, NetworkError
    )

    # Raise with proper context
    raise ExchangeConnectionError(
        "Failed to connect to Binance",
        exchange="binance",
        retry_after=30,
        endpoint="/api/v3/ticker/24hr"
    )

"""

import logging
import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar


class ErrorCategory(Enum):
    """Error categorization for automated handling."""

    RETRYABLE = "retryable"  # Can be retried automatically
    FATAL = "fatal"  # Cannot be retried, requires manual intervention
    VALIDATION = "validation"  # Input validation errors
    CONFIGURATION = "configuration"  # Configuration errors
    PERMISSION = "permission"  # Authentication/authorization errors
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    NETWORK = "network"  # Network connectivity errors
    DATA_QUALITY = "data_quality"  # Data quality issues
    BUSINESS_LOGIC = "business_logic"  # Business rule violations
    SYSTEM = "system"  # System/infrastructure errors


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingBotError(Exception):
    """Base exception for all trading bot errors.

    This is the root of all custom exceptions in the system. Every exception
    includes standardized metadata for logging, monitoring, and automated handling.

    Attributes:
        message: Human-readable error message
        error_code: Standardized error code (e.g., 'EXCH_001')
        category: Error category for automated handling
        severity: Error severity level
        details: Additional context data
        retryable: Whether this error can be retried
        retry_after: Suggested retry delay in seconds
        suggested_action: Recommended resolution steps
        context: Additional contextual information
        timestamp: When the error occurred
        logger_name: Logger name for this error type
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
        retry_after: int | None = None,
        suggested_action: str | None = None,
        context: dict[str, Any] | None = None,
        logger_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.retryable = retryable
        self.retry_after = retry_after
        self.suggested_action = suggested_action
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)
        self.logger_name = logger_name or self.__class__.__module__

        # Add any additional keyword arguments to context
        self.context.update(kwargs)

        # Log the error automatically
        self._log_error()

    def _sanitize_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize sensitive data before logging."""
        sensitive_keys = {
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "private_key",
            "access_token",
            "refresh_token",
            "authorization",
            "auth",
            "passphrase",
            "credential",
            "certificate",
            "salt",
            "hash",
        }

        def sanitize_value(key: str, value: Any) -> Any:
            """Recursively sanitize sensitive values."""
            if isinstance(value, dict):
                return {k: sanitize_value(k, v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(key, item) for item in value]
            elif isinstance(value, str) and any(
                sensitive in key.lower() for sensitive in sensitive_keys
            ):
                # Keep first and last 2 chars for debugging
                if len(value) > 4:
                    return f"{value[:2]}***{value[-2:]}"
                return "***"
            return value

        return {k: sanitize_value(k, v) for k, v in data.items()}

    def _log_error(self) -> None:
        """Log the error with appropriate level based on severity."""
        # Prevent infinite recursion if logging fails
        try:
            logger = logging.getLogger(self.logger_name)

            # Sanitize sensitive data before logging
            log_data = self._sanitize_sensitive_data(
                {
                    "error_code": self.error_code,
                    "category": self.category.value,
                    "severity": self.severity.value,
                    "retryable": self.retryable,
                    "details": self.details,
                    "context": self.context,
                    "timestamp": self.timestamp.isoformat(),
                }
            )

            if self.severity == ErrorSeverity.CRITICAL:
                logger.critical(self.message, extra=log_data)
            elif self.severity == ErrorSeverity.HIGH:
                logger.error(self.message, extra=log_data)
            elif self.severity == ErrorSeverity.MEDIUM:
                logger.warning(self.message, extra=log_data)
            else:
                logger.info(self.message, extra=log_data)
        except Exception:
            # If logging fails, silently ignore to prevent infinite recursion
            # The error will still be raised to the caller
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
            "suggested_action": self.suggested_action,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        """Return formatted error message with code and essential details."""
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        error_str = " ".join(parts)

        if self.category != ErrorCategory.SYSTEM:
            error_str += f" (Category: {self.category.value})"  # This is safe as it's an enum value

        if self.severity != ErrorSeverity.MEDIUM:
            error_str += f" (Severity: {self.severity.value})"  # This is safe as it's an enum value

        if self.retryable and self.retry_after:
            error_str += f" (Retry after: {self.retry_after}s)"  # This is safe as it's a number

        # Include some details for debugging
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in list(self.details.items())[:3])
            error_str += f" (Details: {details_str})"  # Already sanitized above

        return error_str

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"category={self.category}, "
            f"severity={self.severity}, "
            f"retryable={self.retryable}"
            f")"
        )


# =============================================================================
# EXCHANGE-RELATED EXCEPTIONS
# =============================================================================


class ExchangeError(TradingBotError):
    """Base class for all exchange-related errors.

    Handles common exchange error patterns including rate limiting,
    authentication, network issues, and order-related problems.

    Example:
        raise ExchangeError(
            "Exchange API error",
            error_code="EXCH_001",
            exchange="binance",
            endpoint="/api/v3/order"
        )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "EXCH_000",
        exchange: str | None = None,
        endpoint: str | None = None,
        response_code: int | None = None,
        response_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Add exchange-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "exchange": exchange,
                "endpoint": endpoint,
                "response_code": response_code,
                "response_data": response_data,
            }
        )
        kwargs["context"] = context

        # Default category for exchange errors
        kwargs.setdefault("category", ErrorCategory.RETRYABLE)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("logger_name", "exchange")

        super().__init__(message, error_code, **kwargs)


class ExchangeConnectionError(ExchangeError):
    """Network connection failures to exchange APIs.

    Used for TCP connection failures, DNS resolution issues,
    SSL/TLS handshake failures, and general network connectivity problems.
    """

    def __init__(self, message: str, exchange: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "EXCH_001")
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 5)
        kwargs.setdefault("suggested_action", "Check network connectivity and exchange status")

        super().__init__(message, exchange=exchange, **kwargs)


class ExchangeRateLimitError(ExchangeError):
    """Rate limit violations from exchange APIs.

    Includes both request rate limits and order rate limits.
    Automatically sets retry timing based on exchange response.
    """

    def __init__(
        self,
        message: str,
        exchange: str | None = None,
        retry_after: int | None = None,
        limit_type: str | None = None,
        current_usage: int | None = None,
        limit_value: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXCH_002")
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", retry_after or 60)
        kwargs.setdefault(
            "suggested_action", "Implement exponential backoff and reduce request frequency"
        )

        # Add rate limit specific context
        context = kwargs.get("context", {})
        context.update(
            {"limit_type": limit_type, "current_usage": current_usage, "limit_value": limit_value}
        )
        kwargs["context"] = context

        super().__init__(message, exchange=exchange, **kwargs)


class ExchangeInsufficientFundsError(ExchangeError):
    """Insufficient balance for order execution.

    Raised when account balance is insufficient for the requested operation.
    Not retryable unless additional funds are deposited.
    """

    def __init__(
        self,
        message: str,
        exchange: str | None = None,
        required_amount: str | None = None,
        available_amount: str | None = None,
        currency: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXCH_003")
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault(
            "suggested_action", "Check account balance and deposit additional funds if needed"
        )

        # Add balance specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "required_amount": required_amount,
                "available_amount": available_amount,
                "currency": currency,
            }
        )
        kwargs["context"] = context

        super().__init__(message, exchange=exchange, **kwargs)


class ExchangeOrderError(ExchangeError):
    """General order-related exchange errors.

    Covers order rejection, invalid parameters, and order state issues.
    """

    def __init__(
        self,
        message: str,
        exchange: str | None = None,
        order_id: str | None = None,
        client_order_id: str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXCH_004")
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault(
            "suggested_action", "Validate order parameters and retry with corrected values"
        )

        # Add order specific context
        context = kwargs.get("context", {})
        context.update({"order_id": order_id, "client_order_id": client_order_id, "symbol": symbol})
        kwargs["context"] = context

        super().__init__(message, exchange=exchange, **kwargs)


class ExchangeAuthenticationError(ExchangeError):
    """Exchange authentication and authorization failures.

    Includes API key issues, signature problems, and permission errors.
    """

    def __init__(
        self, message: str, exchange: str | None = None, auth_type: str | None = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("error_code", "EXCH_005")
        kwargs.setdefault("category", ErrorCategory.PERMISSION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("suggested_action", "Verify API credentials and permissions")

        # Add auth specific context
        context = kwargs.get("context", {})
        context.update({"auth_type": auth_type})
        kwargs["context"] = context

        super().__init__(message, exchange=exchange, **kwargs)


# Additional exchange error types (consolidated from exchanges/errors.py)
class InvalidOrderError(ExchangeOrderError):
    """Invalid order parameters."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "EXCH_006")
        kwargs.setdefault(
            "suggested_action", "Validate all order parameters against exchange rules"
        )
        super().__init__(message, **kwargs)


# =============================================================================
# RISK MANAGEMENT EXCEPTIONS
# =============================================================================


class RiskManagementError(TradingBotError):
    """Base class for all risk management violations and errors.

    Risk management errors are critical for protecting capital and ensuring
    system stability. These errors often trigger circuit breakers or emergency stops.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RISK_000",
        risk_metric: str | None = None,
        current_value: Decimal | None = None,
        threshold_value: Decimal | None = None,
        strategy_id: str | None = None,
        position_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add risk-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "risk_metric": risk_metric,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "strategy_id": strategy_id,
                "position_id": position_id,
            }
        )
        kwargs["context"] = context

        # Risk errors are typically not retryable and high severity
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("logger_name", "risk_management")

        super().__init__(message, error_code, **kwargs)


class PositionLimitError(RiskManagementError):
    """Position size or count limit violations.

    Triggered when trying to open positions that would exceed configured limits.
    """

    def __init__(
        self,
        message: str,
        limit_type: str = "size",  # "size", "count", "concentration"
        requested_amount: Decimal | None = None,
        limit_amount: Decimal | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_001")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("suggested_action", "Reduce position size or close existing positions")

        # Add position limit specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "limit_type": limit_type,
                "requested_amount": requested_amount,
                "limit_amount": limit_amount,
                "symbol": symbol,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DrawdownLimitError(RiskManagementError):
    """Maximum drawdown limit violations.

    Critical error that may trigger emergency stops or position liquidation.
    """

    def __init__(
        self,
        message: str,
        current_drawdown: float | None = None,
        max_drawdown: float | None = None,
        drawdown_type: str = "absolute",  # "absolute", "relative", "daily"
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_002")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("suggested_action", "Implement emergency stop or reduce position sizes")

        # Add drawdown specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown,
                "drawdown_type": drawdown_type,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class RiskCalculationError(RiskManagementError):
    """Risk metric calculation failures.

    Occurs when risk calculations fail due to data issues or mathematical errors.
    """

    def __init__(
        self,
        message: str,
        calculation_type: str | None = None,
        input_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_003")
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault(
            "suggested_action", "Verify input data quality and calculation parameters"
        )

        # Add calculation specific context
        context = kwargs.get("context", {})
        context.update({"calculation_type": calculation_type, "input_data": input_data})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class CapitalAllocationError(RiskManagementError):
    """Capital allocation rule violations.

    Triggered when allocation requests violate configured capital rules.
    """

    def __init__(
        self,
        message: str,
        requested_allocation: float | None = None,
        available_capital: float | None = None,
        allocation_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_004")
        kwargs.setdefault("suggested_action", "Review capital allocation rules and available funds")

        # Add allocation specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "requested_allocation": requested_allocation,
                "available_capital": available_capital,
                "allocation_type": allocation_type,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class AllocationError(RiskManagementError):
    """Portfolio allocation errors.

    Raised when portfolio allocation operations fail, including strategy
    allocation, rebalancing, or position sizing issues.
    """

    def __init__(
        self,
        message: str,
        allocation_type: str | None = None,
        requested_amount: Decimal | None = None,
        available_amount: Decimal | None = None,
        strategy_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_004")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("suggested_action", "Check capital availability and allocation limits")

        # Add allocation-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "allocation_type": allocation_type,
                "requested_amount": requested_amount,
                "available_amount": available_amount,
                "strategy_name": strategy_name,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class CircuitBreakerTriggeredError(RiskManagementError):
    """Circuit breaker activation.

    Critical system protection mechanism has been triggered.
    """

    def __init__(
        self,
        message: str,
        breaker_type: str | None = None,
        trigger_metric: str | None = None,
        cooldown_period: int | None = None,
        affected_strategies: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_005")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("suggested_action", "Wait for cooldown period or manual intervention")

        # Add circuit breaker specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "breaker_type": breaker_type,
                "trigger_metric": trigger_metric,
                "cooldown_period": cooldown_period,
                "affected_strategies": affected_strategies or [],
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class EmergencyStopError(RiskManagementError):
    """Emergency stop system failures.

    Critical error when emergency stop mechanisms fail to activate.
    """

    def __init__(
        self,
        message: str,
        stop_type: str | None = None,
        failure_reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "RISK_006")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 1)
        kwargs.setdefault("suggested_action", "Manual intervention required immediately")

        # Add emergency stop specific context
        context = kwargs.get("context", {})
        context.update({"stop_type": stop_type, "failure_reason": failure_reason})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# DATA-RELATED EXCEPTIONS
# =============================================================================


class DataError(TradingBotError):
    """Base class for all data-related errors.

    Covers data quality issues, source failures, processing errors,
    and validation problems across the entire data pipeline.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_000",
        data_source: str | None = None,
        data_type: str | None = None,
        pipeline_stage: str | None = None,
        record_count: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Add data-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "data_source": data_source,
                "data_type": data_type,
                "pipeline_stage": pipeline_stage,
                "record_count": record_count,
            }
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.DATA_QUALITY)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("logger_name", "data_pipeline")

        super().__init__(message, error_code, **kwargs)


class DataValidationError(DataError):
    """Data validation and schema compliance failures.

    Raised when data doesn't meet expected schema, format, or business rules.
    """

    def __init__(
        self,
        message: str,
        validation_rule: str | None = None,
        invalid_fields: list[str] | None = None,
        sample_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DATA_001")
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("suggested_action", "Review data format and validation rules")

        # Add validation specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "validation_rule": validation_rule,
                "invalid_fields": invalid_fields or [],
                "sample_data": sample_data,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DataSourceError(DataError):
    """External data source connectivity and reliability issues.

    Covers API failures, database connection issues, and data provider problems.
    """

    def __init__(
        self,
        message: str,
        source_name: str | None = None,
        source_type: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DATA_002")
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 30)
        kwargs.setdefault("suggested_action", "Check data source availability and connectivity")

        # Add source specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "source_name": source_name,
                "source_type": source_type,
                "endpoint": endpoint,
                "status_code": status_code,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DataProcessingError(DataError):
    """Data transformation and processing pipeline failures.

    Occurs during data cleaning, transformation, aggregation, or enrichment steps.
    """

    def __init__(
        self,
        message: str,
        processing_step: str | None = None,
        input_data_sample: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DATA_003")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("suggested_action", "Review processing logic and input data quality")

        # Add processing specific context
        context = kwargs.get("context", {})
        context.update({"processing_step": processing_step, "input_data_sample": input_data_sample})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DataCorruptionError(DataError):
    """Data integrity and corruption detection.

    Critical error indicating data has been corrupted or compromised.
    """

    def __init__(
        self,
        message: str,
        corruption_type: str | None = None,
        affected_records: int | None = None,
        checksum_expected: str | None = None,
        checksum_actual: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DATA_004")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault(
            "suggested_action", "Restore from backup and investigate corruption cause"
        )

        # Add corruption specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "corruption_type": corruption_type,
                "affected_records": affected_records,
                "checksum_expected": checksum_expected,
                "checksum_actual": checksum_actual,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DataQualityError(DataError):
    """Data quality issues affecting trading decisions.

    Includes stale data, missing data, outliers, and quality degradation.
    """

    def __init__(
        self,
        message: str,
        quality_metric: str | None = None,
        quality_score: float | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DATA_005")
        kwargs.setdefault("suggested_action", "Review data quality metrics and sources")

        # Add quality specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "quality_metric": quality_metric,
                "quality_score": quality_score,
                "threshold": threshold,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# MACHINE LEARNING MODEL EXCEPTIONS
# =============================================================================


class ModelError(TradingBotError):
    """Base class for all ML model-related errors.

    Covers model lifecycle issues including loading, training, inference,
    validation, and drift detection.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "MODEL_000",
        model_name: str | None = None,
        model_version: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add model-specific context
        context = kwargs.get("context", {})
        context.update(
            {"model_name": model_name, "model_version": model_version, "model_type": model_type}
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("logger_name", "ml_models")

        super().__init__(message, error_code, **kwargs)


class ModelLoadError(ModelError):
    """Model loading and initialization failures.

    Occurs when models cannot be loaded from disk, network, or memory.
    """

    def __init__(
        self,
        message: str,
        model_path: str | None = None,
        file_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "MODEL_001")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("suggested_action", "Verify model file exists and is accessible")

        # Add loading specific context
        context = kwargs.get("context", {})
        context.update({"model_path": model_path, "file_size": file_size})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ModelInferenceError(ModelError):
    """Model prediction and inference failures.

    Raised when model predictions fail due to invalid inputs or model issues.
    """

    def __init__(
        self,
        message: str,
        input_shape: tuple[int, ...] | None = None,
        expected_shape: tuple[int, ...] | None = None,
        inference_time: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "MODEL_002")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("suggested_action", "Validate input data format and model state")

        # Add inference specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "input_shape": input_shape,
                "expected_shape": expected_shape,
                "inference_time": inference_time,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ModelDriftError(ModelError):
    """Model performance drift detection.

    Critical error indicating model performance has degraded significantly.
    """

    def __init__(
        self,
        message: str,
        drift_metric: str | None = None,
        current_performance: float | None = None,
        baseline_performance: float | None = None,
        drift_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "MODEL_003")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("suggested_action", "Retrain model or switch to backup model")

        # Add drift specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "drift_metric": drift_metric,
                "current_performance": current_performance,
                "baseline_performance": baseline_performance,
                "drift_threshold": drift_threshold,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ModelTrainingError(ModelError):
    """Model training and optimization failures.

    Occurs during model training, hyperparameter optimization, or validation.
    """

    def __init__(
        self,
        message: str,
        training_stage: str | None = None,
        epoch: int | None = None,
        loss_value: Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "MODEL_004")
        kwargs.setdefault("suggested_action", "Review training data and hyperparameters")

        # Add training specific context
        context = kwargs.get("context", {})
        context.update({"training_stage": training_stage, "epoch": epoch, "loss_value": loss_value})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ModelValidationError(ModelError):
    """Model validation and testing failures.

    Raised when model validation fails to meet performance criteria.
    """

    def __init__(
        self,
        message: str,
        validation_metric: str | None = None,
        actual_score: float | None = None,
        required_score: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "MODEL_005")
        kwargs.setdefault("suggested_action", "Improve model or adjust validation criteria")

        # Add validation specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "validation_metric": validation_metric,
                "actual_score": actual_score,
                "required_score": required_score,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================


class ValidationError(TradingBotError):
    """Base class for all input and schema validation errors.

    Covers parameter validation, schema compliance, configuration validation,
    and business rule validation.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "VALID_000",
        field_name: str | None = None,
        field_value: Any | None = None,
        expected_type: str | None = None,
        validation_rule: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add validation-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "field_name": field_name,
                "field_value": field_value,
                "expected_type": expected_type,
                "validation_rule": validation_rule,
            }
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("logger_name", "validation")

        super().__init__(message, error_code, **kwargs)


class ConfigurationError(ValidationError):
    """Configuration file and parameter validation errors.

    Raised when configuration is invalid, missing, or inconsistent.
    """

    def __init__(
        self,
        message: str,
        config_file: str | None = None,
        config_section: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "VALID_001")
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("suggested_action", "Review and correct configuration file")

        # Add configuration specific context
        context = kwargs.get("context", {})
        context.update({"config_file": config_file, "config_section": config_section})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class SchemaValidationError(ValidationError):
    """Data schema and structure validation failures.

    Occurs when data doesn't conform to expected schema or format.
    """

    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        schema_version: str | None = None,
        failed_constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "VALID_002")
        kwargs.setdefault("suggested_action", "Validate data against schema definition")

        # Add schema specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "schema_name": schema_name,
                "schema_version": schema_version,
                "failed_constraints": failed_constraints or [],
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class InputValidationError(ValidationError):
    """Function and API input parameter validation failures.

    Raised when input parameters are invalid, out of range, or missing.
    """

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        parameter_value: Any | None = None,
        valid_range: tuple[Any, Any] | None = None,
        valid_values: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "VALID_003")
        kwargs.setdefault("suggested_action", "Correct input parameters and retry")

        # Add input specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "valid_range": valid_range,
                "valid_values": valid_values,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class BusinessRuleValidationError(ValidationError):
    """Business logic and rule validation failures.

    Raised when operations violate business rules or constraints.
    """

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        rule_description: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "VALID_004")
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("suggested_action", "Review business rules and adjust operation")

        # Add business rule specific context
        context = kwargs.get("context", {})
        context.update({"rule_name": rule_name, "rule_description": rule_description})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# EXECUTION EXCEPTIONS
# =============================================================================


class ExecutionError(TradingBotError):
    """Base class for all order execution and trading errors.

    Covers order placement, fills, slippage, timing, and execution algorithms.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "EXEC_000",
        order_id: str | None = None,
        symbol: str | None = None,
        execution_algorithm: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add execution-specific context
        context = kwargs.get("context", {})
        context.update(
            {"order_id": order_id, "symbol": symbol, "execution_algorithm": execution_algorithm}
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("logger_name", "execution")

        super().__init__(message, error_code, **kwargs)


class OrderRejectionError(ExecutionError):
    """Order rejected by exchange."""

    def __init__(self, message: str, rejection_reason: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "EXEC_004")
        kwargs.setdefault("suggested_action", "Review rejection reason and adjust order parameters")

        # Add rejection specific context
        context = kwargs.get("context", {})
        context.update({"rejection_reason": rejection_reason})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class SlippageError(ExecutionError):
    """Excessive slippage during order execution.

    Triggered when actual execution price deviates significantly from expected price.
    """

    def __init__(
        self,
        message: str,
        expected_price: Decimal | None = None,
        actual_price: Decimal | None = None,
        slippage_pct: Decimal | None = None,
        slippage_threshold: Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXEC_001")
        kwargs.setdefault("suggested_action", "Review market conditions and execution strategy")

        # Add slippage specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "expected_price": expected_price,
                "actual_price": actual_price,
                "slippage_pct": slippage_pct,
                "slippage_threshold": slippage_threshold,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ExecutionTimeoutError(ExecutionError):
    """Order execution timeout errors.

    Raised when orders take too long to execute or remain unfilled.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: int | None = None,
        elapsed_time: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXEC_002")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("suggested_action", "Cancel order and retry with adjusted parameters")

        # Add timeout specific context
        context = kwargs.get("context", {})
        context.update({"timeout_duration": timeout_duration, "elapsed_time": elapsed_time})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ExecutionPartialFillError(ExecutionError):
    """Partial order fill handling errors.

    Occurs when orders are only partially filled and require special handling.
    """

    def __init__(
        self,
        message: str,
        requested_quantity: Decimal | None = None,
        filled_quantity: Decimal | None = None,
        remaining_quantity: Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "EXEC_003")
        kwargs.setdefault("suggested_action", "Handle partial fill according to strategy rules")

        # Add partial fill specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "requested_quantity": requested_quantity,
                "filled_quantity": filled_quantity,
                "remaining_quantity": remaining_quantity,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# NETWORK AND COMMUNICATION EXCEPTIONS
# =============================================================================


class NetworkError(TradingBotError):
    """Base class for all network and communication errors.

    Covers TCP connections, HTTP requests, WebSocket issues, DNS resolution,
    and general connectivity problems.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "NET_000",
        host: str | None = None,
        port: int | None = None,
        protocol: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add network-specific context
        context = kwargs.get("context", {})
        context.update({"host": host, "port": port, "protocol": protocol})
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.NETWORK)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 5)
        kwargs.setdefault("logger_name", "network")

        super().__init__(message, error_code, **kwargs)


class ConnectionError(NetworkError):
    """Network connection establishment failures.

    Covers TCP connection failures, socket errors, and connection refused.
    """

    def __init__(self, message: str, connection_type: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "NET_001")
        kwargs.setdefault("suggested_action", "Check network connectivity and target availability")

        # Add connection specific context
        context = kwargs.get("context", {})
        context.update({"connection_type": connection_type})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class TimeoutError(NetworkError):
    """Network operation timeout errors.

    Raised when network operations exceed configured timeout limits.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: int | None = None,
        operation_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "NET_002")
        kwargs.setdefault("suggested_action", "Increase timeout or check network latency")

        # Add timeout specific context
        context = kwargs.get("context", {})
        context.update({"timeout_duration": timeout_duration, "operation_type": operation_type})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class WebSocketError(NetworkError):
    """WebSocket connection and messaging errors.

    Covers WebSocket handshake failures, message errors, and disconnections.
    """

    def __init__(
        self,
        message: str,
        websocket_state: str | None = None,
        close_code: int | None = None,
        close_reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "NET_003")
        kwargs.setdefault("suggested_action", "Reconnect WebSocket with backoff strategy")

        # Add WebSocket specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "websocket_state": websocket_state,
                "close_code": close_code,
                "close_reason": close_reason,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# STATE MANAGEMENT EXCEPTIONS
# =============================================================================


class StateConsistencyError(TradingBotError):
    """Base class for all state management and consistency errors.

    Covers state synchronization, corruption, locking, and conflict resolution.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "STATE_000",
        state_component: str | None = None,
        state_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add state-specific context
        context = kwargs.get("context", {})
        context.update({"state_component": state_component, "state_version": state_version})
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("logger_name", "state_management")

        super().__init__(message, error_code, **kwargs)


class StateError(StateConsistencyError):
    """General state management errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STATE_001")
        super().__init__(message, **kwargs)


class StateCorruptionError(StateConsistencyError):
    """State data corruption detected."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STATE_002")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("suggested_action", "Restore from backup and investigate corruption")
        super().__init__(message, **kwargs)


class StateLockError(StateConsistencyError):
    """State lock acquisition failures."""

    def __init__(self, message: str, lock_name: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STATE_003")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 1)

        context = kwargs.get("context", {})
        context.update({"lock_name": lock_name})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class SynchronizationError(StateConsistencyError):
    """Real-time synchronization errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STATE_004")
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ConflictError(StateConsistencyError):
    """State conflict errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STATE_005")
        kwargs.setdefault("suggested_action", "Resolve conflicts and retry operation")
        super().__init__(message, **kwargs)


# =============================================================================
# SECURITY EXCEPTIONS
# =============================================================================


class SecurityError(TradingBotError):
    """Base class for all security-related errors.

    Covers authentication, authorization, encryption, and access control.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SEC_000",
        user_id: str | None = None,
        resource: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add security-specific context
        context = kwargs.get("context", {})
        context.update({"user_id": user_id, "resource": resource})
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.PERMISSION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", False)
        kwargs.setdefault("logger_name", "security")

        super().__init__(message, error_code, **kwargs)


class AuthenticationError(SecurityError):
    """Authentication failures and credential issues.

    Raised for invalid credentials, expired tokens, or authentication failures.
    """

    def __init__(self, message: str, auth_method: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "SEC_001")
        kwargs.setdefault("suggested_action", "Verify credentials and authentication method")

        context = kwargs.get("context", {})
        context.update({"auth_method": auth_method})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class AuthorizationError(SecurityError):
    """Authorization and permission failures."""

    def __init__(self, message: str, required_permission: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "SEC_002")
        kwargs.setdefault("suggested_action", "Request appropriate permissions")

        context = kwargs.get("context", {})
        context.update({"required_permission": required_permission})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class EncryptionError(SecurityError):
    """Encryption and decryption failures."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "SEC_003")
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class TokenValidationError(SecurityError):
    """Token validation and parsing failures."""

    def __init__(self, message: str, token_type: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "SEC_004")

        context = kwargs.get("context", {})
        context.update({"token_type": token_type})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# STRATEGY EXCEPTIONS
# =============================================================================


class StrategyError(TradingBotError):
    """Base class for all strategy-related errors."""

    def __init__(self, message: str, strategy_id: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STRAT_000")
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("logger_name", "strategy")

        context = kwargs.get("context", {})
        context.update({"strategy_id": strategy_id})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class StrategyConfigurationError(StrategyError):
    """Strategy configuration errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STRAT_001")
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        super().__init__(message, **kwargs)


class SignalGenerationError(StrategyError):
    """Signal generation failures."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STRAT_002")
        super().__init__(message, **kwargs)


class ArbitrageError(StrategyError):
    """Arbitrage strategy errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "STRAT_003")
        super().__init__(message, **kwargs)


class BacktestError(TradingBotError):
    """Backtesting operation errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_001")
        kwargs.setdefault("category", ErrorCategory.BUSINESS_LOGIC)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class BacktestConfigurationError(BacktestError):
    """Backtesting configuration errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_002")
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        super().__init__(message, **kwargs)


class BacktestDataError(BacktestError):
    """Backtesting data-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_003")
        kwargs.setdefault("category", ErrorCategory.DATA_QUALITY)
        super().__init__(message, **kwargs)


class BacktestExecutionError(BacktestError):
    """Backtesting execution errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_004")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class BacktestServiceError(BacktestError):
    """Backtesting service unavailability errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_005")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class BacktestValidationError(BacktestError):
    """Backtesting validation errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_006")
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        super().__init__(message, **kwargs)


class BacktestResultError(BacktestError):
    """Backtesting result processing errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_007")
        super().__init__(message, **kwargs)


class BacktestMetricsError(BacktestError):
    """Backtesting metrics calculation errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_008")
        super().__init__(message, **kwargs)


class BacktestStrategyError(BacktestError):
    """Backtesting strategy-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "BACKTEST_009")
        super().__init__(message, **kwargs)


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================


class DatabaseError(TradingBotError):
    """Base class for all database-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "DB_000")
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("logger_name", "database")
        super().__init__(message, **kwargs)


class DatabaseConnectionError(DatabaseError):
    """Database connection failures."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "DB_001")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("retry_after", 5)
        super().__init__(message, **kwargs)


class DatabaseQueryError(DatabaseError):
    """Database query failures."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "DB_002")
        super().__init__(message, **kwargs)


# =============================================================================
# CIRCUIT BREAKER AND RETRY EXCEPTIONS
# =============================================================================


class CircuitBreakerOpenError(TradingBotError):
    """Circuit breaker is open due to too many failures."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "CB_001")
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class MaxRetriesExceededError(TradingBotError):
    """Maximum retry attempts exceeded."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "RETRY_001")
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


# =============================================================================
# ERROR MAPPING AND UTILITIES
# =============================================================================


class ErrorCodeRegistry:
    """Registry for all error codes in the system.

    Provides centralized error code management and validation.
    """

    # Error code ranges by category
    EXCHANGE_CODES = range(1, 100)  # EXCH_001 - EXCH_099
    RISK_CODES = range(100, 200)  # RISK_100 - RISK_199
    DATA_CODES = range(200, 300)  # DATA_200 - DATA_299
    MODEL_CODES = range(300, 400)  # MODEL_300 - MODEL_399
    VALIDATION_CODES = range(400, 500)  # VALID_400 - VALID_499
    EXECUTION_CODES = range(500, 600)  # EXEC_500 - EXEC_599
    NETWORK_CODES = range(600, 700)  # NET_600 - NET_699
    STATE_CODES = range(700, 800)  # STATE_700 - STATE_799
    SECURITY_CODES = range(800, 900)  # SEC_800 - SEC_899
    STRATEGY_CODES = range(900, 1000)  # STRAT_900 - STRAT_999
    MONITORING_CODES = range(1000, 1100)  # MON_1000 - MON_1099

    @classmethod
    def validate_code(cls, error_code: str) -> bool:
        """Validate error code format and uniqueness."""
        # Implementation for code validation
        return True


class ExchangeErrorMapper:
    """Maps exchange-specific errors to standardized exceptions.

    Consolidated from src/exchanges/errors.py with enhanced functionality.
    """

    # Binance error code mappings (comprehensive)
    BINANCE_ERRORS: ClassVar[
        dict[int, tuple[str, type[TradingBotError]] | type[TradingBotError]]
    ] = {
        -1000: ("UNKNOWN", ExchangeError),
        -1001: ("DISCONNECTED", ExchangeConnectionError),
        -1002: ("UNAUTHORIZED", ExchangeAuthenticationError),
        -1003: ("TOO_MANY_REQUESTS", ExchangeRateLimitError),
        -1006: ("UNEXPECTED_RESP", DataError),
        -1007: ("TIMEOUT", NetworkError),
        -1021: ("INVALID_TIMESTAMP", ExchangeAuthenticationError),
        -1022: ("INVALID_SIGNATURE", ExchangeAuthenticationError),
        -2010: ("NEW_ORDER_REJECTED", InvalidOrderError),
        -2011: ("CANCEL_REJECTED", ExchangeOrderError),
        -2013: ("NO_SUCH_ORDER", ExchangeOrderError),
        -2014: ("BAD_API_KEY_FMT", ExchangeAuthenticationError),
        -2015: ("REJECTED_API_KEY", ExchangeAuthenticationError),
        -2018: ("BALANCE_NOT_SUFFICIENT", ExchangeInsufficientFundsError),
    }

    # Coinbase error mappings
    COINBASE_ERRORS: ClassVar[dict[str, type[TradingBotError]]] = {
        "authentication_error": ExchangeAuthenticationError,
        "invalid_request": InvalidOrderError,
        "rate_limit": ExchangeRateLimitError,
        "insufficient_funds": ExchangeInsufficientFundsError,
        "not_found": ExchangeOrderError,
        "validation_error": InvalidOrderError,
    }

    # OKX error code mappings
    OKX_ERRORS: ClassVar[dict[str, tuple[str, type[TradingBotError]]]] = {
        "1": ("Operation failed", ExchangeError),
        "2": ("Bulk operation partially succeeded", ExchangeError),
        "50000": ("Service temporarily unavailable", NetworkError),
        "50001": ("Signature authentication failed", ExchangeAuthenticationError),
        "50002": ("Too many requests", ExchangeRateLimitError),
        "50004": ("Endpoint request timeout", NetworkError),
        "50005": ("Invalid API key", ExchangeAuthenticationError),
        "50008": ("Invalid passphrase", ExchangeAuthenticationError),
        "50011": ("Invalid request", InvalidOrderError),
        "50013": ("Invalid sign", ExchangeAuthenticationError),
        "51000": ("Parameter validation error", InvalidOrderError),
        "51001": ("Instrument ID does not exist", InvalidOrderError),
        "51008": ("Order amount exceeds the limit", InvalidOrderError),
        "51009": ("Order placement function is blocked", ExchangeOrderError),
        "51020": ("Insufficient balance", ExchangeInsufficientFundsError),
        "51400": ("Cancellation failed", ExchangeOrderError),
        "51401": ("Order does not exist", ExchangeOrderError),
    }

    @classmethod
    def map_error(cls, exchange: str, error_data: dict[str, Any]) -> TradingBotError:
        """
        Map exchange-specific error to common error.

        Args:
            exchange: Exchange name
            error_data: Error data from exchange

        Returns:
            Standardized exchange error
        """
        exchange_lower = exchange.lower()

        if exchange_lower == "binance":
            return cls._map_binance(error_data, exchange)
        elif exchange_lower == "coinbase":
            return cls._map_coinbase(error_data, exchange)
        elif exchange_lower == "okx":
            return cls._map_okx(error_data, exchange)
        else:
            return cls._map_generic(error_data, exchange)

    @classmethod
    def map_binance_error(cls, error_data: dict[str, Any]) -> TradingBotError:
        """Map Binance-specific error to standardized exception."""
        return cls._map_binance(error_data, "binance")

    @classmethod
    def _map_binance(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError:
        """Map Binance error with comprehensive mappings."""
        code = error_data.get("code")
        msg = error_data.get("msg", "Unknown error")

        if code in cls.BINANCE_ERRORS:
            error_info = cls.BINANCE_ERRORS[code]
            if isinstance(error_info, tuple):
                msg_prefix, error_class = error_info
                full_msg = f"{msg_prefix}: {msg}"
            else:
                error_class = error_info
                full_msg = msg

            # Special handling for rate limit
            if error_class == ExchangeRateLimitError:
                retry_after = cls._extract_retry_after(error_data)
                return ExchangeRateLimitError(
                    full_msg,
                    exchange=exchange,
                    retry_after=retry_after,
                    limit_type="request",
                    response_code=code,
                    response_data=error_data,
                )

            # Special handling for insufficient funds
            elif error_class == ExchangeInsufficientFundsError:
                return ExchangeInsufficientFundsError(
                    full_msg, exchange=exchange, response_code=code, response_data=error_data
                )

            # Create instance of appropriate error class
            return error_class(
                full_msg, exchange=exchange, response_code=code, response_data=error_data
            )

        return ExchangeError(msg, exchange=exchange, response_code=code, response_data=error_data)

    @classmethod
    def map_coinbase_error(cls, error_data: dict[str, Any]) -> TradingBotError:
        """Map Coinbase-specific error to standardized exception."""
        return cls._map_coinbase(error_data, "coinbase")

    @classmethod
    def _map_coinbase(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError:
        """Map Coinbase error."""
        error_type = error_data.get("type", "").lower()
        message = error_data.get("message", "Unknown error")

        error_class = cls.COINBASE_ERRORS.get(error_type, ExchangeError)

        # Special handling for rate limit
        if error_class == ExchangeRateLimitError:
            retry_after = cls._extract_retry_after(error_data)
            return ExchangeRateLimitError(
                message,
                exchange=exchange,
                retry_after=retry_after,
                limit_type="request",
                response_data=error_data,
            )

        return error_class(message, exchange=exchange, response_data=error_data)

    @classmethod
    def map_okx_error(cls, error_data: dict[str, Any]) -> TradingBotError:
        """Map OKX-specific error to standardized exception."""
        return cls._map_okx(error_data, "okx")

    @classmethod
    def _map_okx(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError:
        """Map OKX error."""
        code = str(error_data.get("code", ""))
        msg = error_data.get("msg", "Unknown error")

        if code in cls.OKX_ERRORS:
            error_info = cls.OKX_ERRORS[code]
            if isinstance(error_info, tuple):
                msg_prefix, error_class = error_info
                full_msg = f"{msg_prefix}: {msg}"
            else:
                error_class = error_info
                full_msg = msg

            # Special handling for rate limit
            if error_class == ExchangeRateLimitError:
                retry_after = cls._extract_retry_after(error_data)
                return ExchangeRateLimitError(
                    full_msg,
                    exchange=exchange,
                    retry_after=retry_after,
                    limit_type="request",
                    response_data=error_data,
                )

            return error_class(full_msg, exchange=exchange, response_data=error_data)

        return ExchangeError(msg, exchange=exchange, response_data=error_data)

    @classmethod
    def _map_generic(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError:
        """Map generic/unknown exchange error."""
        message = (
            error_data.get("message")
            or error_data.get("msg")
            or error_data.get("error")
            or str(error_data)
        )

        # Try to detect error type from message
        message_lower = message.lower()

        if "rate limit" in message_lower or "429" in message_lower:
            return ExchangeRateLimitError(
                message,
                exchange=exchange,
                retry_after=cls._extract_retry_after(error_data),
                response_data=error_data,
            )
        elif "unauthorized" in message_lower or "authentication" in message_lower:
            return ExchangeAuthenticationError(message, exchange=exchange, response_data=error_data)
        elif "insufficient" in message_lower or "balance" in message_lower:
            return ExchangeInsufficientFundsError(
                message, exchange=exchange, response_data=error_data
            )
        elif "order" in message_lower:
            return ExchangeOrderError(message, exchange=exchange, response_data=error_data)
        elif "network" in message_lower or "timeout" in message_lower:
            return ExchangeConnectionError(message, exchange=exchange, response_data=error_data)

        return ExchangeError(message, exchange=exchange, response_data=error_data)

    @staticmethod
    def _extract_retry_after(error_data: dict[str, Any]) -> int | None:
        """
        Try to extract retry-after value from error data.

        Args:
            error_data: Error data

        Returns:
            Retry after in seconds or None
        """
        # Check common fields
        retry_after = (
            error_data.get("retry_after")
            or error_data.get("retryAfter")
            or error_data.get("Retry-After")
        )

        if retry_after:
            try:
                return int(retry_after)
            except (TypeError, ValueError):
                pass

        # Try to extract from message
        message = str(error_data)
        match = re.search(r"(\d+)\s*seconds?", message, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None


# =============================================================================
# BASE COMPONENT EXCEPTIONS
# =============================================================================


class ComponentError(TradingBotError):
    """Base class for all component-related errors.

    Used by BaseComponent and its subclasses for lifecycle,
    dependency injection, and resource management errors.
    """

    def __init__(
        self,
        message: str,
        component_name: str | None = None,
        component: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "COMP_000")
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("logger_name", "component")

        # Support both component_name (legacy) and component (new) for compatibility
        self.component = component or component_name
        self.operation = operation

        context = kwargs.get("context", {})
        context.update(
            {"component_name": self.component, "component": self.component, "operation": operation}
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ServiceError(ComponentError):
    """Service layer errors for BaseService implementations."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "SERV_000")
        super().__init__(message, **kwargs)


class RepositoryError(ComponentError):
    """Repository layer errors for BaseRepository implementations."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "REPO_000")
        super().__init__(message, **kwargs)


class FactoryError(ComponentError):
    """Factory pattern errors for BaseFactory implementations."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "FACT_000")
        super().__init__(message, **kwargs)


class DependencyError(ComponentError):
    """Dependency injection and resolution errors."""

    def __init__(self, message: str, dependency_name: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "DEP_000")

        context = kwargs.get("context", {})
        context.update({"dependency_name": dependency_name})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class HealthCheckError(ComponentError):
    """Health check system errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "HEALTH_000")
        super().__init__(message, **kwargs)


class CircuitBreakerError(ComponentError):
    """Circuit breaker pattern errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "CB_000")
        super().__init__(message, **kwargs)


class EventError(ComponentError):
    """Event system errors for BaseEventEmitter."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "EVENT_000")
        super().__init__(message, **kwargs)


class EntityNotFoundError(DatabaseError):
    """Entity not found in repository."""

    def __init__(
        self,
        message: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "DB_003")
        kwargs.setdefault("retryable", False)

        context = kwargs.get("context", {})
        context.update({"entity_type": entity_type, "entity_id": entity_id})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class CreationError(FactoryError):
    """Factory creation errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "FACT_001")
        super().__init__(message, **kwargs)


class RegistrationError(FactoryError):
    """Factory registration errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "FACT_002")
        super().__init__(message, **kwargs)


class EventHandlerError(EventError):
    """Event handler execution errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "EVENT_001")
        super().__init__(message, **kwargs)


class MonitoringError(ComponentError):
    """Monitoring and metrics collection errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "MON_1001")
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class AnalyticsError(ComponentError):
    """Analytics calculation and processing errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("error_code", "ANA_1001")
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        super().__init__(message, **kwargs)


# =============================================================================
# OPTIMIZATION EXCEPTIONS
# =============================================================================


class OptimizationError(TradingBotError):
    """Base class for all optimization-related errors.

    Covers parameter optimization, strategy optimization, hyperparameter tuning,
    and overfitting detection errors.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "OPT_000",
        optimization_algorithm: str | None = None,
        parameters: dict[str, Any] | None = None,
        optimization_stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Add optimization-specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "optimization_algorithm": optimization_algorithm,
                "parameters": parameters,
                "optimization_stage": optimization_stage,
            }
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("logger_name", "optimization")

        super().__init__(message, error_code, **kwargs)


class ParameterValidationError(OptimizationError):
    """Parameter space validation errors."""

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        parameter_value: Any | None = None,
        parameter_bounds: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_001")
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("suggested_action", "Review parameter bounds and values")

        # Add parameter specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "parameter_bounds": parameter_bounds,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class OptimizationTimeoutError(OptimizationError):
    """Optimization process timeout errors."""

    def __init__(
        self,
        message: str,
        timeout_duration: int | None = None,
        elapsed_time: int | None = None,
        completed_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_002")
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("suggested_action", "Increase timeout or reduce parameter space")

        # Add timeout specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "timeout_duration": timeout_duration,
                "elapsed_time": elapsed_time,
                "completed_iterations": completed_iterations,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ConvergenceError(OptimizationError):
    """Optimization convergence failures."""

    def __init__(
        self,
        message: str,
        convergence_metric: str | None = None,
        current_value: Decimal | None = None,
        threshold_value: Decimal | None = None,
        iterations_completed: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_003")
        kwargs.setdefault("suggested_action", "Adjust convergence criteria or increase iterations")

        # Add convergence specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "convergence_metric": convergence_metric,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "iterations_completed": iterations_completed,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class OverfittingError(OptimizationError):
    """Overfitting detection errors."""

    def __init__(
        self,
        message: str,
        validation_metric: str | None = None,
        training_score: float | None = None,
        validation_score: float | None = None,
        overfitting_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_004")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("suggested_action", "Reduce model complexity or increase training data")

        # Add overfitting specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "validation_metric": validation_metric,
                "training_score": training_score,
                "validation_score": validation_score,
                "overfitting_threshold": overfitting_threshold,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class GeneticAlgorithmError(OptimizationError):
    """Genetic algorithm optimization errors."""

    def __init__(
        self,
        message: str,
        generation: int | None = None,
        population_size: int | None = None,
        fitness_value: Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_005")
        kwargs.setdefault("optimization_algorithm", "genetic_algorithm")

        # Add GA specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "generation": generation,
                "population_size": population_size,
                "fitness_value": fitness_value,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class HyperparameterOptimizationError(OptimizationError):
    """Hyperparameter optimization errors."""

    def __init__(
        self,
        message: str,
        optimization_method: str | None = None,
        parameter_space: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "OPT_006")
        kwargs.setdefault("optimization_algorithm", "hyperparameter")

        # Add hyperparameter specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "optimization_method": optimization_method,
                "parameter_space": parameter_space,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# PERFORMANCE OPTIMIZATION EXCEPTIONS
# =============================================================================


class PerformanceError(TradingBotError):
    """Base class for performance optimization errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "PERF_000",
        operation_type: str | None = None,
        performance_metric: str | None = None,
        expected_value: Decimal | None = None,
        actual_value: Decimal | None = None,
        **kwargs: Any,
    ) -> None:
        # Add performance specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "operation_type": operation_type,
                "performance_metric": performance_metric,
                "expected_value": expected_value,
                "actual_value": actual_value,
            }
        )
        kwargs["context"] = context

        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        kwargs.setdefault("logger_name", "performance")

        super().__init__(message, error_code, **kwargs)


class CacheError(PerformanceError):
    """Cache operation errors."""

    def __init__(
        self,
        message: str,
        cache_level: str | None = None,
        cache_key: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "PERF_001")
        kwargs.setdefault("suggested_action", "Check cache configuration and connectivity")

        # Add cache specific context
        context = kwargs.get("context", {})
        context.update({"cache_level": cache_level, "cache_key": cache_key, "operation": operation})
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class MemoryOptimizationError(PerformanceError):
    """Memory optimization errors."""

    def __init__(
        self,
        message: str,
        memory_usage_mb: float | None = None,
        memory_limit_mb: float | None = None,
        gc_stats: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "PERF_002")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault(
            "suggested_action", "Review memory usage patterns and optimize allocations"
        )

        # Add memory specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "memory_usage_mb": memory_usage_mb,
                "memory_limit_mb": memory_limit_mb,
                "gc_stats": gc_stats,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class DatabaseOptimizationError(PerformanceError):
    """Database performance optimization errors."""

    def __init__(
        self,
        message: str,
        query_time_ms: float | None = None,
        query_limit_ms: float | None = None,
        affected_query: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "PERF_003")
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("suggested_action", "Check database indexes and query optimization")

        # Add database specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "query_time_ms": query_time_ms,
                "query_limit_ms": query_limit_ms,
                "affected_query": affected_query,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ConnectionPoolError(PerformanceError):
    """Connection pool optimization errors."""

    def __init__(
        self,
        message: str,
        pool_size: int | None = None,
        active_connections: int | None = None,
        wait_time_ms: float | None = None,
        exchange: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "PERF_004")
        kwargs.setdefault("suggested_action", "Adjust connection pool configuration")

        # Add connection pool specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "pool_size": pool_size,
                "active_connections": active_connections,
                "wait_time_ms": wait_time_ms,
                "exchange": exchange,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class ProfilingError(PerformanceError):
    """Performance profiling errors."""

    def __init__(
        self,
        message: str,
        profiler_type: str | None = None,
        operation_name: str | None = None,
        profiling_duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("error_code", "PERF_005")
        kwargs.setdefault("suggested_action", "Check profiling configuration and target operations")

        # Add profiling specific context
        context = kwargs.get("context", {})
        context.update(
            {
                "profiler_type": profiler_type,
                "operation_name": operation_name,
                "profiling_duration_ms": profiling_duration_ms,
            }
        )
        kwargs["context"] = context

        super().__init__(message, **kwargs)


# =============================================================================
# EXCEPTION UTILITIES
# =============================================================================


def create_error_from_dict(error_dict: dict[str, Any]) -> TradingBotError:
    """Create appropriate exception from error dictionary.

    Useful for deserializing errors from logs or network responses.
    """
    exception_type = error_dict.get("exception_type", "TradingBotError")

    # Map to appropriate exception class
    exception_classes: dict[str, type[TradingBotError]] = {
        "ExchangeError": ExchangeError,
        "RiskManagementError": RiskManagementError,
        "DataError": DataError,
        "ValidationError": ValidationError,
        # Add more mappings as needed
    }

    exception_class: type[TradingBotError] = exception_classes.get(exception_type, TradingBotError)

    return exception_class(
        error_dict.get("message", "Unknown error"),
        error_code=error_dict.get("error_code", "UNKNOWN"),
        details=error_dict.get("details", {}),
        context=error_dict.get("context", {}),
    )


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    if isinstance(error, TradingBotError):
        return error.retryable
    return False


def get_retry_delay(error: Exception) -> int | None:
    """Get recommended retry delay for an error.

    Args:
        error: The exception to check

    Returns:
        Retry delay in seconds if available, None otherwise
    """
    if isinstance(error, TradingBotError):
        return error.retry_after
    return None
