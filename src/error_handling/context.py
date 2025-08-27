"""
Unified error context definitions for the T-Bot trading system.

This module provides a single, comprehensive ErrorContext class that consolidates
all error context functionality previously scattered across multiple files.
"""

import inspect
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class ErrorSeverity(Enum):
    """Error severity levels for classification and escalation."""

    CRITICAL = "critical"  # System failure, data corruption, security breach
    HIGH = "high"  # Trading halted, model failure, risk limit breach
    MEDIUM = "medium"  # Performance degradation, data quality issues
    LOW = "low"  # Configuration warnings, minor validation errors


class ErrorCategory(Enum):
    """Error categories for intelligent routing."""

    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    EXCHANGE = "exchange"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """
    Comprehensive error context for tracking, recovery, and analysis.

    This unified class consolidates functionality from:
    - src.error_handling.base.ErrorContext
    - src.error_handling.error_handler.ErrorContext
    - src.error_handling.decorators.ErrorContext
    """

    # Core error information
    error: Exception
    error_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Error classification
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN

    # Context information
    component: str | None = None
    operation: str | None = None
    module: str | None = None
    function_name: str | None = None

    # Function call context (for decorator usage)
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    attempt_number: int = 1

    # Trading-specific context
    user_id: str | None = None
    bot_id: str | None = None
    symbol: str | None = None
    order_id: str | None = None

    # Error details
    stack_trace: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    additional: dict[str, Any] = field(default_factory=dict)

    # Recovery tracking
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    correlation_id: str | None = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Store attributes for efficient access
        has_error_type = hasattr(self, "error_type")
        has_error_message = hasattr(self, "error_message")
        has_function = hasattr(self, "function")

        # Auto-populate basic error information
        if not has_error_type:
            self.error_type = type(self.error).__name__
        if not has_error_message:
            self.error_message = str(self.error)

        # Auto-generate stack trace if not provided
        if not self.stack_trace:
            self.stack_trace = traceback.format_exc()

        # Ensure we have function_name from either source
        if not self.function_name and has_function:
            self.function_name = self.function

        # Merge additional context into details
        if self.additional:
            self.details.update(self.additional)

    @property
    def error_type(self) -> str:
        """Get the error type name."""
        return type(self.error).__name__

    @property
    def error_message(self) -> str:
        """Get the error message."""
        return str(self.error)

    @property
    def function(self) -> str | None:
        """Backward compatibility property."""
        return self.function_name

    @function.setter
    def function(self, value: str | None):
        """Backward compatibility setter."""
        self.function_name = value

    def to_dict(
        self, sanitize: bool = True, sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    ) -> dict[str, Any]:
        """Convert context to dictionary for serialization with optional sanitization."""
        base_dict = {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": (
                self.severity.value if isinstance(self.severity, ErrorSeverity) else self.severity
            ),
            "category": (
                self.category.value if isinstance(self.category, ErrorCategory) else self.category
            ),
            "component": self.component,
            "operation": self.operation,
            "module": self.module,
            "function_name": self.function_name,
            "user_id": self.user_id,
            "bot_id": self.bot_id,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "stack_trace": self.stack_trace,
            "details": self.details,
            "metadata": self.metadata,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "correlation_id": self.correlation_id,
            "attempt_number": self.attempt_number,
        }

        if sanitize:
            sanitizer = get_security_sanitizer()

            # Sanitize sensitive fields
            error_message = base_dict["error_message"]
            if error_message and isinstance(error_message, str):
                base_dict["error_message"] = sanitizer.sanitize_error_message(
                    error_message, sensitivity_level
                )

            stack_trace = base_dict["stack_trace"]
            if stack_trace and isinstance(stack_trace, str):
                base_dict["stack_trace"] = sanitizer.sanitize_stack_trace(
                    stack_trace, sensitivity_level
                )

            details = base_dict["details"]
            if details and isinstance(details, dict):
                base_dict["details"] = sanitizer.sanitize_context(details, sensitivity_level)

            metadata = base_dict["metadata"]
            if metadata and isinstance(metadata, dict):
                base_dict["metadata"] = sanitizer.sanitize_context(metadata, sensitivity_level)

        return base_dict

    def to_legacy_dict(
        self, sanitize: bool = True, sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    ) -> dict[str, Any]:
        """Convert to legacy format for backward compatibility with optional sanitization."""
        legacy_dict = {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "module": self.module,
            "function": self.function_name,
            "traceback": self.stack_trace,
        }

        # Add optional fields if present
        if self.user_id is not None:
            legacy_dict["user_id"] = self.user_id
        if self.bot_id is not None:
            legacy_dict["bot_id"] = self.bot_id
        if self.symbol is not None:
            legacy_dict["symbol"] = self.symbol
        if self.order_id is not None:
            legacy_dict["order_id"] = self.order_id

        # Merge details and additional
        legacy_dict.update(self.details)
        legacy_dict.update(self.additional)

        if sanitize:
            sanitizer = get_security_sanitizer()

            # Sanitize the entire legacy dict
            legacy_dict = sanitizer.sanitize_context(legacy_dict, sensitivity_level)

        return legacy_dict

    def add_detail(self, key: str, value: Any) -> None:
        """Add a detail to the context."""
        self.details[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value

    def increment_recovery_attempts(self) -> None:
        """Increment the recovery attempt counter."""
        self.recovery_attempts += 1

    def can_retry_recovery(self) -> bool:
        """Check if recovery can be retried."""
        return self.recovery_attempts < self.max_recovery_attempts

    def is_critical(self) -> bool:
        """Check if this is a critical error."""
        return self.severity == ErrorSeverity.CRITICAL

    def is_high_severity(self) -> bool:
        """Check if this is a high severity error."""
        return self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]

    def requires_escalation(self) -> bool:
        """Check if this error requires escalation."""
        return self.is_high_severity()

    @classmethod
    def from_exception(
        cls, error: Exception, component: str | None = None, operation: str | None = None, **kwargs
    ) -> "ErrorContext":
        """Create an ErrorContext from an exception with minimal required info."""
        import uuid

        # Separate known fields from additional context
        known_fields = {
            "user_id",
            "bot_id",
            "symbol",
            "order_id",
            "module",
            "function_name",
            "severity",
            "category",
            "stack_trace",
            "details",
            "metadata",
        }

        context_kwargs = {}
        additional_context = {}

        for key, value in kwargs.items():
            if key in known_fields:
                context_kwargs[key] = value
            else:
                additional_context[key] = value

        return cls(
            error=error,
            error_id=str(uuid.uuid4()),
            component=component,
            operation=operation,
            additional=additional_context,
            **context_kwargs,
        )

    @classmethod
    def from_decorator_context(
        cls,
        error: Exception,
        function_name: str,
        args: tuple,
        kwargs: dict,
        attempt_number: int = 1,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ErrorContext":
        """Create ErrorContext for decorator usage."""
        return cls(
            error=error,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            attempt_number=attempt_number,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )


class ErrorContextFactory(BaseComponent):
    """Factory for creating standardized error contexts - maintained for backward compatibility."""

    @staticmethod
    def create(error: Exception, **kwargs) -> dict[str, Any]:
        """
        Create standardized error context.

        This method automatically captures:
        - Timestamp
        - Module and function where error occurred
        - Full traceback
        - Error type and message
        - Any additional context provided

        Args:
            error: The exception
            **kwargs: Additional context to include

        Returns:
            Standardized error context dictionary
        """
        # Get caller frame information
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            module = caller_frame.f_globals.get("__name__", "unknown")
            function = caller_frame.f_code.co_name
            line = caller_frame.f_lineno
            filename = caller_frame.f_code.co_filename
        else:
            module = "unknown"
            function = "unknown"
            line = 0
            filename = "unknown"

        # Create ErrorContext and convert to dict for backward compatibility
        context = ErrorContext.from_exception(
            error=error,
            component=kwargs.get("component"),
            operation=kwargs.get("operation"),
            module=module,
            function_name=function,
            **kwargs,
        )

        # Determine if sanitization should be applied based on context
        should_sanitize = kwargs.get("sanitize", True)
        sensitivity_level = kwargs.get("sensitivity_level", SensitivityLevel.MEDIUM)

        result = context.to_legacy_dict(
            sanitize=should_sanitize, sensitivity_level=sensitivity_level
        )
        result.update(
            {
                "line": line,
                "filename": filename,
            }
        )

        # Sanitize filename if needed
        if should_sanitize:
            sanitizer = get_security_sanitizer()
            # Use public API method to sanitize filename
            file_context = {"filename": filename}
            sanitized_context = sanitizer.sanitize_context(file_context, sensitivity_level)
            result["filename"] = sanitized_context.get("filename", filename)

        return result

    @staticmethod
    def create_from_frame(error: Exception, frame: Any | None = None, **kwargs) -> dict[str, Any]:
        """
        Create error context from a specific frame.

        Args:
            error: The exception
            frame: Stack frame to use for context
            **kwargs: Additional context

        Returns:
            Error context dictionary
        """
        if frame:
            module = frame.f_globals.get("__name__", "unknown")
            function = frame.f_code.co_name
            line = frame.f_lineno
            filename = frame.f_code.co_filename
        else:
            module = "unknown"
            function = "unknown"
            line = 0
            filename = "unknown"

        context = ErrorContext.from_exception(
            error=error,
            component=kwargs.get("component"),
            operation=kwargs.get("operation"),
            module=module,
            function_name=function,
            **kwargs,
        )

        result = context.to_legacy_dict()
        result.update(
            {
                "line": line,
                "filename": filename,
            }
        )

        return result

    @staticmethod
    def create_minimal(error: Exception) -> dict[str, Any]:
        """
        Create minimal error context (for logging with less overhead).

        Args:
            error: The exception

        Returns:
            Minimal error context
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    @staticmethod
    def enrich_context(base_context: dict[str, Any], **additional) -> dict[str, Any]:
        """
        Enrich existing context with additional information.

        Args:
            base_context: Existing context
            **additional: Additional fields to add

        Returns:
            Enriched context
        """
        enriched = base_context.copy()
        enriched.update(additional)
        return enriched
