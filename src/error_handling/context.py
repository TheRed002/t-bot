"""
Unified error context definitions for the T-Bot trading system.

This module provides a single, comprehensive ErrorContext class that consolidates
all error context functionality previously scattered across multiple files.
"""

import inspect
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.base.factory import BaseFactory
from src.core.exceptions import ErrorCategory, ErrorSeverity
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


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
    category: ErrorCategory = ErrorCategory.SYSTEM

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
        """Convert context to dictionary with consistent data structure across modules."""
        # Standard base structure that matches core module expectations
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

        # Add standard data processing fields for consistency with core modules
        base_dict.update(
            {
                "data_format": "error_context_v1",
                "processing_stage": "error_handling",
                "validation_status": "validated" if self.details else "pending",
            }
        )

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


class ErrorContextFactory(BaseFactory[ErrorContext]):
    """Factory for creating standardized error contexts with DI support."""

    def __init__(self, dependency_container: Any | None = None):
        """Initialize factory with proper core.base.factory pattern."""
        # Use proper config structure from core.types
        from src.core.types.base import ConfigDict
        
        config = ConfigDict({}) if dependency_container is None else None
        super().__init__(product_type=ErrorContext, name="ErrorContextFactory", config=config)

        # Register default context creators using proper BaseFactory interface
        try:
            # Create wrapper functions that match the expected signature
            def standard_creator(*args, **kwargs) -> ErrorContext:
                error = args[0] if args else kwargs.get('error')
                if not error:
                    raise ValueError("Error parameter is required for standard creator")
                return ErrorContext.from_exception(error, **kwargs)
            
            def decorator_creator(*args, **kwargs) -> ErrorContext:
                error = args[0] if args else kwargs.get('error')
                function_name = args[1] if len(args) > 1 else kwargs.get('function_name', 'unknown')
                call_args = args[2] if len(args) > 2 else kwargs.get('args', ())
                call_kwargs = args[3] if len(args) > 3 else kwargs.get('kwargs', {})
                return ErrorContext.from_decorator_context(
                    error=error or Exception("Unknown"),
                    function_name=function_name,
                    args=call_args,
                    kwargs=call_kwargs,
                    **{k: v for k, v in kwargs.items() if k not in ['error', 'function_name', 'args', 'kwargs']}
                )
            
            self.register("standard", standard_creator)
            self.register("decorator", decorator_creator)
        except Exception as e:
            # Log but don't fail initialization
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some context creators: {e}")

        # Configure dependencies if available using core pattern
        if dependency_container and hasattr(self, 'configure_dependencies'):
            try:
                self.configure_dependencies(dependency_container)
            except Exception as e:
                # Log but continue without DI
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to configure dependencies: {e}")

    def create_context_dict(self, error: Exception, **kwargs) -> dict[str, Any]:
        """
        Create standardized error context using simplified factory pattern.

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
        try:
            # Get caller frame information using simplified approach
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

            # Create ErrorContext using factory pattern
            # Extract specific fields to avoid duplicate keyword arguments
            context_kwargs = kwargs.copy()
            component = context_kwargs.pop("component", None)
            operation = context_kwargs.pop("operation", None)
            should_sanitize = context_kwargs.pop("sanitize", True)
            sensitivity_level = context_kwargs.pop("sensitivity_level", SensitivityLevel.MEDIUM)

            # Use factory pattern for context creation
            try:
                context = self.create_context(
                    context_type="standard",
                    error=error,
                    component=component,
                    operation=operation,
                    module=module,
                    function_name=function,
                    **context_kwargs,
                )
            except Exception:
                # Fallback to direct creation
                context = ErrorContext.from_exception(
                    error=error,
                    component=component,
                    operation=operation,
                    module=module,
                    function_name=function,
                    **context_kwargs,
                )

            result = context.to_legacy_dict(
                sanitize=should_sanitize, sensitivity_level=sensitivity_level
            )
            result.update(
                {
                    "line": line,
                    "filename": filename,
                }
            )

            # Sanitize filename if needed using simplified approach
            if should_sanitize:
                try:
                    sanitizer = get_security_sanitizer()
                    file_context = {"filename": filename}
                    sanitized_context = sanitizer.sanitize_context(file_context, sensitivity_level)
                    result["filename"] = sanitized_context.get("filename", filename)
                except Exception as e:
                    # Fallback to original filename if sanitization fails
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to sanitize filename: {e}")
                    # Keep original filename

            return result

        except Exception as e:
            # Fallback to minimal context if factory pattern fails
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Factory pattern failed, using minimal context: {e}")
            return self.create_minimal(error)

    def create(self, error: Exception, **kwargs) -> dict[str, Any]:
        """
        Legacy create method for backward compatibility.

        Args:
            error: The exception
            **kwargs: Additional context to include

        Returns:
            Error context dictionary
        """
        return self.create_context_dict(error, **kwargs)

    def create_context(
        self, context_type: str = "standard", error: Exception | None = None, **kwargs
    ) -> ErrorContext:
        """
        Create ErrorContext using registered creator with proper BaseFactory interface.

        Args:
            context_type: Type of context to create ('standard' or 'decorator')
            error: The exception (required for standard contexts)
            **kwargs: Additional context parameters

        Returns:
            ErrorContext instance
        """
        try:
            # Use proper BaseFactory.create method with correct parameters
            if context_type == "standard" and error:
                # Call parent create method, not recursive self.create_context
                return super().create("standard", error, **kwargs)
            elif context_type == "decorator":
                return super().create("decorator", error, **kwargs)
            else:
                raise ValueError(
                    f"Invalid context type '{context_type}' or missing error parameter"
                )
        except Exception as e:
            # Fallback to direct creation if factory fails
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Factory creation failed, using direct creation: {e}")

            if error:
                return ErrorContext.from_exception(error, **kwargs)
            else:
                return ErrorContext(error=Exception("Unknown error"), **kwargs)

    def create_from_frame(
        self, error: Exception, frame: Any | None = None, **kwargs
    ) -> dict[str, Any]:
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

    def create_minimal(self, error: Exception) -> dict[str, Any]:
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

    def enrich_context(self, base_context: dict[str, Any], **additional) -> dict[str, Any]:
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
