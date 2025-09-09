"""
Comprehensive error handling framework for the trading bot.

This module provides error categorization, severity classification, retry policies,
circuit breaker integration, and error context preservation for debugging and recovery.

CRITICAL: This module integrates with P-001 core exceptions and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from src.error_handling.context import ErrorContext
    from src.error_handling.recovery import RecoveryStrategy

from src.core.config import Config
from src.core.exceptions import (
    DataError,
    ErrorSeverity,
    ExchangeError,
    ExecutionError,
    ModelError,
    RiskManagementError,
    SecurityError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from src.core.logging import get_logger
from src.error_handling.security_rate_limiter import (
    record_recovery_failure,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
)
from src.utils.decorators import time_execution
from src.utils.error_categorization import is_financial_component

# Configuration constants
DEFAULT_ERROR_PATTERN_MAX_SIZE = 1000
DEFAULT_ERROR_PATTERN_TTL_HOURS = 24
DEFAULT_PERFORMANCE_LOG_INTERVAL_MINUTES = 10
DEFAULT_CLEANUP_INTERVAL_MINUTES = 30
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30
DEFAULT_BACKGROUND_TASK_CLEANUP_TIMEOUT = 10.0
DEFAULT_DECIMAL_PRECISION = 8
DEFAULT_MIN_TIME_HOURS = 0.1

# Rate limiting constants
DEFAULT_MAX_ATTEMPTS = 3
NETWORK_ERROR_MAX_ATTEMPTS = 5
API_RATE_LIMIT_MAX_ATTEMPTS = 3
DATABASE_ERROR_MAX_ATTEMPTS = 3

logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: int = DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                else:
                    raise TradingBotError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

            raise e

    def should_transition_to_half_open(self) -> bool:
        """Check if circuit breaker should transition to HALF_OPEN state."""
        with self._lock:
            if self.state == "OPEN" and self.last_failure_time:
                return time.time() - self.last_failure_time > self.recovery_timeout
            return False

    def open(self) -> None:
        """Manually open the circuit breaker."""
        with self._lock:
            self.state = "OPEN"
            self.last_failure_time = time.time()

    def is_open(self) -> bool:
        """Check if circuit breaker is in OPEN state."""
        return self.state == "OPEN"

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state."""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None

    @property
    def threshold(self) -> int:
        """Get the failure threshold."""
        return self.failure_threshold


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ErrorPatternCache:
    """Optimized error pattern storage with size limits and TTL."""

    def __init__(
        self,
        max_patterns: int = DEFAULT_ERROR_PATTERN_MAX_SIZE,
        ttl_hours: int = DEFAULT_ERROR_PATTERN_TTL_HOURS,
    ) -> None:
        self.max_patterns = max_patterns
        self.ttl_hours = ttl_hours
        self._patterns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._last_cleanup = datetime.now(timezone.utc)
        self._cleanup_interval_minutes = DEFAULT_CLEANUP_INTERVAL_MINUTES

    def add_pattern(self, pattern: dict[str, Any] | Any) -> None:
        """Add or update error pattern with automatic cleanup."""
        if self._should_cleanup():
            self._cleanup_expired()

        while len(self._patterns) >= self.max_patterns:
            self._patterns.popitem(last=False)

        if is_dataclass(pattern) and not isinstance(pattern, type):
            pattern_dict = asdict(pattern)
        else:
            pattern_dict = pattern

        self._patterns[pattern_dict["pattern_id"]] = pattern_dict
        self._patterns.move_to_end(pattern_dict["pattern_id"])

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        """Get pattern and mark as recently used."""
        if pattern_id not in self._patterns:
            return None

        pattern = self._patterns[pattern_id]
        current_time_utc = datetime.now(timezone.utc)
        first_time_key = "first_detected" if "first_detected" in pattern else "first_seen"
        pattern_time = ensure_timezone_aware(pattern[first_time_key])
        time_diff_seconds = (current_time_utc - pattern_time).total_seconds()
        age_hours = Decimal(str(time_diff_seconds)) / Decimal("3600")
        if age_hours > self.ttl_hours:
            del self._patterns[pattern_id]
            return None

        self._patterns.move_to_end(pattern_id)
        return pattern

    def get_all_patterns(self) -> dict[str, dict[str, Any]]:
        """Get all active patterns."""
        self._cleanup_expired()
        return dict(self._patterns)

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        current_time_utc = datetime.now(timezone.utc)
        # Ensure both datetimes are timezone-aware for comparison
        last_cleanup_time = ensure_timezone_aware(self._last_cleanup)
        cleanup_time_diff = (current_time_utc - last_cleanup_time).total_seconds()
        minutes_since_cleanup = Decimal(str(cleanup_time_diff)) / Decimal("60")
        return minutes_since_cleanup >= self._cleanup_interval_minutes

    def _cleanup_expired(self) -> None:
        """Remove expired patterns."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for pattern_id, pattern in self._patterns.items():
            # Ensure both datetimes are timezone-aware for comparison
            # Handle both field names: first_detected (dict) and first_seen (dataclass)
            first_time_key = "first_detected" if "first_detected" in pattern else "first_seen"
            pattern_time = ensure_timezone_aware(pattern[first_time_key])
            time_diff = (current_time - pattern_time).total_seconds()
            age_hours = Decimal(str(time_diff)) / Decimal("3600")
            if age_hours > self.ttl_hours:
                expired_keys.append(pattern_id)

        for key in expired_keys:
            del self._patterns[key]

        self._last_cleanup = current_time

    def cleanup_expired(self) -> None:
        """Public method to remove expired patterns."""
        self._cleanup_expired()

    def size(self) -> int:
        """Get current number of patterns."""
        return len(self._patterns)

    def get_last_cleanup(self) -> datetime:
        """Get the last cleanup timestamp."""
        return self._last_cleanup


class ErrorHandler:
    """Comprehensive error handling and recovery system with optimized memory management."""

    def __init__(self, config: Config, sanitizer=None, rate_limiter=None) -> None:
        # Use proper Config type from core
        from src.core.config import Config as CoreConfig

        if config is None:
            self.config = CoreConfig()
        elif hasattr(config, "model_dump"):
            self.config = config
        else:
            # Convert dict config to proper Config if needed
            self.config = CoreConfig()

        self.logger = get_logger(self.__class__.__module__)

        # Security components - use injected dependencies
        self.sanitizer = sanitizer
        self.rate_limiter = rate_limiter

        # Optimized error pattern storage
        self.error_patterns = ErrorPatternCache()
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Performance monitoring
        self._operation_count = 0
        self._last_performance_log = datetime.now(timezone.utc)
        self._performance_log_interval_minutes = DEFAULT_PERFORMANCE_LOG_INTERVAL_MINUTES
        self.retry_policies: dict[str, dict[str, Any]] = {
            "network_errors": {
                "max_attempts": NETWORK_ERROR_MAX_ATTEMPTS,
                "backoff_strategy": "exponential",
                "base_delay": 1,
                "max_delay": 60,
                "jitter": True,
            },
            "api_rate_limits": {
                "max_attempts": API_RATE_LIMIT_MAX_ATTEMPTS,
                "backoff_strategy": "linear",
                "base_delay": 5,
                "respect_retry_after": True,
            },
            "database_errors": {
                "max_attempts": DATABASE_ERROR_MAX_ATTEMPTS,
                "backoff_strategy": "exponential",
                "base_delay": 0.5,
                "max_delay": 10,
            },
        }

        # Initialize circuit breakers for critical components
        self._initialize_circuit_breakers()

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector using core patterns."""
        try:
            # Use proper dependency injection interface
            if not (hasattr(injector, "has_service") and hasattr(injector, "resolve")):
                raise ValueError("Invalid injector provided - must implement DI interface")

            # Try to resolve security components from DI container
            if injector.has_service("SecuritySanitizer") and self.sanitizer is None:
                self.sanitizer = injector.resolve("SecuritySanitizer")

            if injector.has_service("SecurityRateLimiter") and self.rate_limiter is None:
                self.rate_limiter = injector.resolve("SecurityRateLimiter")

            # Ensure all required dependencies are resolved
            if self.sanitizer is None:
                raise ValueError("SecuritySanitizer dependency not available")
            if self.rate_limiter is None:
                raise ValueError("SecurityRateLimiter dependency not available")

            self.logger.debug("ErrorHandler dependencies configured via DI container")
        except Exception as e:
            self.logger.error(f"Failed to configure ErrorHandler dependencies via DI: {e}")
            raise

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for critical system components."""
        self.circuit_breakers = {
            "api_calls": CircuitBreaker(),
            "database_connections": CircuitBreaker(failure_threshold=3, recovery_timeout=15),
            "exchange_connections": CircuitBreaker(failure_threshold=3, recovery_timeout=20),
            "model_inference": CircuitBreaker(failure_threshold=2, recovery_timeout=60),
        }

    @time_execution
    def classify_error(self, error: Exception) -> ErrorSeverity:
        """
        Classify error severity based on error type and context.

        Args:
            error: The exception to classify

        Returns:
            ErrorSeverity enum value indicating the severity level:
            - CRITICAL: StateConsistencyError, SecurityError
            - HIGH: RiskManagementError, ExchangeError, ConnectionError, ExecutionError
            - MEDIUM: DataError, ModelError, or unknown error types
            - LOW: ValidationError
        """
        # Ensure we have a valid exception
        if not isinstance(error, Exception):
            return ErrorSeverity.MEDIUM

        if isinstance(error, StateConsistencyError | SecurityError):
            return ErrorSeverity.CRITICAL
        elif isinstance(
            error, RiskManagementError | ExchangeError | ConnectionError | ExecutionError
        ):
            return ErrorSeverity.HIGH
        elif isinstance(error, DataError | ModelError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, ValidationError):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM

    def validate_module_boundary_input(
        self, data: dict[str, Any], source_module: str
    ) -> dict[str, Any]:
        """
        Validate input data at module boundary with core alignment.

        Args:
            data: Input data to validate
            source_module: Source module name for validation rules

        Returns:
            Validated and normalized data

        Raises:
            ValidationError: If data fails validation
        """
        from src.core.exceptions import ValidationError

        # Apply core-style validation
        validated_data = self.validate_data_flow_consistency(data)

        # Module-specific boundary validations
        if source_module == "core":
            # Validate core event format
            required_fields = ["data_format", "timestamp", "processing_stage"]
            missing_fields = [field for field in required_fields if field not in validated_data]
            if missing_fields:
                raise ValidationError(
                    f"Missing required fields from core module: {missing_fields}",
                    error_code="VALID_001",
                    field_name="module_boundary",
                    validation_rule="core_event_format",
                )

        elif source_module == "database":
            # Validate database context
            if "entity_id" in validated_data and not isinstance(
                validated_data["entity_id"], str | int
            ):
                raise ValidationError(
                    "Database entity_id must be string or int",
                    error_code="VALID_002",
                    field_name="entity_id",
                    field_value=validated_data["entity_id"],
                )

        elif source_module == "exchanges":
            # Validate financial data precision
            financial_fields = ["price", "quantity", "amount"]
            for field in financial_fields:
                if field in validated_data and validated_data[field] is not None:
                    try:
                        from src.utils.decimal_utils import to_decimal

                        validated_data[field] = to_decimal(validated_data[field])
                    except Exception as e:
                        raise ValidationError(
                            f"Invalid financial data format for {field}",
                            error_code="VALID_003",
                            field_name=field,
                            field_value=validated_data[field],
                            validation_rule="financial_precision",
                        ) from e

        return validated_data

    @time_execution
    def create_error_context(
        self, error: Exception, component: str, operation: str, **kwargs
    ) -> "ErrorContext":
        """Create error context for tracking and recovery with security sanitization."""
        # Validate input at module boundary
        boundary_data = {
            "component": component,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs,
        }

        # Determine source module from component for validation
        source_module = "unknown"
        if "core" in component.lower():
            source_module = "core"
        elif "database" in component.lower():
            source_module = "database"
        elif "exchange" in component.lower():
            source_module = "exchanges"

        # Apply boundary validation with enhanced consistency patterns
        try:
            validated_boundary_data = self.validate_module_boundary_input(
                boundary_data, source_module
            )

            # Apply additional boundary validation for state-to-error flow
            if source_module == "state" or "state" in component.lower():
                from src.utils.messaging_patterns import BoundaryValidator

                try:
                    BoundaryValidator.validate_monitoring_to_error_boundary(validated_boundary_data)
                except Exception as boundary_error:
                    self.logger.warning(
                        f"State-to-error boundary validation failed: {boundary_error}"
                    )

        except Exception as validation_error:
            self.logger.warning(
                "Boundary validation failed, proceeding with sanitization only",
                component=component,
                operation=operation,
                validation_error=str(validation_error),
            )
            validated_boundary_data = boundary_data

        # Extract kwargs that should be passed to ErrorContext
        context_kwargs = {}
        for key in ["user_id", "bot_id", "symbol", "order_id"]:
            if key in validated_boundary_data:
                context_kwargs[key] = validated_boundary_data.pop(key)

        # Determine sensitivity level based on error classification
        severity = self.classify_error(error)
        sensitivity_level = self._get_sensitivity_level(severity, component)

        # Ensure sanitizer is available - must be injected via DI
        if self.sanitizer is None:
            raise ValueError("SecuritySanitizer must be injected via dependency injection")

        # Sanitize error message and details using validated data
        sanitized_error_message = self.sanitizer.sanitize_error_message(
            validated_boundary_data.get("error_message", str(error)), sensitivity_level
        )
        sanitized_kwargs = self.sanitizer.sanitize_context(
            validated_boundary_data, sensitivity_level
        )
        sanitized_stack_trace = self.sanitizer.sanitize_stack_trace(
            self._get_stack_trace(), sensitivity_level
        )

        # Import ErrorContext at runtime to avoid circular dependency
        from src.error_handling.context import ErrorContext

        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            component=component,
            operation=operation,
            error=error,
            details={
                "error_type": type(error).__name__,
                "error_message": sanitized_error_message,
                "kwargs": sanitized_kwargs,
                "sensitivity_level": sensitivity_level.value,
                "boundary_validated": True,
                "source_module": source_module,
            },
            stack_trace=sanitized_stack_trace,
            **context_kwargs,
        )

    def _get_stack_trace(self) -> str:
        """Get current stack trace for debugging."""
        return "".join(traceback.format_stack())

    @time_execution
    async def handle_error(
        self,
        error: Exception,
        context: "ErrorContext",
        recovery_strategy: Union["RecoveryStrategy", Callable[..., Any]] | None = None,
    ) -> bool:
        """Handle error with consistent data flow patterns and security controls."""

        # Ensure rate limiter is available - should be injected via DI
        if self.rate_limiter is None:
            raise ValueError(
                "SecurityRateLimiter not configured - ensure dependency injection is set up"
            )

        # Check rate limits before processing
        rate_limit_result = await self.rate_limiter.check_rate_limit(
            component=context.component or "unknown",
            operation="error_recovery",
            context={"error_id": context.error_id},
        )

        if not rate_limit_result.allowed:
            self.logger.warning(
                "Error recovery rate limited",
                error_id=context.error_id,
                component=context.component or "unknown",
                reason=rate_limit_result.reason,
                retry_after=rate_limit_result.suggested_retry_after,
            )
            # Still log the error but don't attempt recovery

        # Transform error data to match core event patterns
        from src.error_handling.propagation_utils import (
            ProcessingStage,
            PropagationMethod,
            add_propagation_step,
        )
        from src.utils.messaging_patterns import MessagingCoordinator

        # Use consistent messaging coordinator for data transformation
        coordinator = MessagingCoordinator("ErrorHandler")

        raw_error_data = {
            "error_id": context.error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "component": context.component,
            "operation": context.operation,
            "severity": context.severity.value,
            "processing_mode": "stream",  # Align with state module default
            "message_pattern": "pub_sub",  # Consistent messaging pattern
            "processing_stage": "error_handling",
            **context.details,
        }

        # Apply consistent data transformation
        error_event_data = coordinator._apply_data_transformation(raw_error_data)

        # Add propagation tracking for consistent data flow
        error_event_data = add_propagation_step(
            error_event_data,
            source_module="error_handling",
            target_module="core",
            method=PropagationMethod.EVENT_EMISSION,
            stage=ProcessingStage.ERROR_HANDLING,
        )

        # Get sanitized data for logging using transformed data
        sensitivity_level = self._get_sensitivity_level(
            context.severity, context.component or "unknown"
        )
        sanitized_details = self.sanitizer.sanitize_context(error_event_data, sensitivity_level)

        # Log error with sanitized structured data in event format
        self.logger.error(
            "Error event processed",
            **{
                **sanitized_details,
                "rate_limited": not rate_limit_result.allowed,
                "event_type": "error_occurred",
                "message_pattern": "pub_sub",  # Consistent messaging pattern
                "processing_mode": "stream",  # Align with state module
            },
        )

        # Update error patterns
        self._update_error_patterns(context)

        # Check if error should trigger circuit breaker
        circuit_breaker_key = self._get_circuit_breaker_key(context)
        if circuit_breaker_key and circuit_breaker_key in self.circuit_breakers:
            try:
                self.circuit_breakers[circuit_breaker_key].call(lambda: self._raise_error(error))
            except TradingBotError:
                self.logger.warning(
                    "Circuit breaker triggered",
                    circuit_breaker=circuit_breaker_key,
                    error_id=context.error_id,
                )

        # Return to service layer for recovery execution
        recovery_attempted = False
        if (
            recovery_strategy
            and context.recovery_attempts < context.max_recovery_attempts
            and rate_limit_result.allowed
        ):
            try:
                context.recovery_attempts += 1
                recovery_attempted = True

                # Don't execute recovery strategy directly - delegate to service
                self.logger.info(
                    "Recovery strategy prepared for service layer execution",
                    error_id=context.error_id,
                    component=context.component or "unknown",
                    recovery_attempts=context.recovery_attempts,
                    strategy_type=type(recovery_strategy).__name__,
                )
                return True
            except Exception as recovery_error:
                # Record recovery failure for rate limiting
                record_recovery_failure(
                    component=context.component or "unknown",
                    operation="error_recovery",
                    error_severity=context.severity.value,
                )

                # Sanitize recovery error message
                sanitized_recovery_error = self.sanitizer.sanitize_error_message(
                    str(recovery_error), sensitivity_level
                )

                self.logger.error(
                    "Recovery preparation failed",
                    error_id=context.error_id,
                    component=context.component or "unknown",
                    recovery_error=sanitized_recovery_error,
                    recovery_attempts=context.recovery_attempts,
                )
                recovery_attempted = True

        # Record failure if recovery wasn't attempted or rate limited
        if not recovery_attempted or not rate_limit_result.allowed:
            record_recovery_failure(
                component=context.component or "unknown",
                operation="error_handling",
                error_severity=context.severity.value,
            )

        # Escalate if critical or high severity with sanitized context
        if context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            await self._escalate_error(context)

        return False

    def handle_error_sync(
        self,
        error: Exception,
        component: str,
        operation: str,
        recovery_strategy: Callable | None = None,
        **kwargs,
    ) -> bool:
        """
        Synchronous version of handle_error with stream processing patterns aligned to core.

        Args:
            error: The exception to handle
            component: Component where error occurred
            operation: Operation being performed
            recovery_strategy: Optional recovery strategy
            **kwargs: Additional context data

        Returns:
            bool: True if error was recovered, False otherwise
        """
        import asyncio

        # Standardize sync context format aligned with state module processing patterns
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        kwargs.update(
            {
                "processing_mode": "stream",  # Align with state module default
                "processing_stage": "error_handling",
                "data_format": "error_context_v1",
                "message_pattern": "pub_sub",  # Consistent messaging
                "stream_position": kwargs.get("stream_position", 0),
                "correlation_id": kwargs.get(
                    "correlation_id", f"error_stream_{datetime.now(timezone.utc).timestamp()}"
                ),
            }
        )

        # Apply processing paradigm alignment if needed
        aligned_kwargs = ProcessingParadigmAligner.align_processing_modes("sync", "stream", kwargs)

        # Get or create event loop with consistent error handling
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task for stream processing
            # Create error context first with standardized format using aligned kwargs
            context = self.create_error_context(
                error=error, component=component, operation=operation, **aligned_kwargs
            )
            task = asyncio.ensure_future(self.handle_error(error, context, recovery_strategy))
            # Store task reference to prevent garbage collection (thread-safe initialization)
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
                self._background_task_lock = threading.RLock()

            with self._background_task_lock:
                self._background_tasks.add(task)

            # Use a wrapper callback to ensure thread-safe removal
            def safe_remove_task(finished_task):
                with getattr(self, "_background_task_lock", threading.RLock()):
                    if hasattr(self, "_background_tasks"):
                        self._background_tasks.discard(finished_task)

            task.add_done_callback(safe_remove_task)
            # Can't wait synchronously in running loop
            self.logger.warning(
                "handle_error_sync called from async context, scheduled as stream task",
                component=component,
                operation=operation,
                processing_mode="stream_async",
                stream_position=kwargs.get("stream_position"),
            )
            return False  # Can't wait for result
        except RuntimeError:
            # No running loop, safe to use run_until_complete with stream processing
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Create error context first with standardized stream format using aligned kwargs
                context = self.create_error_context(
                    error=error, component=component, operation=operation, **aligned_kwargs
                )
                return loop.run_until_complete(self.handle_error(error, context, recovery_strategy))
            finally:
                if loop:
                    try:
                        loop.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing event loop: {e}")
                    finally:
                        try:
                            asyncio.set_event_loop(None)
                        except Exception as cleanup_error:
                            error_msg = f"Event loop cleanup error (ignored): {cleanup_error}"
                            self.logger.debug(error_msg)

    def handle_error_batch(
        self,
        errors: list[Exception],
        contexts: list[dict[str, Any]],
        recovery_strategy: Callable | None = None,
    ) -> list[bool]:
        """
        Batch error handling aligned with core batch processing patterns.

        Args:
            errors: List of exceptions to handle
            contexts: List of context dictionaries for each error
            recovery_strategy: Optional recovery strategy

        Returns:
            List of recovery success flags
        """
        import asyncio

        # Validate batch size alignment with core patterns
        if len(errors) != len(contexts):
            raise ValueError("Errors and contexts lists must have same length")

        # Mark as batch processing
        batch_id = f"batch_{datetime.now(timezone.utc).timestamp()}"

        results = []
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Process all errors in batch
            async def process_batch():
                batch_results = []
                for i, (error, context_data) in enumerate(zip(errors, contexts, strict=False)):
                    # Add batch processing metadata
                    context_data.update(
                        {
                            "processing_mode": "batch",
                            "batch_id": batch_id,
                            "batch_position": i,
                            "batch_size": len(errors),
                        }
                    )

                    context = self.create_error_context(
                        error=error,
                        component=context_data.get("component", "unknown"),
                        operation=context_data.get("operation", "batch_operation"),
                        **context_data,
                    )

                    result = await self.handle_error(error, context, recovery_strategy)
                    batch_results.append(result)

                return batch_results

            results = loop.run_until_complete(process_batch())

        finally:
            if loop:
                try:
                    loop.close()
                except Exception as e:
                    self.logger.warning(f"Error closing batch processing loop: {e}")
                finally:
                    try:
                        asyncio.set_event_loop(None)
                    except Exception as cleanup_error:
                        error_msg = (
                            f"Batch processing loop cleanup error (ignored): {cleanup_error}"
                        )
                        self.logger.debug(error_msg)

        return results

    def _update_error_patterns(self, context: "ErrorContext") -> None:
        """Update error pattern tracking for analytics with optimized storage."""
        pattern_key = f"{context.component}:{type(context.error).__name__}"

        # Get existing pattern or create new one
        existing_pattern = self.error_patterns.get_pattern(pattern_key)

        if existing_pattern is None:
            new_pattern = {
                "pattern_id": pattern_key,
                "pattern_type": "frequency",
                "component": context.component or "unknown",
                "error_type": type(context.error).__name__,
                "frequency": Decimal("1.0"),
                "severity": context.severity.value,
                "first_detected": context.timestamp,
                "last_detected": context.timestamp,
                "occurrence_count": 1,
                "confidence": Decimal("0.8"),
                "description": f"Error pattern for {context.component}",
                "suggested_action": "Monitor and investigate",
            }
            self.error_patterns.add_pattern(new_pattern)
        else:
            # Update existing pattern
            existing_pattern["occurrence_count"] += 1
            existing_pattern["last_detected"] = context.timestamp
            existing_pattern["frequency"] = self._calculate_frequency(existing_pattern)
            self.error_patterns.add_pattern(existing_pattern)  # Update position in LRU

        # Log performance metrics periodically
        self._operation_count += 1
        self._log_performance_metrics()

    def _calculate_frequency(self, pattern: dict[str, Any]) -> Decimal:
        """Calculate error frequency per hour."""
        time_diff = (pattern["last_detected"] - pattern["first_detected"]).total_seconds()
        # Ensure minimum time to avoid division by zero
        if time_diff <= 0:
            # If no time has passed, return the count as instantaneous frequency
            return Decimal(str(pattern["occurrence_count"]))
        hours = Decimal(str(time_diff)) / Decimal("3600")
        # Ensure minimum of 0.1 hours to prevent very high frequencies
        hours = max(hours, Decimal(str(DEFAULT_MIN_TIME_HOURS)))
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = DEFAULT_DECIMAL_PRECISION
            ctx.rounding = ROUND_HALF_UP
            return Decimal(str(pattern["occurrence_count"])) / Decimal(str(hours))

    def _log_performance_metrics(self) -> None:
        """Log performance metrics periodically."""
        now = datetime.now(timezone.utc)
        time_diff = (now - self._last_performance_log).total_seconds()
        minutes_since_log = Decimal(str(time_diff)) / Decimal("60")

        if minutes_since_log >= self._performance_log_interval_minutes:
            self.logger.info(
                "ErrorHandler performance metrics",
                operations_processed=self._operation_count,
                error_patterns_count=self.error_patterns.size(),
                circuit_breakers_count=len(self.circuit_breakers),
                time_window_minutes=minutes_since_log,
            )
            self._operation_count = 0
            self._last_performance_log = now

    def _get_circuit_breaker_key(self, context: "ErrorContext") -> str | None:
        """Determine which circuit breaker should be triggered."""
        component = context.component or ""
        if "api" in component.lower():
            return "api_calls"
        elif "database" in component.lower():
            return "database_connections"
        elif "exchange" in component.lower():
            return "exchange_connections"
        elif "model" in component.lower():
            return "model_inference"
        return None

    def _raise_error(self, error: Exception) -> None:
        """Helper method to raise error for circuit breaker."""
        raise error

    async def _escalate_error(self, context: "ErrorContext") -> None:
        """Escalate critical errors for immediate attention with secure data handling."""

        # Determine escalation sensitivity level (always HIGH for escalations)
        escalation_sensitivity = SensitivityLevel.HIGH

        # Sanitize escalation data to prevent sensitive information leakage
        sanitized_details = self.sanitizer.sanitize_context(context.details, escalation_sensitivity)

        escalation_message = {
            "error_id": context.error_id,
            "severity": context.severity.value,
            "component": context.component,
            "operation": context.operation,
            "timestamp": context.timestamp.isoformat(),
            "details": sanitized_details,
            "security_note": "Escalation data has been sanitized for security",
        }

        self.logger.critical("Error escalation required", escalation_data=escalation_message)

        # Multi-channel alerting placeholder - implement with production notification system

    def get_retry_policy(self, error_type: str) -> dict[str, Any]:
        """Get retry policy for specific error type."""
        return self.retry_policies.get(error_type, self.retry_policies["network_errors"])

    def get_circuit_breaker_status(self) -> dict[str, str]:
        """Get status of all circuit breakers."""
        return {name: breaker.state for name, breaker in self.circuit_breakers.items()}

    def get_error_patterns(self) -> dict[str, dict[str, Any]]:
        """Get current error patterns for analysis."""
        return self.error_patterns.get_all_patterns()

    def get_memory_usage_stats(self) -> dict[str, Any]:
        """Get memory usage statistics for monitoring."""
        return {
            "error_patterns_count": self.error_patterns.size(),
            "circuit_breakers_count": len(self.circuit_breakers),
            "operations_processed": self._operation_count,
            "last_cleanup": self.error_patterns.get_last_cleanup().isoformat(),
        }

    async def cleanup_resources(self) -> None:
        """Manual cleanup of resources for memory management."""
        # Cancel any background tasks with thread-safe access
        if hasattr(self, "_background_tasks"):
            # Get snapshot of tasks to cancel
            with getattr(self, "_background_task_lock", threading.RLock()):
                tasks_to_cancel = list(self._background_tasks)

            # Cancel tasks
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()

            try:
                # Wait for tasks to complete with timeout
                if tasks_to_cancel:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=DEFAULT_BACKGROUND_TASK_CLEANUP_TIMEOUT,
                    )
            except asyncio.TimeoutError:
                self.logger.warning("Background task cleanup timed out")
            finally:
                # Clear the background tasks set with lock
                with getattr(self, "_background_task_lock", threading.RLock()):
                    if hasattr(self, "_background_tasks"):
                        self._background_tasks.clear()

        # Force cleanup of expired patterns
        self.error_patterns.cleanup_expired()

        # Remove inactive circuit breakers
        inactive_breakers = []
        for name, breaker in self.circuit_breakers.items():
            with breaker._lock:
                if (
                    breaker.last_failure_time
                    and (time.time() - breaker.last_failure_time) > 3600  # 1 hour
                ):
                    inactive_breakers.append(name)

        for name in inactive_breakers:
            del self.circuit_breakers[name]

        self.logger.info(
            "Resource cleanup completed",
            patterns_remaining=self.error_patterns.size(),
            circuit_breakers_remaining=len(self.circuit_breakers),
            inactive_breakers_removed=len(inactive_breakers),
        )

    def _get_sensitivity_level(self, severity: ErrorSeverity, component: str) -> SensitivityLevel:
        """Determine sensitivity level based on error severity and component."""
        # Financial/trading components always get high sensitivity
        if is_financial_component(component):
            return SensitivityLevel.CRITICAL

        # Map error severity to sensitivity level
        severity_mapping = {
            ErrorSeverity.CRITICAL: SensitivityLevel.CRITICAL,
            ErrorSeverity.HIGH: SensitivityLevel.HIGH,
            ErrorSeverity.MEDIUM: SensitivityLevel.MEDIUM,
            ErrorSeverity.LOW: SensitivityLevel.LOW,
        }

        return severity_mapping.get(severity, SensitivityLevel.MEDIUM)

    def validate_data_flow_consistency(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and ensure consistent data flow patterns across modules.
        Aligns with core event data transformation patterns.

        Args:
            data: Data dictionary to validate and standardize

        Returns:
            Standardized data dictionary with consistent format
        """
        if not isinstance(data, dict):
            return {
                "data_format": "error_context_v1",
                "processing_stage": "validation",
                "validation_status": "transformed",
                "original_data": data,
                "payload": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Ensure standard fields are present - align with core event patterns
        standardized = data.copy()
        if "data_format" not in standardized:
            standardized["data_format"] = "error_context_v1"
        if "processing_stage" not in standardized:
            standardized["processing_stage"] = "error_handling"
        if "validation_status" not in standardized:
            standardized["validation_status"] = "validated"
        if "timestamp" not in standardized:
            standardized["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Apply consistent financial data transformations like core events
        if "price" in standardized and standardized["price"] is not None:
            from src.utils.decimal_utils import to_decimal

            standardized["price"] = to_decimal(standardized["price"])

        if "quantity" in standardized and standardized["quantity"] is not None:
            from src.utils.decimal_utils import to_decimal

            standardized["quantity"] = to_decimal(standardized["quantity"])

        # Mark as financial data if contains financial fields
        if any(key in standardized for key in ["price", "quantity", "amount", "balance"]):
            standardized["financial_data"] = True

        return standardized

    async def shutdown(self) -> None:
        """Shutdown error handler and cleanup resources."""

        # Cleanup resources first
        try:
            await self.cleanup_resources()
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
            raise

        # Force cleanup of expired patterns
        self.error_patterns.cleanup_expired()

        self.logger.info("Error handler shutdown completed")


def create_error_handler_factory(
    config: Config | None = None, dependency_container: Any | None = None
):
    """Create a factory function for ErrorHandler instances using core factory patterns."""

    def factory():
        # Use proper fallback to create config if none provided
        from src.core.config import Config as CoreConfig

        resolved_config = config if config is not None else CoreConfig()

        # Resolve security components from DI container - no fallbacks allowed
        sanitizer = None
        rate_limiter = None

        if dependency_container and hasattr(dependency_container, "has_service"):
            if dependency_container.has_service("SecuritySanitizer"):
                sanitizer = dependency_container.resolve("SecuritySanitizer")
            else:
                raise ValueError("SecuritySanitizer service not registered in DI container")

            if dependency_container.has_service("SecurityRateLimiter"):
                rate_limiter = dependency_container.resolve("SecurityRateLimiter")
            else:
                raise ValueError("SecurityRateLimiter service not registered in DI container")
        else:
            raise ValueError("Dependency container required for ErrorHandler creation")

        handler = ErrorHandler(resolved_config, sanitizer=sanitizer, rate_limiter=rate_limiter)

        # Configure dependencies - required for proper DI
        handler.configure_dependencies(dependency_container)
        return handler

    return factory


def register_error_handler_with_di(injector, config: Config | None = None) -> None:
    """Register ErrorHandler with dependency injection container using core patterns."""
    # Validate injector has required methods from core.dependency_injection
    if not hasattr(injector, "register_factory"):
        logger.error("Invalid injector provided - missing register_factory method")
        raise ValueError("Injector must implement core dependency injection interface")

    factory = create_error_handler_factory(config, dependency_container=injector)
    injector.register_factory("ErrorHandler", factory, singleton=True)


def error_handler_decorator(
    component: str,
    operation: str,
    error_handler: ErrorHandler | None = None,
    recovery_strategy: Union["RecoveryStrategy", Callable[..., Any]] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Decorator for automatic error handling and recovery.

    Args:
        component: Component where error occurred
        operation: Operation being performed
        error_handler: Optional error handler instance (if None, creates a default one)
        recovery_strategy: Optional recovery strategy
        **kwargs: Additional context data
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Use provided handler or create a default one
                handler = error_handler or ErrorHandler(Config())

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                await handler.handle_error(e, context, recovery_strategy)
                raise e

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Use provided handler or create a default one
                handler = error_handler or ErrorHandler(Config())

                context = handler.create_error_context(
                    error=e, component=component, operation=operation, **kwargs
                )

                # For sync functions, we can't await, so just log
                func_logger = get_logger(func.__module__)
                func_logger.error(
                    "Error in sync function",
                    error_id=context.error_id,
                    severity=context.severity.value,
                    component=context.component or "unknown",
                    operation=context.operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
