"""
Structured logging system for the trading bot framework.

This module provides comprehensive logging with correlation tracking, performance
monitoring, and secure logging practices. Configured for JSON formatting in production.

Features:
- Structured JSON logging for production
- Correlation ID tracking for request tracing
- Performance logging decorators
- Secure logging (no sensitive data)
- Log rotation and retention policies
"""

import structlog
import logging
import logging.handlers
import uuid
import time
import functools
from typing import Optional, Dict, Any, Callable, cast
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import sys
import contextvars
from pathlib import Path


class CorrelationContext:
    """Context manager for correlation ID tracking.

    Provides thread-safe correlation ID management for request tracing
    across the entire application using contextvars.

    Attributes:
        correlation_id: Current correlation ID (optional)
    """

    def __init__(self):
        self.correlation_id: Optional[str] = None
        self._context = contextvars.ContextVar("correlation_id", default=None)

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self.correlation_id = correlation_id
        self._context.set(correlation_id)

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return self._context.get()

    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())

    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID tracking."""
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()

        token = self._context.set(correlation_id)
        try:
            yield correlation_id
        finally:
            self._context.reset(token)


def _add_correlation_id(logger: Any, method_name: str,
                        event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add correlation ID to event dict."""
    if event_dict is not None:
        event_dict.update(
            correlation_id=correlation_context.get_correlation_id())
        return event_dict
    return {"correlation_id": correlation_context.get_correlation_id()}


def _safe_unicode_decoder(logger: Any, method_name: str,
                          event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Safe unicode decoder for event dict."""
    if event_dict is not None:
        return cast(Dict[str, Any], structlog.processors.UnicodeDecoder()(
            logger, method_name, event_dict))
    return {}


# Global correlation context
correlation_context = CorrelationContext()


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    retention_days: int = 30
) -> None:
    """Setup structured logging configuration with rotation and retention.

    Args:
        environment: Environment (development, staging, production)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (None for stdout only)
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep
        retention_days: Days to retain log files
    """
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Add correlation ID processor with proper None handling
        _add_correlation_id,
        # Custom UnicodeDecoder that handles None for tests
        _safe_unicode_decoder,
    ]

    # Add JSON formatting for production
    if environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging with rotation if file specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Configure logging with both handlers
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level.upper()),
            handlers=[file_handler, console_handler]
        )

        # Clean up old log files based on retention policy
        _cleanup_old_logs(log_path.parent, log_path.stem, retention_days)
    else:
        # Console only
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper()),
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structured logger with correlation ID support
    """
    return structlog.get_logger(name)


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function performance metrics.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        correlation_id = correlation_context.get_correlation_id()

        logger.info(
            "Function execution started",
            function_name=func.__name__,
            correlation_id=correlation_id,
        )

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.info(
                "Function execution completed",
                function_name=func.__name__,
                execution_time_ms=execution_time * 1000,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Function execution failed",
                function_name=func.__name__,
                execution_time_ms=execution_time * 1000,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            raise

    return wrapper


def log_async_performance(func: Callable) -> Callable:
    """
    Decorator to log async function performance metrics.

    Args:
        func: Async function to decorate

    Returns:
        Decorated async function with performance logging
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        correlation_id = correlation_context.get_correlation_id()

        logger.info(
            "Async function execution started",
            function_name=func.__name__,
            correlation_id=correlation_id,
        )

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.info(
                "Async function execution completed",
                function_name=func.__name__,
                execution_time_ms=execution_time * 1000,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Async function execution failed",
                function_name=func.__name__,
                execution_time_ms=execution_time * 1000,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            raise

    return wrapper


class SecureLogger:
    """Logger wrapper that prevents sensitive data from being logged.

    Automatically sanitizes sensitive fields like passwords, API keys,
    and tokens before logging to prevent accidental exposure.

    Attributes:
        logger: Underlying structured logger
        sensitive_fields: Set of field names to sanitize
    """

    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
        self.sensitive_fields = {
            'password', 'secret', 'key', 'token', 'api_key', 'private_key',
            'access_token', 'refresh_token', 'authorization', 'auth'
        }

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from logging."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, str) and any(field in key.lower() for field in self.sensitive_fields):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized

    def info(self, message: str, **kwargs) -> None:
        """Log info message with sanitized data."""
        sanitized_kwargs = self._sanitize_data(kwargs)
        self.logger.info(message, **sanitized_kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with sanitized data."""
        sanitized_kwargs = self._sanitize_data(kwargs)
        self.logger.warning(message, **sanitized_kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with sanitized data."""
        sanitized_kwargs = self._sanitize_data(kwargs)
        self.logger.error(message, **sanitized_kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with sanitized data."""
        sanitized_kwargs = self._sanitize_data(kwargs)
        self.logger.critical(message, **sanitized_kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with sanitized data."""
        sanitized_kwargs = self._sanitize_data(kwargs)
        self.logger.debug(message, **sanitized_kwargs)


def get_secure_logger(name: str) -> SecureLogger:
    """
    Get a secure logger instance that prevents sensitive data logging.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Secure logger instance
    """
    logger = get_logger(name)
    return SecureLogger(logger)


# TODO: Remove in production - Debug logging configuration
def setup_debug_logging() -> None:
    """Setup debug logging for development."""
    setup_development_logging()
    logger = get_logger(__name__)
    logger.info("Debug logging enabled")


# Performance monitoring utilities
class PerformanceMonitor:
    """Performance monitoring utility for tracking operation metrics."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.logger = get_logger(__name__)

    def __enter__(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.logger.info(
            "Performance monitoring started",
            operation=self.operation_name,
            correlation_id=correlation_context.get_correlation_id(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End performance monitoring."""
        if self.start_time:
            execution_time = time.time() - self.start_time

            if exc_type is None:
                self.logger.info(
                    "Performance monitoring completed",
                    operation=self.operation_name,
                    execution_time_ms=execution_time * 1000,
                    correlation_id=correlation_context.get_correlation_id(),
                )
            else:
                self.logger.error(
                    "Performance monitoring failed",
                    operation=self.operation_name,
                    execution_time_ms=execution_time * 1000,
                    error=str(exc_val),
                    error_type=exc_type.__name__,
                    correlation_id=correlation_context.get_correlation_id(),
                )


def _cleanup_old_logs(
        log_dir: Path,
        log_name: str,
        retention_days: int) -> None:
    """Clean up old log files based on retention policy.

    Args:
        log_dir: Directory containing log files
        log_name: Base name of log files
        retention_days: Number of days to retain log files
    """
    import time
    from datetime import datetime, timedelta

    if not log_dir.exists():
        return

    cutoff_time = time.time() - (retention_days * 24 * 3600)

    # Find and remove old log files
    for log_file in log_dir.glob(f"{log_name}*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Removed old log file: {log_file}")
        except (OSError, IOError) as e:
            print(f"Failed to remove old log file {log_file}: {e}")


def setup_production_logging(
    log_dir: str = "logs",
    app_name: str = "trading-bot"
) -> None:
    """Setup production logging with file rotation and retention.

    Args:
        log_dir: Directory for log files
        app_name: Application name for log file naming
    """
    log_file = f"{log_dir}/{app_name}.log"
    setup_logging(
        environment="production",
        log_level="INFO",
        log_file=log_file,
        max_bytes=50 * 1024 * 1024,  # 50MB per file
        backup_count=10,  # Keep 10 backup files
        retention_days=90  # Retain for 90 days
    )


def setup_development_logging() -> None:
    """Setup development logging with debug level and console output."""
    setup_logging(
        environment="development",
        log_level="DEBUG",
        log_file=None  # Console only for development
    )


# Initialize default logging configuration
setup_logging()
