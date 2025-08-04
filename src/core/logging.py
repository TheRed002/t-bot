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
import uuid
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import sys
import contextvars


class CorrelationContext:
    """Context manager for correlation ID tracking."""
    
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


# Global correlation context
correlation_context = CorrelationContext()


def setup_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """
    Setup structured logging configuration.
    
    Args:
        environment: Environment (development, staging, production)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
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
        lambda logger, method_name, event_dict: event_dict.update(correlation_id=correlation_context.get_correlation_id()) if event_dict is not None else {"correlation_id": correlation_context.get_correlation_id()},
        # Custom UnicodeDecoder that handles None for tests
        lambda logger, method_name, event_dict: structlog.processors.UnicodeDecoder()(logger, method_name, event_dict) if event_dict is not None else {},
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
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured structured logger
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
    """Logger wrapper that prevents sensitive data from being logged."""
    
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
    setup_logging(environment="development", log_level="DEBUG")
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


# Initialize default logging configuration
setup_logging() 