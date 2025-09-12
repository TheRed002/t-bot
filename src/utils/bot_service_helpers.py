"""
Bot service helper utilities for common patterns in bot_management module.

This module extracts commonly duplicated patterns from bot_management files:
- Service dependency resolution
- Error handling setup
- Monitoring and metrics initialization
- State management patterns
- Async task management utilities
"""

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any, TypeVar

from src.core.logging import get_logger

T = TypeVar("T")


def create_fallback_decorator(name: str, func_type: str = "sync") -> Callable:
    """
    Create fallback decorators for missing utils.decorators functionality.

    Args:
        name: Name of the decorator
        func_type: Type of function ("sync" or "async")

    Returns:
        Callable: Fallback decorator
    """

    def decorator(func: Callable) -> Callable:
        if func_type == "async":

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = logging.getLogger(func.__module__)
                if name == "log_calls":
                    logger.info(f"Calling {func.__name__}")
                elif name == "time_execution":
                    start = time.time()
                    result = await func(*args, **kwargs)
                    logger.info(f"{func.__name__} took {time.time() - start:.3f}s")
                    return result
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger = logging.getLogger(func.__module__)
                if name == "log_calls":
                    logger.info(f"Calling {func.__name__}")
                elif name == "time_execution":
                    start = time.time()
                    result = func(*args, **kwargs)
                    logger.info(f"{func.__name__} took {time.time() - start:.3f}s")
                    return result
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def safe_import_decorators() -> dict[str, Callable]:
    """
    Safely import decorators with fallback implementations.

    Returns:
        Dict[str, Callable]: Dictionary of decorator functions
    """
    decorators = {}

    try:
        from src.utils.decorators import log_calls, time_execution

        # Validate imported decorators are callable
        if not callable(log_calls):
            raise ImportError(f"log_calls is not callable: {type(log_calls)}")
        if not callable(time_execution):
            raise ImportError(f"time_execution is not callable: {type(time_execution)}")

        decorators["log_calls"] = log_calls
        decorators["time_execution"] = time_execution

    except ImportError as e:
        # Create fallback decorators
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import decorators, using fallback: {e}")

        decorators["log_calls"] = create_fallback_decorator("log_calls", "sync")
        decorators["time_execution"] = create_fallback_decorator("time_execution", "sync")

    return decorators


def safe_import_error_handling() -> dict[str, Any]:
    """
    Safely import error handling components with proper structure.

    Returns:
        Dict[str, Any]: Dictionary of error handling components
    """
    components = {}

    try:
        from src.error_handling import (
            ErrorSeverity,
            FallbackStrategy,
            get_global_error_handler,
            with_circuit_breaker,
            with_error_context,
            with_fallback,
            with_retry,
        )

        components.update(
            {
                "ErrorSeverity": ErrorSeverity,
                "FallbackStrategy": FallbackStrategy,
                "get_global_error_handler": get_global_error_handler,
                "with_circuit_breaker": with_circuit_breaker,
                "with_error_context": with_error_context,
                "with_fallback": with_fallback,
                "with_retry": with_retry,
            }
        )

    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import error handling components: {e}")

        # Create minimal fallback implementations
        class MockErrorSeverity:
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            CRITICAL = "critical"

        class MockFallbackStrategy:
            RETURN_DEFAULT = "return_default"
            RETURN_EMPTY = "return_empty"

        def mock_error_handler():
            class MockHandler:
                async def handle_error(self, error, context, severity="medium"):
                    logger = logging.getLogger(__name__)
                    logger.error(f"Mock error handler: {error} (severity: {severity})")

            return MockHandler()

        def mock_decorator(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        components.update(
            {
                "ErrorSeverity": MockErrorSeverity,
                "FallbackStrategy": MockFallbackStrategy,
                "get_global_error_handler": mock_error_handler,
                "with_circuit_breaker": mock_decorator,
                "with_error_context": mock_decorator,
                "with_fallback": mock_decorator,
                "with_retry": mock_decorator,
            }
        )

    return components


def safe_import_monitoring() -> dict[str, Any]:
    """
    Safely import monitoring components with fallback implementations.

    Returns:
        Dict[str, Any]: Dictionary of monitoring components
    """
    components = {}

    try:
        from src.monitoring import TradingMetrics, get_metrics_collector, get_tracer

        components.update(
            {
                "TradingMetrics": TradingMetrics,
                "get_tracer": get_tracer,
                "get_metrics_collector": get_metrics_collector,
            }
        )

    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import monitoring components: {e}")

        # Create fallback implementations
        components.update(
            {
                "TradingMetrics": None,
                "get_tracer": lambda x: None,
                "get_metrics_collector": lambda: None,
            }
        )

    return components


def resolve_service_dependencies(
    service_instance: Any,
    dependency_mapping: dict[str, str],
    injected_services: dict[str, Any] | None = None
) -> None:
    """
    Resolve service dependencies using proper dependency injection pattern.

    Args:
        service_instance: Service instance to inject dependencies into
        dependency_mapping: Mapping of attribute names to service names
        injected_services: Pre-injected services (required from service layer)
    """
    logger = get_logger(service_instance.__class__.__module__)
    injected_services = injected_services or {}

    for attr_name, service_name in dependency_mapping.items():
        current_service = getattr(service_instance, attr_name, None)

        # Only resolve services that weren't injected
        if not current_service:
            if service_name in injected_services:
                setattr(service_instance, attr_name, injected_services[service_name])
                logger.debug(f"Injected {service_name} from service layer")
            else:
                logger.warning(
                    f"Service {service_name} not provided - should be injected from service layer"
                )


async def safe_close_connection(
    connection: Any, connection_name: str = "connection", timeout: float = 5.0
) -> None:
    """
    Safely close a connection with timeout protection.

    Args:
        connection: Connection object to close
        connection_name: Name for logging purposes
        timeout: Timeout in seconds
    """
    if connection and hasattr(connection, "close"):
        logger = get_logger(__name__)
        try:
            await asyncio.wait_for(connection.close(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"{connection_name} close timeout")
        except Exception as e:
            logger.debug(f"Failed to close {connection_name}: {e}")


async def safe_close_connections(
    connections: list[Any], connection_name: str = "connections", timeout: float = 5.0
) -> None:
    """
    Safely close multiple connections concurrently with timeout protection.

    Args:
        connections: List of connection objects to close
        connection_name: Name for logging purposes
        timeout: Timeout in seconds per connection
    """
    if not connections:
        return

    logger = get_logger(__name__)
    close_tasks = []

    for conn in connections:
        if conn and hasattr(conn, "close"):
            try:
                close_tasks.append(asyncio.wait_for(conn.close(), timeout=timeout))
            except Exception as e:
                logger.debug(f"Error preparing {connection_name} close task: {e}")

    if close_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*close_tasks, return_exceptions=True),
                timeout=timeout * len(close_tasks),
            )
        except asyncio.TimeoutError:
            logger.warning(f"{connection_name} close timeout")


def create_resource_usage_entry(
    resource_type: Any, total_allocated: Decimal, total_used: Decimal, limit: Decimal
) -> dict[str, Any]:
    """
    Create a standardized resource usage entry.

    Args:
        resource_type: Type of resource
        total_allocated: Total allocated amount
        total_used: Total used amount
        limit: Resource limit

    Returns:
        Dict[str, Any]: Resource usage entry
    """
    from datetime import datetime, timezone

    return {
        "timestamp": datetime.now(timezone.utc),
        "total_allocated": total_allocated,
        "total_used": total_used,
        "usage_percentage": ((total_used / limit) if limit > 0 else Decimal("0.0")),
    }


def safe_record_metric(
    metrics_collector: Any,
    metric_name: str,
    value: Any,
    labels: dict[str, str] | None = None,
    metric_type: str = "gauge",
) -> None:
    """
    Safely record a metric with error handling.

    Args:
        metrics_collector: Metrics collector instance
        metric_name: Name of the metric
        value: Metric value
        labels: Optional metric labels
        metric_type: Type of metric (gauge, counter, etc.)
    """
    if not metrics_collector:
        return

    try:
        if metric_type == "gauge":
            metrics_collector.gauge(metric_name, value, labels=labels or {})
        elif metric_type == "counter":
            metrics_collector.increment(metric_name, labels=labels or {})
        else:
            # Try generic record method
            if hasattr(metrics_collector, "record"):
                metrics_collector.record(metric_name, value, labels=labels or {})

    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"Failed to record metric {metric_name}: {e}")


async def execute_with_timeout_and_cleanup(
    operation: Awaitable[Any],
    timeout: float,
    cleanup_func: Callable | None = None,
    operation_name: str = "operation",
) -> Any:
    """
    Execute an operation with timeout and automatic cleanup.

    Args:
        operation: Async operation to execute
        timeout: Timeout in seconds
        cleanup_func: Optional cleanup function to call on timeout/error
        operation_name: Name for logging

    Returns:
        Operation result

    Raises:
        asyncio.TimeoutError: If operation times out
    """
    logger = get_logger(__name__)

    try:
        return await asyncio.wait_for(operation, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"{operation_name} timed out after {timeout}s")
        if cleanup_func:
            try:
                await cleanup_func()
            except Exception as e:
                logger.debug(f"Cleanup failed after {operation_name} timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        if cleanup_func:
            try:
                await cleanup_func()
            except Exception as cleanup_error:
                logger.debug(f"Cleanup failed after {operation_name} error: {cleanup_error}")
        raise


def create_bot_state_data(
    bot_id: str,
    status: str,
    configuration: dict[str, Any],
    allocated_capital: str,
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create standardized bot state data structure.

    Args:
        bot_id: Bot identifier
        status: Bot status
        configuration: Bot configuration
        allocated_capital: Allocated capital amount
        additional_fields: Optional additional fields

    Returns:
        Dict[str, Any]: Bot state data
    """
    from datetime import datetime, timezone

    state_data = {
        "bot_id": bot_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "allocated_capital": allocated_capital,
        "current_positions": {},
        "active_orders": {},
        "last_heartbeat": None,
        "error_count": 0,
        "restart_count": 0,
    }

    if additional_fields:
        state_data.update(additional_fields)

    return state_data


def create_bot_metrics_data(
    bot_id: str, additional_fields: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Create standardized bot metrics data structure.

    Args:
        bot_id: Bot identifier
        additional_fields: Optional additional fields

    Returns:
        Dict[str, Any]: Bot metrics data
    """
    from datetime import datetime, timezone

    metrics_data = {
        "bot_id": bot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_trades": 0,
        "profitable_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "uptime_percentage": 0.0,
        "error_count": 0,
        "last_heartbeat": None,
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "api_calls_count": 0,
    }

    if additional_fields:
        metrics_data.update(additional_fields)

    return metrics_data


class ServiceHealthChecker:
    """Helper for standardized service health checks."""

    @staticmethod
    async def check_service_health(
        service: Any, service_name: str, default_healthy: bool = True
    ) -> dict[str, Any]:
        """
        Check health of a service with standardized output.

        Args:
            service: Service instance to check
            service_name: Name of the service
            default_healthy: Default health status if no health_check method

        Returns:
            Dict[str, Any]: Health check result
        """
        if not service:
            return {
                "service": service_name,
                "healthy": False,
                "status": "service_unavailable",
                "details": "Service not available",
            }

        try:
            if hasattr(service, "health_check"):
                health_result = await service.health_check()
                return {
                    "service": service_name,
                    "healthy": health_result.get("status") == "healthy",
                    "status": health_result.get("status", "unknown"),
                    "details": health_result,
                }
            else:
                # Basic health check - just verify service exists
                return {
                    "service": service_name,
                    "healthy": default_healthy,
                    "status": "service_available" if default_healthy else "no_health_check",
                    "details": {
                        "status": ("service available" if service else "service unavailable")
                    },
                }
        except Exception as e:
            return {
                "service": service_name,
                "healthy": False,
                "status": "health_check_error",
                "error": str(e),
            }


def batch_process_async(
    items: list[T], process_func: Callable, batch_size: int = 5, delay_between_batches: float = 2.0
) -> Callable:
    """
    Create a batch processor for async operations.

    Args:
        items: Items to process
        process_func: Async function to process each item
        batch_size: Size of each batch
        delay_between_batches: Delay in seconds between batches

    Returns:
        Callable: Async function that processes all items in batches
    """

    async def batch_processor():
        results = {}

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            # Process batch concurrently
            tasks = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Record results
            for item, result in zip(batch, batch_results, strict=False):
                results[str(item)] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    logger = get_logger(__name__)
                    logger.error(f"Failed to process item: {result}")

            # Delay between batches
            if i + batch_size < len(items):
                await asyncio.sleep(delay_between_batches)

        return results

    return batch_processor
