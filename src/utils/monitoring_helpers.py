"""
Shared utility functions for monitoring components to eliminate code duplication.

This module extracts common patterns used across monitoring components including
HTTP session management, error handling, validation, and async task management.
"""

import asyncio
import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import aiohttp

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Use consistent error handling imports - no fallback for critical dependencies
from src.error_handling.context import ErrorContext


class HTTPSessionManager:
    """Shared HTTP session manager for monitoring components."""

    def __init__(self) -> None:
        self._sessions: dict[str, aiohttp.ClientSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_session(self, key: str = "default", **session_kwargs) -> aiohttp.ClientSession:
        """Get or create a shared HTTP session."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            if key not in self._sessions or self._sessions[key].closed:
                session_config = {
                    "connector": aiohttp.TCPConnector(
                        limit=20,
                        limit_per_host=10,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    ),
                    "timeout": aiohttp.ClientTimeout(total=30),
                    "raise_for_status": False,
                }
                session_config.update(session_kwargs)
                self._sessions[key] = aiohttp.ClientSession(**session_config)

        return self._sessions[key]

    async def close_all(self) -> None:
        """Close all sessions with proper timeout handling."""
        close_tasks = []
        for key, session in list(self._sessions.items()):
            if session and not session.closed:
                close_task = asyncio.create_task(self._safe_session_close(key, session))
                close_tasks.append(close_task)

        if close_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*close_tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Some sessions did not close within timeout")
                # Force close remaining sessions
                for key, session in list(self._sessions.items()):
                    if session and not session.closed:
                        try:
                            if hasattr(session, "_connector") and session._connector:
                                await session._connector.close()
                        except Exception as e:
                            logger.debug(f"Error force-closing session connector: {e}")

        self._sessions.clear()
        self._locks.clear()

    async def _safe_session_close(self, key: str, session) -> None:
        """Safely close a single session with timeout protection."""
        try:
            await asyncio.wait_for(session.close(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning(f"Session {key} close timed out")
            # Force close if timeout
            try:
                if hasattr(session, "_connector") and session._connector:
                    await asyncio.wait_for(session._connector.close(), timeout=1.0)
            except Exception as e:
                logger.debug(f"Error force-closing connector after timeout: {e}")
        except Exception as e:
            logger.warning(f"Error closing session {key}: {e}")
        finally:
            self._sessions.pop(key, None)


async def get_http_session(key: str = "default", session_manager: HTTPSessionManager | None = None, **kwargs) -> aiohttp.ClientSession:
    """Get a shared HTTP session using proper dependency injection.
    
    Args:
        key: Session key identifier
        session_manager: Injected session manager (required for clean architecture)
        **kwargs: Additional session parameters
        
    Returns:
        ClientSession: HTTP session
        
    Raises:
        ValidationError: If session manager not injected properly
    """
    if session_manager is None:
        # Try DI container as fallback but warn about violation
        logger.warning("HTTPSessionManager not injected - violates clean architecture. Inject from service layer.")
        try:
            from src.core.dependency_injection import injector
            session_manager = injector.resolve("HTTPSessionManager")
        except Exception as e:
            raise ValidationError(
                "HTTPSessionManager must be injected from service layer. "
                "Do not access DI container directly from utility functions.",
                error_code="SERV_001"
            ) from e
    
    return await session_manager.get_session(key, **kwargs)


async def cleanup_http_sessions(session_manager: HTTPSessionManager | None = None):
    """Cleanup all HTTP sessions with proper dependency injection.
    
    Args:
        session_manager: Injected session manager (should be provided by service layer)
    """
    if session_manager is None:
        # Fallback to DI but warn about violation
        logger.warning("HTTPSessionManager not injected for cleanup - violates clean architecture.")
        try:
            from src.core.dependency_injection import injector
            session_manager = injector.resolve("HTTPSessionManager")
        except Exception as e:
            logger.debug(f"No HTTPSessionManager to cleanup: {e}")
            return
    
    await session_manager.close_all()


def generate_correlation_id() -> str:
    """Generate a short correlation ID for tracking operations."""
    return str(uuid.uuid4())[:8]


def generate_fingerprint(data: dict[str, Any]) -> str:
    """Generate a consistent fingerprint from data for deduplication."""
    fingerprint_data = f"{sorted(data.items())}"
    return hashlib.md5(fingerprint_data.encode()).hexdigest()


async def create_error_context(
    component: str, operation: str, error: Exception, details: dict[str, Any] | None = None
) -> ErrorContext:
    """Create a standardized error context."""
    correlation_id = generate_correlation_id()

    context_details = {
        "correlation_id": correlation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
    }

    if details:
        context_details.update(details)

    return ErrorContext.from_exception(
        error=error,
        component=component,
        operation=operation,
        details=context_details,
    )


async def handle_error_with_fallback(error: Exception, error_handler: Any | None, context: ErrorContext) -> bool:
    """Handle error with fallback logic, returns True if handled successfully."""
    if not error_handler:
        return False

    try:
        if hasattr(error_handler, "handle_error"):
            await error_handler.handle_error(error, context)
            return True
        elif hasattr(error_handler, "handle_error_sync"):
            await error_handler.handle_error_sync(error, context)
            return True
        else:
            logger.error(f"Error handler has no valid methods. Correlation: {context.details.get('correlation_id')}")
            return False
    except Exception as handler_error:
        logger.error(f"Error handler failed: {handler_error}")
        return False


def validate_monitoring_parameter(
    value: Any, param_name: str, expected_type: type, allow_none: bool = False, validation_rule: str | None = None
) -> None:
    """Validate a parameter with standardized error messages."""
    if value is None and not allow_none:
        raise ValidationError(
            f"{param_name} cannot be None",
            field_name=param_name,
            field_value=value,
            expected_type=expected_type.__name__,
        )

    if value is not None and not isinstance(value, expected_type):
        # Handle tuple types properly - get the name from individual types in the tuple
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_type_name = f"one of {type_names}"
        else:
            expected_type_name = expected_type.__name__

        raise ValidationError(
            f"Invalid {param_name} parameter",
            field_name=param_name,
            field_value=value,
            expected_type=expected_type_name,
        )

    if validation_rule and value is not None:
        if validation_rule == "positive" and value <= 0:
            raise ValidationError(
                f"{param_name} must be positive",
                field_name=param_name,
                field_value=value,
                validation_rule="must be positive",
            )
        elif validation_rule == "non_negative" and value < 0:
            raise ValidationError(
                f"{param_name} must be non-negative",
                field_name=param_name,
                field_value=value,
                validation_rule="must be non-negative",
            )


class AsyncTaskManager:
    """Manages async task lifecycle with proper cleanup."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

    def create_task(self, coro, name: str | None = None) -> asyncio.Task:
        """Create a managed task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done_callback)
        return task

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Handle task completion."""
        self._tasks.discard(task)
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Task {task.get_name()} failed: {exc}")
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown all managed tasks with proper cleanup."""
        self._shutdown_event.set()

        if not self._tasks:
            return

        # Create list of tasks to cancel
        tasks_to_cancel = list(self._tasks)

        # Cancel all tasks
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Some {self.component_name} tasks did not complete within timeout")
                # Force cleanup of remaining tasks
                for task in tasks_to_cancel:
                    if not task.done():
                        try:
                            task.cancel()
                        except Exception as e:
                            logger.debug(f"Error cancelling task during cleanup: {e}")

        self._tasks.clear()

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()


@asynccontextmanager
async def http_request_with_retry(
    session: aiohttp.ClientSession, method: str, url: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs
):
    """HTTP request with retry logic and proper cleanup."""
    last_error = None

    for attempt in range(max_retries):
        try:
            # Add timeout to individual request if not specified
            if "timeout" not in kwargs:
                kwargs["timeout"] = aiohttp.ClientTimeout(total=30)

            async with getattr(session, method.lower())(url, **kwargs) as response:
                yield response
                return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                logger.warning(f"HTTP {method} {url} failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"HTTP {method} {url} failed after {max_retries} attempts: {e}")
        except Exception as e:
            # Non-retryable error
            logger.error(f"HTTP {method} {url} failed with non-retryable error: {e}")
            last_error = e
            break

    if last_error:
        raise last_error


def safe_duration_parse(duration_str: str) -> int:
    """Parse duration string to minutes with validation."""
    if not duration_str or not isinstance(duration_str, str):
        raise ValidationError("Duration must be a non-empty string")

    duration = duration_str.strip().lower()

    try:
        if duration.endswith("s"):
            seconds = int(duration[:-1])
            if seconds <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, seconds // 60)  # Convert to minutes, minimum 1
        elif duration.endswith("m"):
            minutes = int(duration[:-1])
            if minutes <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, minutes)  # Minimum 1 minute
        elif duration.endswith("h"):
            hours = int(duration[:-1])
            if hours <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, hours * 60)  # Convert to minutes
        elif duration.endswith("d"):
            days = int(duration[:-1])
            if days <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, days * 24 * 60)  # Convert to minutes
        elif duration.endswith("w"):
            weeks = int(duration[:-1])
            if weeks <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, weeks * 7 * 24 * 60)  # Convert to minutes
        else:
            # Try parsing as raw number (assume minutes)
            minutes = int(duration)
            if minutes <= 0:
                raise ValidationError(f"Duration must be positive: {duration}")
            return max(1, minutes)
    except (ValueError, IndexError, TypeError) as e:
        if isinstance(e, ValueError) and "must be positive" in str(e):
            raise  # Re-raise our custom validation error
        raise ValidationError(
            f"Invalid duration format: '{duration}'. Use format like '5m', '1h', '30s', '1d'. " f"Original error: {e}"
        ) from e


def log_unusual_values(value: float, threshold: float, metric_name: str, unit: str = ""):
    """Log warnings for unusual metric values."""
    if abs(value) > threshold:
        unit_str = f" {unit}" if unit else ""
        logger.warning(f"Unusually large {metric_name}: {value:,.2f}{unit_str}")


class MetricValueProcessor:
    """Processes and validates metric values consistently."""

    @staticmethod
    def process_financial_value(
        value: Any, metric_name: str, decimal_places: int = 8, max_value: float | None = None
    ) -> float:
        """Process and validate financial metric values."""
        from src.monitoring.financial_precision import safe_decimal_to_float

        if isinstance(value, Decimal):
            result = safe_decimal_to_float(value, metric_name, decimal_places)
        else:
            result = round(float(value), decimal_places)

        if max_value and abs(result) > max_value:
            log_unusual_values(result, max_value, metric_name, "USD")

        return result

    @staticmethod
    def process_latency_value(value: float, metric_name: str) -> float:
        """Process latency values with validation."""
        if value < 0:
            raise ValidationError(f"Latency cannot be negative for {metric_name}")

        # Warn on very high latency
        if value > 60:  # > 60 seconds
            logger.warning(f"Very high latency for {metric_name}: {value:.2f}s")

        return round(value, 6)  # Microsecond precision


class SystemMetricsCollector:
    """Shared system metrics collection to eliminate duplication."""
    
    @staticmethod
    async def collect_system_metrics() -> dict[str, Any]:
        """
        Collect system metrics using async-safe operations.
        
        Returns:
            Dictionary of system metrics
        """
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available - skipping system metrics")
            return {}
            
        try:
            loop = asyncio.get_event_loop()
            
            # Run all psutil operations concurrently in thread pool
            tasks = [
                loop.run_in_executor(None, psutil.cpu_percent, None),
                loop.run_in_executor(None, psutil.virtual_memory),
                loop.run_in_executor(None, psutil.disk_io_counters),
                loop.run_in_executor(None, psutil.net_io_counters),
                loop.run_in_executor(None, psutil.Process),
            ]
            
            # Wait for all tasks to complete with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5.0
            )
            
            cpu_percent, memory, disk_io, network_io, process = results
            
            # Handle any exceptions from individual operations
            if isinstance(cpu_percent, Exception):
                logger.warning(f"Failed to collect CPU metrics: {cpu_percent}")
                cpu_percent = 0.0
                
            if isinstance(memory, Exception):
                logger.warning(f"Failed to collect memory metrics: {memory}")
                memory = None
                
            if isinstance(disk_io, Exception):
                logger.warning(f"Failed to collect disk I/O metrics: {disk_io}")
                disk_io = None
                
            if isinstance(network_io, Exception):
                logger.warning(f"Failed to collect network I/O metrics: {network_io}")
                network_io = None
                
            if isinstance(process, Exception):
                logger.warning(f"Failed to get process info: {process}")
                process = None
            
            # Build metrics dictionary
            metrics = {
                'cpu_percent': float(cpu_percent) if cpu_percent is not None else 0.0,
            }
            
            if memory:
                metrics.update({
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'memory_used': memory.used,
                    'memory_total': memory.total,
                })
                
            if disk_io:
                metrics.update({
                    'disk_read_bytes': disk_io.read_bytes,
                    'disk_write_bytes': disk_io.write_bytes,
                    'disk_read_count': disk_io.read_count,
                    'disk_write_count': disk_io.write_count,
                })
                
            if network_io:
                metrics.update({
                    'network_bytes_sent': network_io.bytes_sent,
                    'network_bytes_recv': network_io.bytes_recv,
                    'network_packets_sent': network_io.packets_sent,
                    'network_packets_recv': network_io.packets_recv,
                })
                
            if process:
                try:
                    # Get process-specific metrics in thread pool
                    process_tasks = [
                        loop.run_in_executor(None, process.memory_info),
                        loop.run_in_executor(None, process.cpu_percent),
                        loop.run_in_executor(None, process.num_threads),
                    ]
                    
                    proc_results = await asyncio.wait_for(
                        asyncio.gather(*process_tasks, return_exceptions=True),
                        timeout=2.0
                    )
                    
                    proc_memory, proc_cpu, proc_threads = proc_results
                    
                    if not isinstance(proc_memory, Exception):
                        metrics.update({
                            'process_memory_rss': proc_memory.rss,
                            'process_memory_vms': proc_memory.vms,
                        })
                    
                    if not isinstance(proc_cpu, Exception):
                        metrics['process_cpu_percent'] = proc_cpu
                        
                    if not isinstance(proc_threads, Exception):
                        metrics['process_thread_count'] = proc_threads
                        
                except Exception as e:
                    logger.warning(f"Failed to collect process metrics: {e}")
            
            return metrics
            
        except asyncio.TimeoutError:
            logger.warning("System metrics collection timed out")
            return {}
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
