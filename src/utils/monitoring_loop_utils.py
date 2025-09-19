"""
Shared monitoring loop patterns and utilities.

Extracted from duplicated code in BotMonitor, ResourceManager, and other monitoring services.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable
from abc import ABC, abstractmethod

from src.core.logging import get_logger

logger = get_logger(__name__)


class MonitoringLoopManager:
    """Manages standardized monitoring loops with consistent patterns."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = get_logger(f"{__name__}.{service_name}")
        self._running_loops: dict[str, asyncio.Task] = {}
        self._is_stopping = False

    async def start_monitoring_loop(
        self,
        loop_name: str,
        loop_function: Callable[[], Awaitable[None]],
        interval_seconds: float = 60.0,
        initial_delay: float = 0.0
    ) -> bool:
        """
        Start a standardized monitoring loop.

        Args:
            loop_name: Unique name for the loop
            loop_function: Async function to execute in loop
            interval_seconds: Interval between executions
            initial_delay: Delay before first execution

        Returns:
            True if started successfully, False otherwise
        """
        if loop_name in self._running_loops:
            self._logger.warning(f"Monitoring loop '{loop_name}' is already running")
            return False

        try:
            task = asyncio.create_task(
                self._monitoring_loop_wrapper(
                    loop_name,
                    loop_function,
                    interval_seconds,
                    initial_delay
                )
            )
            self._running_loops[loop_name] = task
            self._logger.info(f"Started monitoring loop: {loop_name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start monitoring loop {loop_name}: {e}")
            return False

    async def stop_monitoring_loop(self, loop_name: str) -> bool:
        """Stop a specific monitoring loop."""
        if loop_name not in self._running_loops:
            self._logger.warning(f"Monitoring loop '{loop_name}' is not running")
            return False

        try:
            task = self._running_loops[loop_name]
            task.cancel()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                self._logger.warning(f"Monitoring loop {loop_name} did not stop gracefully")

            del self._running_loops[loop_name]
            self._logger.info(f"Stopped monitoring loop: {loop_name}")
            return True

        except Exception as e:
            self._logger.error(f"Error stopping monitoring loop {loop_name}: {e}")
            return False

    async def stop_all_loops(self) -> None:
        """Stop all running monitoring loops."""
        self._is_stopping = True

        if not self._running_loops:
            return

        self._logger.info(f"Stopping {len(self._running_loops)} monitoring loops")

        # Cancel all tasks
        for loop_name, task in self._running_loops.items():
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._running_loops:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_loops.values(), return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self._logger.warning("Some monitoring loops did not stop gracefully")

        self._running_loops.clear()
        self._logger.info("All monitoring loops stopped")

    async def _monitoring_loop_wrapper(
        self,
        loop_name: str,
        loop_function: Callable[[], Awaitable[None]],
        interval_seconds: float,
        initial_delay: float
    ) -> None:
        """Wrapper that provides consistent error handling and timing."""
        try:
            # Initial delay
            if initial_delay > 0:
                await asyncio.sleep(initial_delay)

            consecutive_errors = 0
            max_consecutive_errors = 5

            while not self._is_stopping:
                loop_start_time = datetime.now(timezone.utc)

                try:
                    # Execute the monitoring function
                    await loop_function()

                    # Reset error counter on success
                    consecutive_errors = 0

                except asyncio.CancelledError:
                    self._logger.info(f"Monitoring loop {loop_name} was cancelled")
                    break

                except Exception as e:
                    consecutive_errors += 1
                    self._logger.error(
                        f"Error in monitoring loop {loop_name} "
                        f"(attempt {consecutive_errors}): {e}"
                    )

                    # Stop loop if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        self._logger.error(
                            f"Stopping monitoring loop {loop_name} after "
                            f"{max_consecutive_errors} consecutive errors"
                        )
                        break

                    # Exponential backoff on errors
                    error_delay = min(60.0, 2 ** consecutive_errors)
                    await asyncio.sleep(error_delay)
                    continue

                # Calculate sleep time to maintain consistent interval
                execution_time = (datetime.now(timezone.utc) - loop_start_time).total_seconds()
                sleep_time = max(0, interval_seconds - execution_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except Exception as e:
            self._logger.error(f"Fatal error in monitoring loop {loop_name}: {e}")
        finally:
            # Clean up the task reference
            if loop_name in self._running_loops:
                del self._running_loops[loop_name]

    def get_loop_status(self) -> dict[str, Any]:
        """Get status of all monitoring loops."""
        status = {
            'service_name': self.service_name,
            'total_loops': len(self._running_loops),
            'is_stopping': self._is_stopping,
            'loops': {}
        }

        for loop_name, task in self._running_loops.items():
            status['loops'][loop_name] = {
                'running': not task.done(),
                'cancelled': task.cancelled(),
                'exception': str(task.exception()) if task.done() and task.exception() else None
            }

        return status


class BaseMonitoringService(ABC):
    """Base class for services that need standardized monitoring loops."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = get_logger(f"{__name__}.{service_name}")
        self._loop_manager = MonitoringLoopManager(service_name)
        self._monitoring_config = self._get_monitoring_config()

    @abstractmethod
    def _get_monitoring_config(self) -> dict[str, Any]:
        """Get monitoring configuration for this service."""
        pass

    async def start_monitoring(self) -> None:
        """Start all monitoring loops for this service."""
        try:
            await self._setup_monitoring_loops()
            self._logger.info(f"Monitoring started for {self.service_name}")
        except Exception as e:
            self._logger.error(f"Failed to start monitoring for {self.service_name}: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop all monitoring loops for this service."""
        try:
            await self._loop_manager.stop_all_loops()
            self._logger.info(f"Monitoring stopped for {self.service_name}")
        except Exception as e:
            self._logger.error(f"Failed to stop monitoring for {self.service_name}: {e}")

    @abstractmethod
    async def _setup_monitoring_loops(self) -> None:
        """Setup service-specific monitoring loops."""
        pass

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get monitoring status for this service."""
        return self._loop_manager.get_loop_status()


class PerformanceMonitor:
    """Monitor and track performance metrics for functions and operations."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = get_logger(f"{__name__}.{service_name}")
        self._performance_data: dict[str, list[dict[str, Any]]] = {}

    async def track_operation(
        self,
        operation_name: str,
        operation_func: Callable[[], Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Track performance of an async operation.

        Args:
            operation_name: Name of the operation for tracking
            operation_func: Async function to execute and track
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation
        """
        start_time = datetime.now(timezone.utc)
        error_occurred = False
        error_details = None

        try:
            result = await operation_func(*args, **kwargs)
            return result

        except Exception as e:
            error_occurred = True
            error_details = str(e)
            raise

        finally:
            # Record performance data
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            performance_record = {
                'timestamp': start_time.isoformat(),
                'execution_time_seconds': execution_time,
                'error_occurred': error_occurred,
                'error_details': error_details,
                'operation_name': operation_name
            }

            # Store performance data
            if operation_name not in self._performance_data:
                self._performance_data[operation_name] = []

            self._performance_data[operation_name].append(performance_record)

            # Keep only last 100 records per operation
            if len(self._performance_data[operation_name]) > 100:
                self._performance_data[operation_name] = self._performance_data[operation_name][-100:]

            # Log slow operations
            if execution_time > 10.0:  # More than 10 seconds
                self._logger.warning(
                    f"Slow operation detected: {operation_name} took {execution_time:.2f}s"
                )

    def get_performance_summary(self, operation_name: str | None = None) -> dict[str, Any]:
        """
        Get performance summary for operations.

        Args:
            operation_name: Specific operation name, or None for all operations

        Returns:
            Performance summary data
        """
        try:
            if operation_name:
                if operation_name not in self._performance_data:
                    return {'operation_name': operation_name, 'records': 0}

                records = self._performance_data[operation_name]
                return self._calculate_operation_summary(operation_name, records)

            # Summary for all operations
            summary = {
                'service_name': self.service_name,
                'total_operations': len(self._performance_data),
                'operations': {}
            }

            for op_name, records in self._performance_data.items():
                summary['operations'][op_name] = self._calculate_operation_summary(op_name, records)

            return summary

        except Exception as e:
            self._logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}

    def _calculate_operation_summary(self, operation_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate summary statistics for an operation's performance records."""
        if not records:
            return {'operation_name': operation_name, 'records': 0}

        try:
            execution_times = [r['execution_time_seconds'] for r in records]
            error_count = sum(1 for r in records if r['error_occurred'])

            return {
                'operation_name': operation_name,
                'total_records': len(records),
                'error_count': error_count,
                'error_rate': error_count / len(records) if records else 0,
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'latest_execution': records[-1]['timestamp'] if records else None
            }

        except Exception as e:
            self._logger.error(f"Error calculating summary for {operation_name}: {e}")
            return {'operation_name': operation_name, 'error': str(e)}


class MetricsCollectionManager:
    """Manage standardized metrics collection patterns."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logger = get_logger(f"{__name__}.{service_name}")
        self._metrics_data: dict[str, list[dict[str, Any]]] = {}
        self._collection_intervals: dict[str, float] = {}

    def register_metric_collector(
        self,
        metric_name: str,
        collector_func: Callable[[], Awaitable[dict[str, Any]]],
        interval_seconds: float = 300.0  # 5 minutes default
    ) -> None:
        """
        Register a metric collector function.

        Args:
            metric_name: Name of the metric
            collector_func: Async function that collects and returns metric data
            interval_seconds: Collection interval
        """
        self._collection_intervals[metric_name] = interval_seconds
        self._logger.info(f"Registered metric collector: {metric_name}")

    async def collect_metric(self, metric_name: str, collector_func: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        """Collect a single metric and store it."""
        try:
            timestamp = datetime.now(timezone.utc)
            metric_data = await collector_func()

            if not isinstance(metric_data, dict):
                self._logger.warning(f"Metric {metric_name} returned non-dict data")
                return

            # Add timestamp to metric data
            metric_data['timestamp'] = timestamp.isoformat()
            metric_data['metric_name'] = metric_name

            # Store metric data
            if metric_name not in self._metrics_data:
                self._metrics_data[metric_name] = []

            self._metrics_data[metric_name].append(metric_data)

            # Keep only last 1000 records per metric
            if len(self._metrics_data[metric_name]) > 1000:
                self._metrics_data[metric_name] = self._metrics_data[metric_name][-1000:]

        except Exception as e:
            self._logger.error(f"Error collecting metric {metric_name}: {e}")

    def get_metric_data(self, metric_name: str, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get metric data for a specific metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours of data to return

        Returns:
            List of metric data points
        """
        if metric_name not in self._metrics_data:
            return []

        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_data = []

            for record in self._metrics_data[metric_name]:
                try:
                    record_time = datetime.fromisoformat(
                        record['timestamp'].replace('Z', '+00:00')
                    )
                    if record_time >= cutoff_time:
                        recent_data.append(record)
                except (KeyError, ValueError):
                    continue

            return recent_data

        except Exception as e:
            self._logger.error(f"Error retrieving metric data for {metric_name}: {e}")
            return []

    def cleanup_old_metrics(self, days: int = 7) -> int:
        """
        Clean up metric data older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of records cleaned up
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            cleaned_count = 0

            for metric_name, records in self._metrics_data.items():
                original_count = len(records)
                self._metrics_data[metric_name] = [
                    record for record in records
                    if datetime.fromisoformat(
                        record['timestamp'].replace('Z', '+00:00')
                    ) >= cutoff_time
                ]
                cleaned_count += original_count - len(self._metrics_data[metric_name])

            self._logger.info(f"Cleaned up {cleaned_count} old metric records")
            return cleaned_count

        except Exception as e:
            self._logger.error(f"Error cleaning up old metrics: {e}")
            return 0