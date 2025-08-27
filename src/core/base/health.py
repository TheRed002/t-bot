"""
Comprehensive health check system for all base components.

This module provides centralized health monitoring, aggregation,
and reporting for all components in the trading bot system.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.core.base.interfaces import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
)
from src.core.exceptions import HealthCheckError
from src.core.types.base import ConfigDict


class HealthCheckType(Enum):
    """Types of health checks."""

    LIVENESS = "liveness"  # Is the component alive?
    READINESS = "readiness"  # Is the component ready to serve?
    HEALTH = "health"  # Overall health status


class ComponentHealthInfo:
    """Health information for a registered component."""

    def __init__(
        self,
        component: HealthCheckable,
        name: str,
        check_interval: int = 30,
        timeout: float = 5.0,
        enabled: bool = True,
    ):
        self.component = component
        self.name = name
        self.check_interval = check_interval  # seconds
        self.timeout = timeout  # seconds
        self.enabled = enabled

        # Health tracking
        self.last_check_time: datetime | None = None
        self.last_health_result: HealthCheckResult | None = None
        self.last_readiness_result: HealthCheckResult | None = None
        self.last_liveness_result: HealthCheckResult | None = None

        # Failure tracking
        self.consecutive_failures = 0
        self.total_checks = 0
        self.total_failures = 0
        self.failure_threshold = 3

        # Performance tracking
        self.check_durations: list[float] = []
        self.max_duration_history = 100


class HealthCheckManager(BaseComponent):
    """
    Centralized health check manager for all system components.

    Provides:
    - Component registration and monitoring
    - Periodic health checks
    - Health aggregation and reporting
    - Alert generation on failures
    - Performance monitoring
    - Health check caching
    - Circuit breaker patterns for failed checks

    Example:
        ```python
        health_manager = HealthCheckManager(name="SystemHealthManager")

        # Register components
        health_manager.register_component(trading_service, "TradingService")
        health_manager.register_component(database_repo, "DatabaseRepo")

        # Start monitoring
        await health_manager.start()

        # Get health status
        overall_health = await health_manager.get_overall_health()
        component_health = await health_manager.get_component_health("TradingService")
        ```
    """

    def __init__(
        self,
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize health check manager.

        Args:
            name: Manager name
            config: Manager configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "HealthCheckManager", config, correlation_id)

        # Component registration
        self._components: dict[str, ComponentHealthInfo] = {}

        # Health checking
        self._check_tasks: dict[str, asyncio.Task] = {}
        self._global_check_task: asyncio.Task | None = None
        self._check_interval = 30  # seconds
        self._check_timeout = 5.0  # seconds

        # Health caching
        self._health_cache: dict[str, HealthCheckResult] = {}
        self._cache_ttl = 10  # seconds
        self._cache_timestamps: dict[str, datetime] = {}

        # Aggregation settings
        self._enable_auto_checks = True
        self._enable_caching = True
        self._parallel_checks = True
        self._max_parallel_checks = 10

        # Alert settings
        self._alert_callbacks: list[Callable[[str, HealthCheckResult], Awaitable[None]]] = []
        self._alert_on_status_change = True
        self._alert_on_consecutive_failures = 3

        # Performance tracking
        self._health_metrics: dict[str, Any] = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "average_check_time": 0.0,
            "last_full_check_time": None,
            "components_count": 0,
            "unhealthy_components": 0,
        }

        self._logger.debug("Health check manager initialized", manager=self._name)

    @property
    def health_metrics(self) -> dict[str, Any]:
        """Get health check performance metrics."""
        metrics = self._health_metrics.copy()
        metrics["components_count"] = len(self._components)
        metrics["running_checks"] = len([t for t in self._check_tasks.values() if not t.done()])
        return metrics

    # Component Registration
    def register_component(
        self,
        component: HealthCheckable,
        name: str,
        check_interval: int = 30,
        timeout: float = 5.0,
        enabled: bool = True,
    ) -> None:
        """
        Register component for health monitoring.

        Args:
            component: Component implementing HealthCheckable
            name: Unique name for the component
            check_interval: Check interval in seconds
            timeout: Check timeout in seconds
            enabled: Enable health checking for this component

        Raises:
            HealthCheckError: If component registration fails
        """
        if name in self._components:
            raise HealthCheckError(f"Component '{name}' already registered")

        try:
            health_info = ComponentHealthInfo(
                component=component,
                name=name,
                check_interval=check_interval,
                timeout=timeout,
                enabled=enabled,
            )

            self._components[name] = health_info

            # Start monitoring if manager is running and auto-checks enabled
            if self._is_running and self._enable_auto_checks and enabled:
                self._start_component_monitoring(name)

            self._logger.info(
                "Component registered for health monitoring",
                manager=self._name,
                component=name,
                interval=check_interval,
                timeout=timeout,
                enabled=enabled,
            )

        except Exception as e:
            raise HealthCheckError(f"Failed to register component '{name}': {e}") from e

    def unregister_component(self, name: str) -> None:
        """
        Unregister component from health monitoring.

        Args:
            name: Component name

        Raises:
            HealthCheckError: If component not found
        """
        if name not in self._components:
            raise HealthCheckError(f"Component '{name}' not registered")

        try:
            # Stop monitoring task
            if name in self._check_tasks:
                self._check_tasks[name].cancel()
                del self._check_tasks[name]

            # Remove from cache
            self._health_cache.pop(name, None)
            self._cache_timestamps.pop(name, None)

            # Remove component
            del self._components[name]

            self._logger.info(
                "Component unregistered from health monitoring",
                manager=self._name,
                component=name,
            )

        except Exception as e:
            raise HealthCheckError(f"Failed to unregister component '{name}': {e}") from e

    def enable_component_monitoring(self, name: str) -> None:
        """Enable monitoring for specific component."""
        if name not in self._components:
            raise HealthCheckError(f"Component '{name}' not registered")

        self._components[name].enabled = True

        if self._is_running and self._enable_auto_checks:
            self._start_component_monitoring(name)

        self._logger.info(
            "Component monitoring enabled",
            manager=self._name,
            component=name,
        )

    def disable_component_monitoring(self, name: str) -> None:
        """Disable monitoring for specific component."""
        if name not in self._components:
            raise HealthCheckError(f"Component '{name}' not registered")

        self._components[name].enabled = False

        # Stop monitoring task
        if name in self._check_tasks:
            self._check_tasks[name].cancel()
            del self._check_tasks[name]

        self._logger.info(
            "Component monitoring disabled",
            manager=self._name,
            component=name,
        )

    # Health Checking
    async def check_component_health(
        self,
        name: str,
        check_type: HealthCheckType = HealthCheckType.HEALTH,
        use_cache: bool = True,
    ) -> HealthCheckResult:
        """
        Check health of specific component.

        Args:
            name: Component name
            check_type: Type of health check to perform
            use_cache: Whether to use cached results

        Returns:
            HealthCheckResult for the component

        Raises:
            HealthCheckError: If component not found or check fails
        """
        if name not in self._components:
            raise HealthCheckError(f"Component '{name}' not registered")

        health_info = self._components[name]

        # Check cache first
        if use_cache and self._enable_caching:
            cached_result = self._get_cached_result(name, check_type)
            if cached_result:
                return cached_result

        try:
            start_time = datetime.now(timezone.utc)

            # Perform health check based on type
            if check_type == HealthCheckType.HEALTH:
                result = await asyncio.wait_for(
                    health_info.component.health_check(), timeout=health_info.timeout
                )
                health_info.last_health_result = result

            elif check_type == HealthCheckType.READINESS:
                result = await asyncio.wait_for(
                    health_info.component.ready_check(), timeout=health_info.timeout
                )
                health_info.last_readiness_result = result

            elif check_type == HealthCheckType.LIVENESS:
                result = await asyncio.wait_for(
                    health_info.component.live_check(), timeout=health_info.timeout
                )
                health_info.last_liveness_result = result

            # Record metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_check_metrics(health_info, execution_time, True)

            # Cache result
            if self._enable_caching:
                self._cache_result(name, check_type, result)

            # Handle status changes and alerts
            await self._handle_check_result(name, result)

            self._logger.debug(
                "Component health check completed",
                manager=self._name,
                component=name,
                check_type=check_type.value,
                status=result.status.value,
                execution_time=execution_time,
            )

            return result

        except asyncio.TimeoutError:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {health_info.timeout}s",
                check_time=datetime.now(timezone.utc),
            )

            self._record_check_metrics(health_info, health_info.timeout, False)
            await self._handle_check_result(name, result)

            return result

        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
                check_time=datetime.now(timezone.utc),
            )

            self._record_check_metrics(health_info, 0, False)
            await self._handle_check_result(name, result)

            self._logger.error(
                "Component health check failed",
                manager=self._name,
                component=name,
                error=str(e),
            )

            return result

    async def check_all_components(
        self,
        check_type: HealthCheckType = HealthCheckType.HEALTH,
        parallel: bool = True,
    ) -> dict[str, HealthCheckResult]:
        """
        Check health of all registered components.

        Args:
            check_type: Type of health check to perform
            parallel: Whether to run checks in parallel

        Returns:
            Dictionary mapping component names to health results
        """
        start_time = datetime.now(timezone.utc)
        results = {}

        enabled_components = [name for name, info in self._components.items() if info.enabled]

        if not enabled_components:
            return results

        try:
            if parallel and self._parallel_checks:
                # Run checks in parallel with semaphore
                semaphore = asyncio.Semaphore(self._max_parallel_checks)

                async def check_with_semaphore(component_name: str):
                    async with semaphore:
                        return await self.check_component_health(
                            component_name, check_type, use_cache=False
                        )

                # Execute all checks
                tasks = [check_with_semaphore(name) for name in enabled_components]

                check_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for name, result in zip(enabled_components, check_results, strict=False):
                    if isinstance(result, Exception):
                        results[name] = HealthCheckResult(
                            status=HealthStatus.UNHEALTHY,
                            message=f"Check failed: {result}",
                            check_time=datetime.now(timezone.utc),
                        )
                    else:
                        results[name] = result

            else:
                # Run checks sequentially
                for name in enabled_components:
                    try:
                        result = await self.check_component_health(
                            name, check_type, use_cache=False
                        )
                        results[name] = result
                    except Exception as e:
                        results[name] = HealthCheckResult(
                            status=HealthStatus.UNHEALTHY,
                            message=f"Check failed: {e}",
                            check_time=datetime.now(timezone.utc),
                        )

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self._health_metrics["last_full_check_time"] = datetime.now(timezone.utc)

            self._logger.info(
                "All components health check completed",
                manager=self._name,
                components_count=len(results),
                execution_time=execution_time,
                parallel=parallel,
            )

            return results

        except Exception as e:
            self._logger.error(
                "Failed to check all components health",
                manager=self._name,
                error=str(e),
            )
            raise HealthCheckError(f"Failed to check all components: {e}") from e

    async def get_overall_health(self) -> HealthCheckResult:
        """
        Get overall system health by aggregating component health.

        Returns:
            HealthCheckResult representing overall system health
        """
        try:
            # Get all component health results
            component_results = await self.check_all_components()

            if not component_results:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="No components registered",
                    check_time=datetime.now(timezone.utc),
                )

            # Aggregate status
            statuses = [result.status for result in component_results.values()]
            unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
            degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)
            healthy_count = sum(1 for s in statuses if s == HealthStatus.HEALTHY)

            # Determine overall status
            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
                message = f"{unhealthy_count} unhealthy components detected"
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
                message = f"{degraded_count} degraded components detected"
            else:
                overall_status = HealthStatus.HEALTHY
                message = "All components healthy"

            # Gather details
            details = {
                "components_total": len(component_results),
                "components_healthy": healthy_count,
                "components_degraded": degraded_count,
                "components_unhealthy": unhealthy_count,
                "component_statuses": {
                    name: result.status.value for name, result in component_results.items()
                },
            }

            # Add unhealthy component details
            if unhealthy_count > 0:
                details["unhealthy_components"] = {
                    name: result.message
                    for name, result in component_results.items()
                    if result.status == HealthStatus.UNHEALTHY
                }

            return HealthCheckResult(
                status=overall_status,
                message=message,
                details=details,
                check_time=datetime.now(timezone.utc),
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to determine overall health: {e}",
                check_time=datetime.now(timezone.utc),
            )

    # Monitoring and Alerts
    def add_alert_callback(
        self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]
    ) -> None:
        """
        Add callback for health status alerts.

        Args:
            callback: Async function called when health issues detected
        """
        self._alert_callbacks.append(callback)

        self._logger.info(
            "Alert callback registered",
            manager=self._name,
            callback_count=len(self._alert_callbacks),
        )

    def remove_alert_callback(
        self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]
    ) -> None:
        """Remove alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    async def _handle_check_result(self, component_name: str, result: HealthCheckResult) -> None:
        """Handle health check result and trigger alerts if needed."""
        health_info = self._components[component_name]

        # Update failure tracking
        if result.status == HealthStatus.UNHEALTHY:
            health_info.consecutive_failures += 1
            health_info.total_failures += 1
        else:
            health_info.consecutive_failures = 0

        health_info.total_checks += 1
        health_info.last_check_time = datetime.now(timezone.utc)

        # Check for alert conditions
        should_alert = False

        if self._alert_on_consecutive_failures > 0:
            if health_info.consecutive_failures >= self._alert_on_consecutive_failures:
                should_alert = True

        if self._alert_on_status_change:
            previous_result = getattr(health_info, "last_health_result", None)
            if previous_result and previous_result.status != result.status:
                should_alert = True

        # Trigger alerts
        if should_alert and self._alert_callbacks:
            for callback in self._alert_callbacks:
                try:
                    await callback(component_name, result)
                except Exception as e:
                    self._logger.error(
                        "Alert callback failed",
                        manager=self._name,
                        component=component_name,
                        callback=callback.__name__,
                        error=str(e),
                    )

    # Monitoring Tasks
    def _start_component_monitoring(self, component_name: str) -> None:
        """Start periodic monitoring for a component."""
        if component_name in self._check_tasks:
            return  # Already monitoring

        health_info = self._components[component_name]

        async def monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(health_info.check_interval)

                    if not health_info.enabled:
                        break

                    await self.check_component_health(component_name, use_cache=False)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._logger.error(
                        "Monitoring loop error",
                        manager=self._name,
                        component=component_name,
                        error=str(e),
                    )
                    await asyncio.sleep(5)  # Brief pause before retry

        task = asyncio.create_task(monitoring_loop())
        self._check_tasks[component_name] = task

        self._logger.debug(
            "Component monitoring started",
            manager=self._name,
            component=component_name,
        )

    # Caching
    def _get_cached_result(
        self, component_name: str, check_type: HealthCheckType
    ) -> HealthCheckResult | None:
        """Get cached health check result if still valid."""
        cache_key = f"{component_name}_{check_type.value}"

        if cache_key not in self._health_cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key)
        if not timestamp:
            return None

        # Check if cache is still valid
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self._cache_ttl:
            # Remove expired cache entry
            self._health_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            return None

        return self._health_cache[cache_key]

    def _cache_result(
        self, component_name: str, check_type: HealthCheckType, result: HealthCheckResult
    ) -> None:
        """Cache health check result."""
        cache_key = f"{component_name}_{check_type.value}"
        self._health_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

    def clear_cache(self, component_name: str | None = None) -> None:
        """
        Clear health check cache.

        Args:
            component_name: Specific component to clear, or None for all
        """
        if component_name:
            # Clear cache for specific component
            keys_to_remove = [
                key for key in self._health_cache.keys() if key.startswith(f"{component_name}_")
            ]
            for key in keys_to_remove:
                self._health_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        else:
            # Clear all cache
            self._health_cache.clear()
            self._cache_timestamps.clear()

        self._logger.debug(
            "Health check cache cleared",
            manager=self._name,
            component=component_name or "all",
        )

    # Information and Management
    def get_component_info(self, component_name: str) -> dict[str, Any] | None:
        """Get information about registered component."""
        if component_name not in self._components:
            return None

        health_info = self._components[component_name]

        return {
            "name": component_name,
            "enabled": health_info.enabled,
            "check_interval": health_info.check_interval,
            "timeout": health_info.timeout,
            "last_check_time": (
                health_info.last_check_time.isoformat() if health_info.last_check_time else None
            ),
            "consecutive_failures": health_info.consecutive_failures,
            "total_checks": health_info.total_checks,
            "total_failures": health_info.total_failures,
            "failure_rate": (
                health_info.total_failures / health_info.total_checks
                if health_info.total_checks > 0
                else 0
            ),
            "average_check_duration": (
                sum(health_info.check_durations) / len(health_info.check_durations)
                if health_info.check_durations
                else 0
            ),
            "last_health_status": (
                health_info.last_health_result.status.value
                if health_info.last_health_result
                else None
            ),
        }

    def get_all_component_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered components."""
        return {name: self.get_component_info(name) for name in self._components.keys()}

    def list_components(self, enabled_only: bool = False) -> list[str]:
        """
        List registered component names.

        Args:
            enabled_only: Only return enabled components

        Returns:
            List of component names
        """
        if enabled_only:
            return [name for name, info in self._components.items() if info.enabled]
        return list(self._components.keys())

    # Metrics
    def _record_check_metrics(
        self,
        health_info: ComponentHealthInfo,
        execution_time: float,
        success: bool,
    ) -> None:
        """Record health check metrics."""
        # Update component metrics
        health_info.check_durations.append(execution_time)
        if len(health_info.check_durations) > health_info.max_duration_history:
            health_info.check_durations.pop(0)

        # Update global metrics
        self._health_metrics["total_checks"] += 1

        if success:
            self._health_metrics["successful_checks"] += 1
        else:
            self._health_metrics["failed_checks"] += 1

        # Update average check time
        total_checks = self._health_metrics["total_checks"]
        current_avg = self._health_metrics["average_check_time"]
        self._health_metrics["average_check_time"] = (
            current_avg * (total_checks - 1) + execution_time
        ) / total_checks

    def get_metrics(self) -> dict[str, Any]:
        """Get combined component and health manager metrics."""
        metrics = super().get_metrics()
        metrics.update(self.health_metrics)
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        super().reset_metrics()

        # Reset global metrics
        self._health_metrics = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "average_check_time": 0.0,
            "last_full_check_time": None,
            "components_count": 0,
            "unhealthy_components": 0,
        }

        # Reset component metrics
        for health_info in self._components.values():
            health_info.total_checks = 0
            health_info.total_failures = 0
            health_info.consecutive_failures = 0
            health_info.check_durations.clear()

    # Configuration
    def configure_checks(
        self,
        enable_auto_checks: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 10,
        parallel_checks: bool = True,
        max_parallel_checks: int = 10,
        check_interval: int = 30,
        check_timeout: float = 5.0,
    ) -> None:
        """
        Configure health check settings.

        Args:
            enable_auto_checks: Enable automatic periodic checks
            enable_caching: Enable result caching
            cache_ttl: Cache time-to-live in seconds
            parallel_checks: Enable parallel checking
            max_parallel_checks: Maximum parallel check limit
            check_interval: Default check interval
            check_timeout: Default check timeout
        """
        self._enable_auto_checks = enable_auto_checks
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl
        self._parallel_checks = parallel_checks
        self._max_parallel_checks = max_parallel_checks
        self._check_interval = check_interval
        self._check_timeout = check_timeout

        self._logger.info(
            "Health check settings configured",
            manager=self._name,
            auto_checks=enable_auto_checks,
            caching=enable_caching,
            cache_ttl=cache_ttl,
            parallel=parallel_checks,
            max_parallel=max_parallel_checks,
        )

    def configure_alerts(
        self,
        alert_on_status_change: bool = True,
        alert_on_consecutive_failures: int = 3,
    ) -> None:
        """
        Configure alert settings.

        Args:
            alert_on_status_change: Alert when health status changes
            alert_on_consecutive_failures: Alert after N consecutive failures
        """
        self._alert_on_status_change = alert_on_status_change
        self._alert_on_consecutive_failures = alert_on_consecutive_failures

        self._logger.info(
            "Alert settings configured",
            manager=self._name,
            status_change_alerts=alert_on_status_change,
            consecutive_failure_threshold=alert_on_consecutive_failures,
        )

    # Lifecycle Management
    async def _do_start(self) -> None:
        """Start health check manager and monitoring tasks."""
        # Start monitoring for enabled components
        if self._enable_auto_checks:
            for name, health_info in self._components.items():
                if health_info.enabled:
                    self._start_component_monitoring(name)

        self._logger.info(
            "Health check manager started",
            manager=self._name,
            monitoring_tasks=len(self._check_tasks),
        )

    async def _do_stop(self) -> None:
        """Stop health check manager and cleanup resources."""
        # Cancel all monitoring tasks
        for task in self._check_tasks.values():
            task.cancel()

        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)

        self._check_tasks.clear()

        # Cancel global check task
        if self._global_check_task:
            self._global_check_task.cancel()
            try:
                await self._global_check_task
            except asyncio.CancelledError:
                pass

        # Clear cache
        self.clear_cache()

        self._logger.info("Health check manager stopped", manager=self._name)

    # Health Check Implementation
    async def _health_check_internal(self) -> HealthStatus:
        """Health check manager's own health check."""
        try:
            # Check if monitoring tasks are running properly
            failed_tasks = [
                name
                for name, task in self._check_tasks.items()
                if task.done() and not task.cancelled()
            ]

            if failed_tasks:
                self._logger.warning(
                    "Some monitoring tasks have failed",
                    manager=self._name,
                    failed_tasks=failed_tasks,
                )
                return HealthStatus.DEGRADED

            # Check overall system health
            overall_health = await self.get_overall_health()

            if overall_health.status == HealthStatus.UNHEALTHY:
                return HealthStatus.DEGRADED  # Manager is working, but system is unhealthy
            elif overall_health.status == HealthStatus.DEGRADED:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(
                "Health check manager health check failed",
                manager=self._name,
                error=str(e),
            )
            return HealthStatus.UNHEALTHY
