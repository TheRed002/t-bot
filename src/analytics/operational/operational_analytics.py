"""
Operational Analytics System.

This module provides comprehensive operational analytics and system monitoring
for trading infrastructure, including performance metrics, error tracking,
and health monitoring.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import psutil

from src.analytics.types import (
    AlertSeverity,
    AnalyticsConfiguration,
    OperationalMetrics,
)
from src.base import BaseComponent
from src.monitoring.metrics import get_metrics_collector
from src.utils.datetime_utils import get_current_utc_timestamp
from src.utils.decimal_utils import safe_decimal


class OperationalAnalyticsEngine(BaseComponent):
    """
    Comprehensive operational analytics engine.

    Provides institutional-grade operational monitoring including:
    - System performance metrics and health monitoring
    - Order execution statistics and quality analysis
    - Market data quality monitoring and latency tracking
    - Strategy uptime and reliability metrics
    - Error rate and exception tracking
    - Infrastructure performance and capacity monitoring
    """

    def __init__(self, config: AnalyticsConfiguration):
        """
        Initialize operational analytics engine.

        Args:
            config: Analytics configuration
        """
        super().__init__()
        self.config = config
        self.metrics_collector = get_metrics_collector()

        # System metrics storage
        self._system_metrics: deque = deque(maxlen=1440)  # 24 hours of minute data
        self._performance_metrics: deque = deque(maxlen=1440)
        self._error_metrics: deque = deque(maxlen=10000)  # Keep error history

        # Trading operation metrics
        self._order_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._execution_metrics: deque = deque(maxlen=5000)
        self._strategy_metrics: dict[str, dict[str, Any]] = defaultdict(dict)

        # Market data metrics
        self._market_data_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._latency_metrics: deque = deque(maxlen=1000)

        # Infrastructure metrics
        self._connection_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
        self._database_metrics: deque = deque(maxlen=1000)
        self._api_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Alert thresholds
        self._operational_thresholds = {
            "max_cpu_usage": 80.0,
            "max_memory_usage": 85.0,
            "max_disk_usage": 90.0,
            "min_order_fill_rate": 95.0,
            "max_avg_execution_time": 5000.0,  # milliseconds
            "max_error_rate": 1.0,  # percent
            "max_latency_p95": 100.0,  # milliseconds
            "min_uptime_percent": 99.5,
            "max_failed_connections": 5,
        }

        # Monitoring state
        self._start_time = get_current_utc_timestamp()
        self._monitoring_tasks: set = set()
        self._running = False

        self.logger.info("OperationalAnalyticsEngine initialized")

    async def start(self) -> None:
        """Start operational monitoring tasks."""
        if self._running:
            self.logger.warning("Operational analytics already running")
            return

        self._running = True

        # Start monitoring tasks
        tasks = [
            self._system_monitoring_loop(),
            self._trading_monitoring_loop(),
            self._infrastructure_monitoring_loop(),
            self._health_check_loop(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.add(task)
            task.add_done_callback(self._monitoring_tasks.discard)

        self.logger.info("Operational analytics started")

    async def stop(self) -> None:
        """Stop operational monitoring tasks."""
        self._running = False

        # Cancel all tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        self.logger.info("Operational analytics stopped")

    def record_order_event(
        self,
        event_type: str,
        exchange: str,
        order_id: str,
        timestamp: datetime | None = None,
        execution_time_ms: float | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record order-related event for operational tracking.

        Args:
            event_type: Type of order event (placed, filled, cancelled, failed)
            exchange: Exchange name
            order_id: Order identifier
            timestamp: Event timestamp
            execution_time_ms: Execution time in milliseconds
            success: Whether the operation was successful
            metadata: Additional metadata
        """
        timestamp = timestamp or get_current_utc_timestamp()

        event_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "exchange": exchange,
            "order_id": order_id,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "metadata": metadata or {},
        }

        self._order_metrics[exchange].append(event_data)

        if execution_time_ms is not None:
            self._execution_metrics.append(
                {
                    "timestamp": timestamp,
                    "exchange": exchange,
                    "execution_time_ms": execution_time_ms,
                    "event_type": event_type,
                }
            )

        # Update metrics
        self.metrics_collector.increment_counter(
            "operational_order_events",
            labels={"exchange": exchange, "event_type": event_type, "success": str(success)},
        )

        if execution_time_ms is not None:
            self.metrics_collector.observe_histogram(
                "operational_order_execution_time",
                execution_time_ms,
                labels={"exchange": exchange, "event_type": event_type},
            )

    def record_strategy_event(
        self,
        strategy_name: str,
        event_type: str,
        timestamp: datetime | None = None,
        success: bool = True,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record strategy-related event.

        Args:
            strategy_name: Name of strategy
            event_type: Type of event (start, stop, signal, trade, error)
            timestamp: Event timestamp
            success: Whether the operation was successful
            error_message: Error message if not successful
            metadata: Additional metadata
        """
        timestamp = timestamp or get_current_utc_timestamp()

        if strategy_name not in self._strategy_metrics:
            self._strategy_metrics[strategy_name] = {
                "start_time": timestamp,
                "events": deque(maxlen=1000),
                "error_count": 0,
                "success_count": 0,
                "last_activity": timestamp,
            }

        strategy_data = self._strategy_metrics[strategy_name]

        event_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {},
        }

        strategy_data["events"].append(event_data)
        strategy_data["last_activity"] = timestamp

        if success:
            strategy_data["success_count"] += 1
        else:
            strategy_data["error_count"] += 1

        # Update metrics
        self.metrics_collector.increment_counter(
            "operational_strategy_events",
            labels={"strategy": strategy_name, "event_type": event_type, "success": str(success)},
        )

    def record_market_data_event(
        self,
        exchange: str,
        symbol: str,
        event_type: str,
        latency_ms: float | None = None,
        success: bool = True,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record market data event.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            event_type: Type of event (price_update, orderbook_update, trade)
            latency_ms: Data latency in milliseconds
            success: Whether data was successfully processed
            timestamp: Event timestamp
            metadata: Additional metadata
        """
        timestamp = timestamp or get_current_utc_timestamp()

        event_data = {
            "timestamp": timestamp,
            "exchange": exchange,
            "symbol": symbol,
            "event_type": event_type,
            "latency_ms": latency_ms,
            "success": success,
            "metadata": metadata or {},
        }

        key = f"{exchange}:{symbol}"
        self._market_data_metrics[key].append(event_data)

        if latency_ms is not None:
            self._latency_metrics.append(
                {
                    "timestamp": timestamp,
                    "exchange": exchange,
                    "symbol": symbol,
                    "latency_ms": latency_ms,
                }
            )

        # Update metrics
        self.metrics_collector.increment_counter(
            "operational_market_data_events",
            labels={"exchange": exchange, "event_type": event_type, "success": str(success)},
        )

        if latency_ms is not None:
            self.metrics_collector.observe_histogram(
                "operational_market_data_latency",
                latency_ms,
                labels={"exchange": exchange, "event_type": event_type},
            )

    def record_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        severity: AlertSeverity = AlertSeverity.LOW,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record system error for tracking.

        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            severity: Error severity
            timestamp: Error timestamp
            metadata: Additional metadata
        """
        timestamp = timestamp or get_current_utc_timestamp()

        error_data = {
            "timestamp": timestamp,
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "severity": severity,
            "metadata": metadata or {},
        }

        self._error_metrics.append(error_data)

        # Update error metrics
        self.metrics_collector.increment_counter(
            "operational_errors",
            labels={"component": component, "error_type": error_type, "severity": severity.value},
        )

    def record_api_call(
        self,
        service: str,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        success: bool = True,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record API call metrics.

        Args:
            service: Service name (exchange, database, etc.)
            endpoint: API endpoint
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            success: Whether call was successful
            timestamp: Call timestamp
        """
        timestamp = timestamp or get_current_utc_timestamp()

        api_data = {
            "timestamp": timestamp,
            "service": service,
            "endpoint": endpoint,
            "response_time_ms": response_time_ms,
            "status_code": status_code,
            "success": success,
        }

        self._api_metrics[service].append(api_data)

        # Update API metrics
        self.metrics_collector.increment_counter(
            "operational_api_calls",
            labels={"service": service, "endpoint": endpoint, "success": str(success)},
        )

        self.metrics_collector.observe_histogram(
            "operational_api_response_time",
            response_time_ms,
            labels={"service": service, "endpoint": endpoint},
        )

    async def calculate_operational_metrics(self) -> OperationalMetrics:
        """
        Calculate comprehensive operational metrics.

        Returns:
            Current operational metrics
        """
        try:
            now = get_current_utc_timestamp()

            # System uptime
            uptime_seconds = (now - self._start_time).total_seconds()
            uptime_hours = safe_decimal(uptime_seconds / 3600)

            # Strategy metrics
            active_strategies = len(
                [
                    s
                    for s, data in self._strategy_metrics.items()
                    if (now - data["last_activity"]).total_seconds()
                    < 300  # Active in last 5 minutes
                ]
            )
            total_strategies = len(self._strategy_metrics)

            # Order metrics (last 24 hours)
            cutoff_time = now - timedelta(hours=24)
            orders_placed = 0
            orders_filled = 0
            total_orders = 0
            execution_times = []

            for exchange_orders in self._order_metrics.values():
                for order in exchange_orders:
                    if order["timestamp"] > cutoff_time:
                        total_orders += 1
                        if order["event_type"] == "placed":
                            orders_placed += 1
                        elif order["event_type"] == "filled":
                            orders_filled += 1

                        if order.get("execution_time_ms"):
                            execution_times.append(order["execution_time_ms"])

            fill_rate = safe_decimal(
                (orders_filled / orders_placed * 100) if orders_placed > 0 else 0
            )
            avg_execution_time = safe_decimal(np.mean(execution_times) if execution_times else 0)

            # Error metrics (last 24 hours)
            recent_errors = [e for e in self._error_metrics if e["timestamp"] > cutoff_time]

            total_events = total_orders + len(recent_errors)  # Simplified event count
            error_rate = safe_decimal(
                (len(recent_errors) / total_events * 100) if total_events > 0 else 0
            )
            critical_errors = len(
                [e for e in recent_errors if e["severity"] == AlertSeverity.CRITICAL]
            )

            # Market data metrics
            recent_latencies = [
                l["latency_ms"]
                for l in self._latency_metrics
                if l["timestamp"] > cutoff_time and l.get("latency_ms") is not None
            ]

            latency_p50 = safe_decimal(
                np.percentile(recent_latencies, 50) if recent_latencies else 0
            )
            latency_p95 = safe_decimal(
                np.percentile(recent_latencies, 95) if recent_latencies else 0
            )

            # System resource metrics
            cpu_usage = safe_decimal(psutil.cpu_percent())
            memory = psutil.virtual_memory()
            memory_usage = safe_decimal(memory.percent)
            disk = psutil.disk_usage("/")
            disk_usage = safe_decimal((disk.used / disk.total) * 100)

            # API success rate
            api_success_rate = await self._calculate_api_success_rate(cutoff_time)

            # WebSocket uptime (simplified)
            websocket_uptime = safe_decimal(
                99.5
            )  # Placeholder - would track actual WebSocket connections

            # Database metrics
            db_connections = await self._get_database_connections()
            db_avg_query_time = await self._get_average_query_time(cutoff_time)

            # Cache metrics
            cache_hit_rate = await self._get_cache_hit_rate()

            return OperationalMetrics(
                timestamp=now,
                system_uptime=uptime_hours,
                strategies_active=active_strategies,
                strategies_total=total_strategies,
                exchanges_connected=await self._count_connected_exchanges(),
                exchanges_total=await self._count_total_exchanges(),
                orders_placed_today=orders_placed,
                orders_filled_today=orders_filled,
                order_fill_rate=fill_rate,
                avg_order_execution_time=avg_execution_time,
                avg_order_slippage=safe_decimal(0),  # Would calculate from actual slippage data
                api_call_success_rate=api_success_rate,
                websocket_uptime_percent=websocket_uptime,
                data_latency_p50=latency_p50,
                data_latency_p95=latency_p95,
                error_rate=error_rate,
                critical_errors_today=critical_errors,
                memory_usage_percent=memory_usage,
                cpu_usage_percent=cpu_usage,
                disk_usage_percent=disk_usage,
                database_connections_active=db_connections,
                database_query_avg_time=db_avg_query_time,
                cache_hit_rate=cache_hit_rate,
                backup_status="completed",  # Would get from backup system
                compliance_checks_passed=0,  # Would integrate with compliance system
                compliance_checks_failed=0,
                risk_limit_breaches=0,  # Would integrate with risk system
                circuit_breaker_triggers=0,
                performance_degradation_events=await self._count_performance_issues(cutoff_time),
                data_quality_issues=await self._count_data_quality_issues(cutoff_time),
                exchange_outages=await self._count_exchange_outages(cutoff_time),
            )

        except Exception as e:
            self.logger.error(f"Error calculating operational metrics: {e}")
            return OperationalMetrics(timestamp=get_current_utc_timestamp())

    async def generate_health_report(self) -> dict[str, Any]:
        """
        Generate comprehensive system health report.

        Returns:
            System health report with status and recommendations
        """
        try:
            metrics = await self.calculate_operational_metrics()

            health_status = {
                "overall_status": "healthy",
                "component_status": {},
                "alerts": [],
                "recommendations": [],
                "performance_summary": {},
                "resource_utilization": {},
                "uptime_summary": {},
            }

            # Component health assessment
            components = {
                "system": {
                    "cpu_usage": float(metrics.cpu_usage_percent),
                    "memory_usage": float(metrics.memory_usage_percent),
                    "disk_usage": float(metrics.disk_usage_percent),
                    "uptime_hours": float(metrics.system_uptime),
                },
                "trading": {
                    "order_fill_rate": float(metrics.order_fill_rate),
                    "avg_execution_time": float(metrics.avg_order_execution_time or 0),
                    "active_strategies": metrics.strategies_active,
                    "total_strategies": metrics.strategies_total,
                },
                "data": {
                    "api_success_rate": float(metrics.api_call_success_rate),
                    "websocket_uptime": float(metrics.websocket_uptime_percent),
                    "latency_p95": float(metrics.data_latency_p95 or 0),
                    "data_quality_issues": metrics.data_quality_issues,
                },
                "infrastructure": {
                    "database_connections": metrics.database_connections_active,
                    "cache_hit_rate": float(metrics.cache_hit_rate),
                    "error_rate": float(metrics.error_rate),
                    "critical_errors": metrics.critical_errors_today,
                },
            }

            # Assess each component
            overall_health_score = 100

            for component, metrics_dict in components.items():
                component_score = await self._assess_component_health(component, metrics_dict)
                health_status["component_status"][component] = {
                    "status": (
                        "healthy"
                        if component_score >= 80
                        else "warning" if component_score >= 60 else "critical"
                    ),
                    "score": component_score,
                    "metrics": metrics_dict,
                }

                # Impact overall score
                if component_score < overall_health_score:
                    overall_health_score = component_score

            # Set overall status
            if overall_health_score >= 80:
                health_status["overall_status"] = "healthy"
            elif overall_health_score >= 60:
                health_status["overall_status"] = "warning"
            else:
                health_status["overall_status"] = "critical"

            # Generate alerts and recommendations
            health_status["alerts"] = await self._generate_health_alerts(components)
            health_status["recommendations"] = await self._generate_health_recommendations(
                components
            )

            # Performance summary
            health_status["performance_summary"] = {
                "system_performance": overall_health_score,
                "trading_efficiency": float(metrics.order_fill_rate),
                "data_quality": 100 - float(metrics.error_rate),
                "resource_efficiency": 100
                - max(float(metrics.cpu_usage_percent), float(metrics.memory_usage_percent)),
            }

            # Resource utilization
            health_status["resource_utilization"] = {
                "cpu": {
                    "current": float(metrics.cpu_usage_percent),
                    "threshold": self._operational_thresholds["max_cpu_usage"],
                },
                "memory": {
                    "current": float(metrics.memory_usage_percent),
                    "threshold": self._operational_thresholds["max_memory_usage"],
                },
                "disk": {
                    "current": float(metrics.disk_usage_percent),
                    "threshold": self._operational_thresholds["max_disk_usage"],
                },
            }

            # Uptime summary
            health_status["uptime_summary"] = {
                "system_uptime_hours": float(metrics.system_uptime),
                "websocket_uptime_percent": float(metrics.websocket_uptime_percent),
                "api_availability_percent": float(metrics.api_call_success_rate),
            }

            return health_status

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {"error": str(e)}

    async def _system_monitoring_loop(self) -> None:
        """Background loop for system monitoring."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _trading_monitoring_loop(self) -> None:
        """Background loop for trading operation monitoring."""
        while self._running:
            try:
                await self._analyze_trading_performance()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trading monitoring loop: {e}")
                await asyncio.sleep(300)

    async def _infrastructure_monitoring_loop(self) -> None:
        """Background loop for infrastructure monitoring."""
        while self._running:
            try:
                await self._monitor_infrastructure_health()
                await asyncio.sleep(120)  # Monitor every 2 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in infrastructure monitoring loop: {e}")
                await asyncio.sleep(120)

    async def _health_check_loop(self) -> None:
        """Background loop for health checks and alerting."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(300)

    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            timestamp = get_current_utc_timestamp()

            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage("/")

            # Network metrics (simplified)
            network = psutil.net_io_counters()

            system_data = {
                "timestamp": timestamp,
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
            }

            self._system_metrics.append(system_data)

            # Update Prometheus metrics
            self.metrics_collector.set_gauge("operational_cpu_usage", cpu_usage)
            self.metrics_collector.set_gauge("operational_memory_usage", memory.percent)
            self.metrics_collector.set_gauge(
                "operational_disk_usage", (disk.used / disk.total) * 100
            )

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _analyze_trading_performance(self) -> None:
        """Analyze trading operation performance."""
        try:
            cutoff_time = get_current_utc_timestamp() - timedelta(hours=1)

            # Analyze order execution performance
            execution_times = []
            fill_rates_by_exchange = defaultdict(list)

            for exchange, orders in self._order_metrics.items():
                placed_orders = 0
                filled_orders = 0

                for order in orders:
                    if order["timestamp"] > cutoff_time:
                        if order["event_type"] == "placed":
                            placed_orders += 1
                        elif order["event_type"] == "filled":
                            filled_orders += 1

                        if order.get("execution_time_ms"):
                            execution_times.append(order["execution_time_ms"])

                if placed_orders > 0:
                    fill_rate = (filled_orders / placed_orders) * 100
                    fill_rates_by_exchange[exchange].append(fill_rate)

                    # Alert on low fill rates
                    if fill_rate < self._operational_thresholds["min_order_fill_rate"]:
                        self.logger.warning(f"Low fill rate on {exchange}: {fill_rate:.1f}%")

            # Alert on high execution times
            if execution_times:
                avg_execution_time = np.mean(execution_times)
                if avg_execution_time > self._operational_thresholds["max_avg_execution_time"]:
                    self.logger.warning(f"High average execution time: {avg_execution_time:.1f}ms")

        except Exception as e:
            self.logger.error(f"Error analyzing trading performance: {e}")

    async def _monitor_infrastructure_health(self) -> None:
        """Monitor infrastructure component health."""
        try:
            # Database health check
            db_health = await self._check_database_health()

            # API endpoint health checks
            api_health = await self._check_api_health()

            # Cache health check
            cache_health = await self._check_cache_health()

            # Update health metrics
            self.metrics_collector.set_gauge("operational_database_health", 1 if db_health else 0)
            self.metrics_collector.set_gauge("operational_api_health", 1 if api_health else 0)
            self.metrics_collector.set_gauge("operational_cache_health", 1 if cache_health else 0)

        except Exception as e:
            self.logger.error(f"Error monitoring infrastructure health: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks."""
        try:
            metrics = await self.calculate_operational_metrics()

            # Check system resource thresholds
            if float(metrics.cpu_usage_percent) > self._operational_thresholds["max_cpu_usage"]:
                self.logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")

            if (
                float(metrics.memory_usage_percent)
                > self._operational_thresholds["max_memory_usage"]
            ):
                self.logger.warning(f"High memory usage: {metrics.memory_usage_percent:.1f}%")

            if float(metrics.disk_usage_percent) > self._operational_thresholds["max_disk_usage"]:
                self.logger.warning(f"High disk usage: {metrics.disk_usage_percent:.1f}%")

            # Check trading operation health
            if float(metrics.order_fill_rate) < self._operational_thresholds["min_order_fill_rate"]:
                self.logger.warning(f"Low order fill rate: {metrics.order_fill_rate:.1f}%")

            if float(metrics.error_rate) > self._operational_thresholds["max_error_rate"]:
                self.logger.warning(f"High error rate: {metrics.error_rate:.1f}%")

        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")

    async def _assess_component_health(self, component: str, metrics_dict: dict[str, Any]) -> float:
        """Assess health score for a component (0-100)."""
        try:
            score = 100.0

            if component == "system":
                # Penalize high resource usage
                score -= max(0, metrics_dict.get("cpu_usage", 0) - 50) * 2
                score -= max(0, metrics_dict.get("memory_usage", 0) - 50) * 2
                score -= max(0, metrics_dict.get("disk_usage", 0) - 70) * 3

            elif component == "trading":
                # Penalize low fill rates and high execution times
                fill_rate = metrics_dict.get("order_fill_rate", 100)
                score -= max(0, 95 - fill_rate) * 2

                exec_time = metrics_dict.get("avg_execution_time", 0)
                if exec_time > 1000:  # >1 second
                    score -= min(50, (exec_time - 1000) / 100)

            elif component == "data":
                # Penalize low API success rates and high latencies
                api_success = metrics_dict.get("api_success_rate", 100)
                score -= max(0, 95 - api_success) * 2

                latency = metrics_dict.get("latency_p95", 0)
                if latency > 100:  # >100ms
                    score -= min(30, (latency - 100) / 10)

            elif component == "infrastructure":
                # Penalize high error rates and low cache hit rates
                error_rate = metrics_dict.get("error_rate", 0)
                score -= error_rate * 10

                cache_hit_rate = metrics_dict.get("cache_hit_rate", 100)
                score -= max(0, 90 - cache_hit_rate)

            return max(0, min(100, score))

        except Exception as e:
            self.logger.error(f"Error assessing component health: {e}")
            return 50.0

    async def _generate_health_alerts(self, components: dict[str, dict[str, Any]]) -> list[str]:
        """Generate health alerts based on component status."""
        alerts = []

        try:
            system_metrics = components.get("system", {})
            if system_metrics.get("cpu_usage", 0) > 80:
                alerts.append("HIGH: CPU usage exceeds 80%")

            if system_metrics.get("memory_usage", 0) > 85:
                alerts.append("HIGH: Memory usage exceeds 85%")

            if system_metrics.get("disk_usage", 0) > 90:
                alerts.append("CRITICAL: Disk usage exceeds 90%")

            trading_metrics = components.get("trading", {})
            if trading_metrics.get("order_fill_rate", 100) < 90:
                alerts.append("MEDIUM: Order fill rate below 90%")

            data_metrics = components.get("data", {})
            if data_metrics.get("api_success_rate", 100) < 95:
                alerts.append("MEDIUM: API success rate below 95%")

            if data_metrics.get("latency_p95", 0) > 200:
                alerts.append("MEDIUM: High data latency (>200ms)")

            infra_metrics = components.get("infrastructure", {})
            if infra_metrics.get("error_rate", 0) > 2:
                alerts.append("HIGH: Error rate exceeds 2%")

            if infra_metrics.get("critical_errors", 0) > 0:
                alerts.append(f"CRITICAL: {infra_metrics['critical_errors']} critical errors")

        except Exception as e:
            self.logger.error(f"Error generating health alerts: {e}")

        return alerts

    async def _generate_health_recommendations(
        self, components: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Generate health improvement recommendations."""
        recommendations = []

        try:
            system_metrics = components.get("system", {})
            if system_metrics.get("cpu_usage", 0) > 70:
                recommendations.append(
                    "Consider scaling CPU resources or optimizing high-CPU processes"
                )

            if system_metrics.get("memory_usage", 0) > 80:
                recommendations.append(
                    "Review memory usage patterns and consider increasing available memory"
                )

            trading_metrics = components.get("trading", {})
            if trading_metrics.get("order_fill_rate", 100) < 95:
                recommendations.append(
                    "Review order routing and execution algorithms for better fill rates"
                )

            if trading_metrics.get("avg_execution_time", 0) > 2000:
                recommendations.append("Optimize order execution pipeline to reduce latency")

            data_metrics = components.get("data", {})
            if data_metrics.get("latency_p95", 0) > 100:
                recommendations.append("Optimize market data processing and network connectivity")

            infra_metrics = components.get("infrastructure", {})
            if infra_metrics.get("cache_hit_rate", 100) < 85:
                recommendations.append("Review cache configuration and warming strategies")

            if not recommendations:
                recommendations.append("System is operating within normal parameters")

        except Exception as e:
            self.logger.error(f"Error generating health recommendations: {e}")
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return recommendations

    async def _calculate_api_success_rate(self, cutoff_time: datetime) -> Decimal:
        """Calculate API success rate for recent period."""
        try:
            total_calls = 0
            successful_calls = 0

            for service_calls in self._api_metrics.values():
                for call in service_calls:
                    if call["timestamp"] > cutoff_time:
                        total_calls += 1
                        if call["success"]:
                            successful_calls += 1

            return safe_decimal((successful_calls / total_calls * 100) if total_calls > 0 else 100)

        except Exception as e:
            self.logger.error(f"Error calculating API success rate: {e}")
            return safe_decimal(100)

    async def _get_database_connections(self) -> int:
        """Get active database connections count."""
        # Placeholder - would integrate with actual database monitoring
        return 10

    async def _get_average_query_time(self, cutoff_time: datetime) -> Decimal | None:
        """Get average database query time."""
        # Placeholder - would calculate from actual database metrics
        return safe_decimal(15.5)  # 15.5ms

    async def _get_cache_hit_rate(self) -> Decimal:
        """Get cache hit rate percentage."""
        # Placeholder - would integrate with actual cache monitoring
        return safe_decimal(92.5)

    async def _count_connected_exchanges(self) -> int:
        """Count currently connected exchanges."""
        # Placeholder - would check actual exchange connections
        return 3

    async def _count_total_exchanges(self) -> int:
        """Count total configured exchanges."""
        # Placeholder - would get from configuration
        return 3

    async def _count_performance_issues(self, cutoff_time: datetime) -> int:
        """Count performance degradation events in period."""
        # Placeholder - would analyze performance metrics
        return 0

    async def _count_data_quality_issues(self, cutoff_time: datetime) -> int:
        """Count data quality issues in period."""
        # Placeholder - would analyze data quality metrics
        return 0

    async def _count_exchange_outages(self, cutoff_time: datetime) -> int:
        """Count exchange outages in period."""
        # Placeholder - would track exchange connectivity
        return 0

    async def _check_database_health(self) -> bool:
        """Check database health status."""
        # Placeholder - would perform actual database health check
        return True

    async def _check_api_health(self) -> bool:
        """Check API endpoints health."""
        # Placeholder - would check API endpoint responses
        return True

    async def _check_cache_health(self) -> bool:
        """Check cache system health."""
        # Placeholder - would check cache system status
        return True
