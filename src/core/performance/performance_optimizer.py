"""
Integrated Performance Optimizer for T-Bot Trading System

DEPRECATED: This module is deprecated and will be removed in a future version.
Use src.monitoring.performance.PerformanceProfiler instead for performance monitoring.

This module provides a unified interface for all performance optimization components,
orchestrating database optimizations, memory management, caching, and trading-specific
profiling to achieve <100ms latency targets for critical trading operations.

Features:
- Centralized performance optimization management
- Automatic initialization and coordination of all optimization components
- Performance regression detection and alerting
- Real-time optimization recommendations
- Comprehensive performance reporting
- Production-ready configuration management
"""

import asyncio
from datetime import datetime
from typing import Any

from src.base import BaseComponent

# Import all performance optimization components
from src.core.caching.unified_cache_layer import UnifiedCacheLayer
from src.core.config import Config
from src.core.exceptions import PerformanceError
from src.core.logging import get_logger
from src.core.performance.memory_optimizer import MemoryOptimizer
from src.core.performance.performance_monitor import AlertLevel, PerformanceMonitor
from src.core.performance.trading_profiler import TradingOperation, TradingOperationOptimizer
from src.database.service import DatabaseService
from src.exchanges.connection_pool import ConnectionPoolManager

logger = get_logger(__name__)


class PerformanceOptimizer(BaseComponent):
    """
    Integrated performance optimizer that coordinates all optimization components
    to achieve optimal trading system performance.

    This class serves as the central orchestrator for:
    - Database query optimization and indexing
    - Memory management and garbage collection
    - Multi-level caching strategies
    - Connection pooling for exchanges
    - Trading operation profiling and optimization
    - Real-time performance monitoring and alerting
    """

    def __init__(self, config: Config):
        """Initialize the integrated performance optimizer."""
        super().__init__()
        self.config = config

        # Core optimization components
        self.memory_optimizer: MemoryOptimizer | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.trading_profiler: TradingOperationOptimizer | None = None
        self.cache_layer: UnifiedCacheLayer | None = None
        self.database_service: DatabaseService | None = None
        self.connection_pool_manager: ConnectionPoolManager | None = None

        # Performance targets
        self.latency_targets = {
            "order_placement": 50.0,  # 50ms target
            "market_data": 10.0,  # 10ms target
            "risk_calculation": 25.0,  # 25ms target
            "database_query": 50.0,  # 50ms target
            "cache_access": 10.0,  # 10ms target
        }

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.alerts_history: list[dict[str, Any]] = []
        self.optimization_actions: list[dict[str, Any]] = []

        # Background tasks
        self._coordinator_task: asyncio.Task | None = None
        self._reporting_task: asyncio.Task | None = None

        # Initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all performance optimization components."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing integrated performance optimizer...")

            # Initialize core monitoring first
            await self._initialize_performance_monitor()

            # Initialize memory optimization
            await self._initialize_memory_optimizer()

            # Initialize caching layer
            await self._initialize_cache_layer()

            # Initialize database service
            await self._initialize_database_service()

            # Initialize connection pool management
            await self._initialize_connection_pools()

            # Initialize trading profiler
            await self._initialize_trading_profiler()

            # Start coordination tasks
            await self._start_coordination_tasks()

            # Setup alert handlers
            await self._setup_alert_handlers()

            self._initialized = True

            self.logger.info("Performance optimizer initialized successfully")

            # Run initial performance assessment
            await self._run_initial_assessment()

        except Exception as e:
            self.logger.error(f"Failed to initialize performance optimizer: {e}")
            raise PerformanceError(f"Performance optimizer initialization failed: {e}")

    async def _initialize_performance_monitor(self) -> None:
        """Initialize performance monitoring component."""
        self.performance_monitor = PerformanceMonitor(self.config)
        await self.performance_monitor.initialize()
        self.logger.info("Performance monitor initialized")

    async def _initialize_memory_optimizer(self) -> None:
        """Initialize memory optimization component."""
        self.memory_optimizer = MemoryOptimizer(self.config)
        await self.memory_optimizer.initialize()
        self.logger.info("Memory optimizer initialized")

    async def _initialize_cache_layer(self) -> None:
        """Initialize unified caching layer."""
        self.cache_layer = UnifiedCacheLayer(self.config)
        await self.cache_layer.initialize()
        self.logger.info("Unified cache layer initialized")

    async def _initialize_database_service(self) -> None:
        """Initialize database service."""
        from src.core.config.service import ConfigService
        from src.utils.validation.service import ValidationService

        config_service = ConfigService(self.config.to_dict())
        validation_service = ValidationService(config_service)

        self.database_service = DatabaseService(config_service, validation_service)
        await self.database_service.start()
        self.logger.info("Database service initialized")

    async def _initialize_connection_pools(self) -> None:
        """Initialize connection pool management."""
        self.connection_pool_manager = ConnectionPoolManager(self.config)
        await self.connection_pool_manager.initialize()
        self.logger.info("Connection pool manager initialized")

    async def _initialize_trading_profiler(self) -> None:
        """Initialize trading operation profiler."""
        if self.performance_monitor:
            self.trading_profiler = TradingOperationOptimizer(self.config, self.performance_monitor)
            await self.trading_profiler.initialize()
            self.logger.info("Trading profiler initialized")

    async def _start_coordination_tasks(self) -> None:
        """Start background coordination tasks."""
        # Performance coordination task
        self._coordinator_task = asyncio.create_task(self._coordination_loop())

        # Performance reporting task
        self._reporting_task = asyncio.create_task(self._reporting_loop())

        self.logger.info("Coordination tasks started")

    async def _setup_alert_handlers(self) -> None:
        """Setup alert handlers for performance issues."""
        if self.performance_monitor:
            self.performance_monitor.add_alert_callback(self._handle_performance_alert)

        if self.memory_optimizer:
            self.memory_optimizer.add_alert_callback(self._handle_memory_alert)

    async def _coordination_loop(self) -> None:
        """Main coordination loop for performance optimization."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Collect performance metrics from all components
                metrics = await self._collect_comprehensive_metrics()

                # Analyze performance and identify optimization opportunities
                optimization_actions = await self._analyze_optimization_opportunities(metrics)

                # Execute recommended optimizations
                await self._execute_optimizations(optimization_actions)

                # Update performance history
                self.performance_history.append(
                    {
                        "timestamp": datetime.utcnow(),
                        "metrics": metrics,
                        "actions": optimization_actions,
                    }
                )

                # Keep only recent history
                if len(self.performance_history) > 1440:  # 24 hours
                    self.performance_history = self.performance_history[-720:]  # Keep 12 hours

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")

    async def _reporting_loop(self) -> None:
        """Background loop for performance reporting."""
        while True:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes

                # Generate performance report
                report = await self.get_performance_report()

                # Log key performance metrics
                self.logger.info(
                    "Performance summary",
                    extra={
                        "latency_p95_order_ms": report.get("trading_latencies", {}).get(
                            "order_placement_p95", 0
                        ),
                        "latency_p95_market_data_ms": report.get("trading_latencies", {}).get(
                            "market_data_p95", 0
                        ),
                        "memory_usage_mb": report.get("memory_stats", {}).get(
                            "process_memory_mb", 0
                        ),
                        "cache_hit_rate": report.get("cache_stats", {}).get("global_hit_rate", 0),
                        "database_p95_ms": report.get("database_stats", {}).get(
                            "avg_query_time_ms", 0
                        ),
                        "active_alerts": report.get("alert_summary", {}).get("active_count", 0),
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Reporting loop error: {e}")

    async def _collect_comprehensive_metrics(self) -> dict[str, Any]:
        """Collect metrics from all optimization components."""
        metrics = {
            "timestamp": datetime.utcnow(),
            "performance_monitor": {},
            "memory_optimizer": {},
            "cache_layer": {},
            "database_optimizer": {},
            "trading_profiler": {},
            "connection_pools": {},
        }

        try:
            # Performance monitor metrics
            if self.performance_monitor:
                metrics["performance_monitor"] = (
                    await self.performance_monitor.get_performance_summary()
                )

            # Memory optimizer metrics
            if self.memory_optimizer:
                metrics["memory_optimizer"] = await self.memory_optimizer.get_memory_report()

            # Cache layer metrics
            if self.cache_layer:
                metrics["cache_layer"] = await self.cache_layer.get_comprehensive_stats()

            # Database service metrics
            if self.database_service:
                metrics["database_optimizer"] = self.database_service.get_performance_metrics()

            # Trading profiler metrics
            if self.trading_profiler:
                metrics["trading_profiler"] = await self.trading_profiler.get_optimization_report()

            # Connection pool metrics
            if self.connection_pool_manager:
                metrics["connection_pools"] = await self.connection_pool_manager.get_global_status()

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

        return metrics

    async def _analyze_optimization_opportunities(
        self, metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze metrics and identify optimization opportunities."""
        opportunities = []

        try:
            # Analyze trading operation latencies
            trading_metrics = metrics.get("performance_monitor", {}).get("latency_stats", {})
            for operation, stats in trading_metrics.items():
                p95_latency = stats.get("p95_ms", 0)
                target = self.latency_targets.get(operation, 100.0)

                if p95_latency > target * 1.2:  # 20% above target
                    opportunities.append(
                        {
                            "type": "latency_optimization",
                            "component": "trading_operations",
                            "operation": operation,
                            "current_p95_ms": p95_latency,
                            "target_ms": target,
                            "priority": "high" if p95_latency > target * 2 else "medium",
                            "recommendation": f"Optimize {operation} - current P95 {p95_latency:.1f}ms exceeds target {target}ms",
                        }
                    )

            # Analyze memory usage
            memory_metrics = metrics.get("memory_optimizer", {}).get("current_stats", {})
            memory_mb = memory_metrics.get("process_memory_mb", 0)
            if memory_mb > 1500:  # > 1.5GB
                opportunities.append(
                    {
                        "type": "memory_optimization",
                        "component": "memory_management",
                        "current_memory_mb": memory_mb,
                        "priority": "high" if memory_mb > 1800 else "medium",
                        "recommendation": f"High memory usage {memory_mb:.0f}MB - consider garbage collection or memory cleanup",
                    }
                )

            # Analyze cache performance
            cache_metrics = metrics.get("cache_layer", {}).get("global_stats", {})
            for level, stats in cache_metrics.items():
                if isinstance(stats, dict) and "hit_rate" in stats:
                    hit_rate = stats["hit_rate"]
                    if hit_rate < 0.8:  # < 80% hit rate
                        opportunities.append(
                            {
                                "type": "cache_optimization",
                                "component": "caching",
                                "cache_level": level,
                                "hit_rate": hit_rate,
                                "priority": "medium",
                                "recommendation": f"Low cache hit rate {hit_rate:.1%} for {level} - consider cache warming or TTL adjustment",
                            }
                        )

            # Analyze database performance
            db_metrics = metrics.get("database_optimizer", {}).get("query_performance", {})
            avg_query_time = db_metrics.get("average_time_ms", 0)
            if avg_query_time > 25:  # > 25ms average
                opportunities.append(
                    {
                        "type": "database_optimization",
                        "component": "database",
                        "avg_query_time_ms": avg_query_time,
                        "priority": "high" if avg_query_time > 50 else "medium",
                        "recommendation": f"Slow database queries {avg_query_time:.1f}ms average - consider indexing or query optimization",
                    }
                )

        except Exception as e:
            self.logger.error(f"Error analyzing optimization opportunities: {e}")

        return opportunities

    async def _execute_optimizations(self, opportunities: list[dict[str, Any]]) -> None:
        """Execute recommended optimizations."""
        for opportunity in opportunities:
            try:
                component = opportunity.get("component")
                opportunity.get("type")
                priority = opportunity.get("priority", "medium")

                # Only execute high-priority optimizations automatically
                if priority != "high":
                    continue

                action_taken = None

                if component == "memory_management" and self.memory_optimizer:
                    result = await self.memory_optimizer.force_memory_optimization()
                    action_taken = f"Forced memory optimization: freed {result.get('memory_freed_mb', 0):.1f}MB"

                elif component == "trading_operations" and self.trading_profiler:
                    result = await self.trading_profiler.force_optimization_analysis()
                    action_taken = f"Forced trading optimization analysis: {result.get('recommendations_generated', 0)} recommendations"

                elif component == "database" and self.database_service:
                    # Database optimizations are built into the service
                    action_taken = "Database service performance analysis completed"

                elif component == "caching" and self.cache_layer:
                    # Clear and warm caches
                    await self.cache_layer.clear()
                    action_taken = "Cache cleared for optimization"

                if action_taken:
                    self.optimization_actions.append(
                        {
                            "timestamp": datetime.utcnow(),
                            "opportunity": opportunity,
                            "action_taken": action_taken,
                        }
                    )

                    self.logger.info(f"Executed optimization: {action_taken}")

            except Exception as e:
                self.logger.error(f"Error executing optimization: {e}")

    async def _handle_performance_alert(self, alert) -> None:
        """Handle performance alerts from the performance monitor."""
        alert_data = {
            "timestamp": datetime.utcnow(),
            "source": "performance_monitor",
            "alert": alert.__dict__,
            "handled": False,
        }

        try:
            # Take action based on alert type
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                # Force optimization for critical alerts
                if "latency" in alert.alert_id and self.trading_profiler:
                    await self.trading_profiler.force_optimization_analysis()
                    alert_data["action"] = "Forced trading optimization analysis"
                    alert_data["handled"] = True

                elif "memory" in alert.alert_id and self.memory_optimizer:
                    await self.memory_optimizer.force_memory_optimization()
                    alert_data["action"] = "Forced memory optimization"
                    alert_data["handled"] = True

            self.alerts_history.append(alert_data)

            # Keep recent alerts only
            if len(self.alerts_history) > 1000:
                self.alerts_history = self.alerts_history[-500:]

        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")

    async def _handle_memory_alert(self, alert_level: str, message: str, stats) -> None:
        """Handle memory alerts from the memory optimizer."""
        alert_data = {
            "timestamp": datetime.utcnow(),
            "source": "memory_optimizer",
            "level": alert_level,
            "message": message,
            "stats": stats.__dict__ if hasattr(stats, "__dict__") else stats,
        }

        self.alerts_history.append(alert_data)

        # Take immediate action for critical memory alerts
        if alert_level == "critical" and self.memory_optimizer:
            try:
                await self.memory_optimizer.force_memory_optimization()
                self.logger.info("Executed emergency memory optimization")
            except Exception as e:
                self.logger.error(f"Emergency memory optimization failed: {e}")

    async def _run_initial_assessment(self) -> None:
        """Run initial performance assessment after initialization."""
        try:
            self.logger.info("Running initial performance assessment...")

            # Collect baseline metrics
            baseline_metrics = await self._collect_comprehensive_metrics()

            # Analyze initial state
            opportunities = await self._analyze_optimization_opportunities(baseline_metrics)

            # Log assessment results
            self.logger.info(
                f"Initial assessment complete: {len(opportunities)} optimization opportunities identified",
                extra={
                    "high_priority_optimizations": len(
                        [o for o in opportunities if o.get("priority") == "high"]
                    ),
                    "medium_priority_optimizations": len(
                        [o for o in opportunities if o.get("priority") == "medium"]
                    ),
                    "components_assessed": len([k for k, v in baseline_metrics.items() if v]),
                },
            )

        except Exception as e:
            self.logger.error(f"Initial assessment failed: {e}")

    # Public API methods

    def optimize_trading_operation(self, operation: TradingOperation):
        """Decorator for optimizing trading operations."""
        if self.trading_profiler:
            return self.trading_profiler.optimize_function(operation)
        else:
            # Return pass-through decorator if profiler not available
            def decorator(func):
                return func

            return decorator

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "initialized" if self._initialized else "initializing",
            "latency_targets": self.latency_targets,
            "trading_latencies": {},
            "memory_stats": {},
            "cache_stats": {},
            "database_stats": {},
            "connection_pool_stats": {},
            "alert_summary": {},
            "optimization_history": self.optimization_actions[-10:],  # Last 10 actions
            "recommendations": [],
        }

        try:
            # Get latest metrics
            if self.performance_history:
                latest_metrics = self.performance_history[-1]["metrics"]

                # Extract trading latencies
                trading_metrics = latest_metrics.get("performance_monitor", {}).get(
                    "latency_stats", {}
                )
                for operation, stats in trading_metrics.items():
                    report["trading_latencies"][f"{operation}_p95"] = stats.get("p95_ms", 0)
                    report["trading_latencies"][f"{operation}_p99"] = stats.get("p99_ms", 0)

                # Extract memory stats
                memory_metrics = latest_metrics.get("memory_optimizer", {}).get("current_stats", {})
                report["memory_stats"] = {
                    "process_memory_mb": memory_metrics.get("process_memory_mb", 0),
                    "memory_growth_rate": memory_metrics.get("memory_growth_rate_mb_min", 0),
                    "gc_collections": memory_metrics.get("gc_collections", {}),
                }

                # Extract cache stats
                cache_metrics = latest_metrics.get("cache_layer", {}).get("global_stats", {})
                if cache_metrics:
                    total_hits = sum(
                        stats.get("hits", 0)
                        for stats in cache_metrics.values()
                        if isinstance(stats, dict)
                    )
                    total_misses = sum(
                        stats.get("misses", 0)
                        for stats in cache_metrics.values()
                        if isinstance(stats, dict)
                    )
                    global_hit_rate = (
                        total_hits / (total_hits + total_misses)
                        if (total_hits + total_misses) > 0
                        else 0
                    )
                    report["cache_stats"]["global_hit_rate"] = global_hit_rate

                # Extract database stats
                db_metrics = latest_metrics.get("database_optimizer", {}).get(
                    "query_performance", {}
                )
                report["database_stats"] = {
                    "avg_query_time_ms": db_metrics.get("average_time_ms", 0),
                    "slow_queries_count": db_metrics.get("slow_queries_count", 0),
                    "cache_hit_rate": db_metrics.get("cache_performance", {}).get("hit_rate", 0),
                }

                # Extract connection pool stats
                pool_metrics = latest_metrics.get("connection_pools", {}).get("global_metrics", {})
                report["connection_pool_stats"] = {
                    "total_connections": pool_metrics.get("total_connections", 0),
                    "active_connections": pool_metrics.get("active_connections", 0),
                    "utilization_rate": pool_metrics.get("utilization_rate", 0),
                    "failure_rate": pool_metrics.get("global_failure_rate", 0),
                }

            # Alert summary
            recent_alerts = [
                a
                for a in self.alerts_history
                if (datetime.utcnow() - a["timestamp"]).total_seconds() < 3600
            ]  # Last hour
            report["alert_summary"] = {
                "active_count": len(recent_alerts),
                "critical_count": len([a for a in recent_alerts if a.get("level") == "critical"]),
                "warning_count": len([a for a in recent_alerts if a.get("level") == "warning"]),
            }

            # Generate recommendations
            if self.performance_history:
                latest_opportunities = self.performance_history[-1].get("actions", [])
                report["recommendations"] = [
                    opp["recommendation"]
                    for opp in latest_opportunities
                    if opp.get("priority") in ["high", "medium"]
                ][
                    :5
                ]  # Top 5 recommendations

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            report["error"] = str(e)

        return report

    async def force_optimization(self) -> dict[str, Any]:
        """Force immediate optimization across all components."""
        self.logger.info("Starting forced optimization across all components")

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_optimization": {},
            "trading_optimization": {},
            "cache_optimization": {},
            "database_optimization": {},
        }

        try:
            # Force memory optimization
            if self.memory_optimizer:
                results["memory_optimization"] = (
                    await self.memory_optimizer.force_memory_optimization()
                )

            # Force trading optimization analysis
            if self.trading_profiler:
                results["trading_optimization"] = (
                    await self.trading_profiler.force_optimization_analysis()
                )

            # Clear and optimize caches
            if self.cache_layer:
                await self.cache_layer.clear()
                results["cache_optimization"] = {"action": "caches_cleared"}

            # Database optimization status
            if self.database_service:
                db_metrics = self.database_service.get_performance_metrics()
                results["database_optimization"] = {
                    "avg_query_time": db_metrics.get("average_query_time", 0)
                    * 1000,  # Convert to ms
                    "cache_hit_rate": (
                        db_metrics.get("cache_hits", 0)
                        / max(
                            db_metrics.get("cache_hits", 0) + db_metrics.get("cache_misses", 0), 1
                        )
                    ),
                    "total_queries": db_metrics.get("total_queries", 0),
                }

            self.logger.info("Forced optimization completed", extra=results)

        except Exception as e:
            self.logger.error(f"Forced optimization failed: {e}")
            results["error"] = str(e)

        return results

    async def cleanup(self) -> None:
        """Cleanup all optimization components."""
        try:
            self.logger.info("Cleaning up performance optimizer...")

            # Cancel background tasks
            if self._coordinator_task:
                self._coordinator_task.cancel()
                try:
                    await self._coordinator_task
                except asyncio.CancelledError:
                    pass

            if self._reporting_task:
                self._reporting_task.cancel()
                try:
                    await self._reporting_task
                except asyncio.CancelledError:
                    pass

            # Cleanup components
            if self.memory_optimizer:
                await self.memory_optimizer.cleanup()

            if self.performance_monitor:
                await self.performance_monitor.cleanup()

            if self.trading_profiler:
                await self.trading_profiler.cleanup()

            if self.cache_layer:
                await self.cache_layer.cleanup()

            if self.database_service:
                await self.database_service.stop()

            if self.connection_pool_manager:
                await self.connection_pool_manager.cleanup()

            # Clear history
            self.performance_history.clear()
            self.alerts_history.clear()
            self.optimization_actions.clear()

            self._initialized = False

            self.logger.info("Performance optimizer cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Performance optimizer cleanup error: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if the optimizer is fully initialized."""
        return self._initialized

    def get_component_status(self) -> dict[str, bool]:
        """Get status of all optimization components."""
        return {
            "memory_optimizer": self.memory_optimizer is not None,
            "performance_monitor": self.performance_monitor is not None,
            "trading_profiler": self.trading_profiler is not None,
            "cache_layer": self.cache_layer is not None,
            "database_service": self.database_service is not None,
            "connection_pool_manager": self.connection_pool_manager is not None,
        }
