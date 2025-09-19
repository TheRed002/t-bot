"""
Bot health and performance monitoring system using service layer architecture.

This module implements the BotMonitor class that provides comprehensive
monitoring of bot instances through service layer dependencies, including
health checks, performance metrics collection, alert generation, and error
pattern detection.

REFACTORED: Now uses service layer pattern with proper dependency injection:
- BotService: Bot management operations and status
- StateService: Bot state monitoring and persistence
- RiskService: Risk assessment and monitoring integration
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import psutil

from src.core.base.service import BaseService
from src.core.data_transformer import CoreDataTransformer
from src.core.exceptions import (
    NetworkError,
    ServiceError,
    ValidationError,
)
from src.core.types import BotMetrics, BotStatus
from src.utils.bot_health_utils import HealthCheckUtils

# Import common utilities
from src.utils.bot_service_helpers import (
    safe_import_decorators,
    safe_import_error_handling,
    safe_import_monitoring,
)

# Production configuration constants
DEFAULT_MONITORING_INTERVAL = 30
DEFAULT_HEALTH_CHECK_INTERVAL = 60
DEFAULT_METRICS_COLLECTION_INTERVAL = 10
DEFAULT_ALERT_RETENTION_HOURS = 24
DEFAULT_MAX_METRICS_HISTORY = 1000

# Performance threshold constants
DEFAULT_CPU_WARNING_THRESHOLD = 70.0
DEFAULT_CPU_CRITICAL_THRESHOLD = 90.0
DEFAULT_MEMORY_WARNING_THRESHOLD = 500.0
DEFAULT_MEMORY_CRITICAL_THRESHOLD = 1000.0
DEFAULT_ERROR_RATE_WARNING = 0.15
DEFAULT_ERROR_RATE_CRITICAL = 0.30
DEFAULT_HEARTBEAT_TIMEOUT = 120
DEFAULT_WIN_RATE_THRESHOLD = 0.30

# Network and timeout constants
DEFAULT_WEBSOCKET_TIMEOUT = 30.0
DEFAULT_HEARTBEAT_INTERVAL = 10.0
DEFAULT_CONNECTION_TIMEOUT = 5.0
DEFAULT_CLEANUP_TIMEOUT = 15.0
DEFAULT_PING_TIMEOUT = 2.0

# Performance monitoring constants
CPU_DEVIATION_THRESHOLD = 1.0
MEMORY_DEVIATION_THRESHOLD = 1.0
CPU_TREND_WARNING = 2.0
CPU_TREND_CRITICAL = 5.0
MEMORY_TREND_WARNING = 10.0
MEMORY_TREND_CRITICAL = 20.0

# Get error handling components with fallback
_error_handling = safe_import_error_handling()
ErrorSeverity = _error_handling.get("ErrorSeverity")
get_global_error_handler = _error_handling["get_global_error_handler"]
with_circuit_breaker = _error_handling["with_circuit_breaker"]
with_error_context = _error_handling["with_error_context"]
with_retry = _error_handling["with_retry"]

# Get monitoring components with fallback
_monitoring = safe_import_monitoring()
MetricsCollector = _monitoring.get("MetricsCollector")
RiskMetrics = _monitoring.get("RiskMetrics")
TradingMetrics = _monitoring["TradingMetrics"]
get_tracer = _monitoring["get_tracer"]


# REMOVED: Direct database error decorator - using service layer pattern instead
# Get decorators with fallback
_decorators = safe_import_decorators()
log_calls = _decorators["log_calls"]

# Forward references for service dependencies - use interfaces to avoid circular imports
if TYPE_CHECKING:
    from src.risk_management.service import RiskService
    from src.state import StateService

    from .service import BotService


class BotMonitor(BaseService):
    """
    Comprehensive bot health and performance monitoring system using service layer.

    This service provides:
    - Bot status monitoring and health checks via BotService
    - Performance metrics collection and analysis via StateService
    - Alert generation for anomalies and issues
    - Resource usage tracking per bot through service layer
    - Error pattern detection and reporting
    - Historical performance analysis via service layer

    The monitor coordinates with other services rather than accessing data directly.
    """

    def __init__(self):
        """
        Initialize bot monitor with service layer dependencies.

        Dependencies resolved via DI:
        - BotService: Bot management operations and status queries
        - StateService: Bot state monitoring and persistence
        - RiskService: Risk assessment and alert integration
        """
        super().__init__(name="BotMonitor")

        # Declare service dependencies for DI resolution - avoid circular dependency with BotService
        self.add_dependency("StateService")
        self.add_dependency("RiskServiceInterface")
        self.add_dependency("MetricsCollector")  # Add monitoring dependency
        self.add_dependency("ConfigService")

        # Initialize error handler
        self.error_handler = get_global_error_handler()

        # Service instances (resolved during startup)
        self._bot_service: BotService | None = None
        self._state_service: StateService | None = None
        self._risk_service: RiskService | None = None
        self._metrics_collector: MetricsCollector | None = None
        self._config_service = None

        # Initialize monitoring components with error handling
        try:
            self._tracer = get_tracer(__name__) if get_tracer else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize tracer: {e}")
            self._tracer = None

        try:
            # Use singleton metrics collector to avoid duplicate metric registrations
            from src.monitoring import get_metrics_collector

            metrics_collector = get_metrics_collector()
            self._trading_metrics = metrics_collector.trading_metrics if TradingMetrics else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize trading metrics: {e}")
            self._trading_metrics = None

        try:
            # Use singleton metrics collector to avoid duplicate metric registrations
            from src.monitoring import get_metrics_collector

            metrics_collector = get_metrics_collector()
            self._risk_metrics = metrics_collector.risk_metrics if RiskMetrics else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize risk metrics: {e}")
            self._risk_metrics = None

        # Bot monitoring state (local cache for performance)
        self.monitored_bots: dict[str, dict[str, Any]] = {}
        self.bot_health_status: dict[str, dict[str, Any]] = {}
        self.performance_baselines: dict[str, dict[str, float]] = {}

        # Monitor state (is_running is handled by BaseComponent)
        self.monitoring_task = None
        self.health_check_task = None
        self.metrics_collection_task = None

        # Alert tracking (local cache - persisted via StateService)
        self.active_alerts: dict[str, list[dict[str, Any]]] = {}
        self.alert_history: list[dict[str, Any]] = []

        # Metrics history storage (local cache)
        self.metrics_history: dict[str, list[BotMetrics]] = {}

        # Load configuration from config service
        config = self._load_configuration()

        # Monitoring intervals from configuration
        self.monitoring_interval = config.get("monitoring_interval", DEFAULT_MONITORING_INTERVAL)
        self.health_check_interval = config.get("health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL)
        self.metrics_collection_interval = config.get("metrics_collection_interval", DEFAULT_METRICS_COLLECTION_INTERVAL)
        self.alert_retention_hours = config.get("alert_retention_hours", DEFAULT_ALERT_RETENTION_HOURS)

        # Performance thresholds from configuration
        thresholds_config = config.get("performance_thresholds", {})
        self.performance_thresholds = {
            "cpu_usage_warning": thresholds_config.get("cpu_usage_warning", DEFAULT_CPU_WARNING_THRESHOLD),
            "cpu_usage_critical": thresholds_config.get("cpu_usage_critical", DEFAULT_CPU_CRITICAL_THRESHOLD),
            "memory_usage_warning": thresholds_config.get("memory_usage_warning", DEFAULT_MEMORY_WARNING_THRESHOLD),
            "memory_usage_critical": thresholds_config.get("memory_usage_critical", DEFAULT_MEMORY_CRITICAL_THRESHOLD),
            "error_rate_warning": thresholds_config.get("error_rate_warning", DEFAULT_ERROR_RATE_WARNING),
            "error_rate_critical": thresholds_config.get("error_rate_critical", DEFAULT_ERROR_RATE_CRITICAL),
            "heartbeat_timeout": thresholds_config.get("heartbeat_timeout", DEFAULT_HEARTBEAT_TIMEOUT),
            "win_rate_threshold": thresholds_config.get("win_rate_threshold", DEFAULT_WIN_RATE_THRESHOLD),
        }

        # Monitoring statistics
        self.monitoring_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "bots_monitored": 0,
            "last_monitoring_time": None,
        }

        self._logger.info("BotMonitor initialized with service layer dependencies")

    def _load_configuration(self) -> dict[str, Any]:
        """Load monitoring configuration from config service."""
        try:
            # Load configuration from config service or use production defaults
            return {
                "monitoring_interval": DEFAULT_MONITORING_INTERVAL,
                "health_check_interval": DEFAULT_HEALTH_CHECK_INTERVAL,
                "metrics_collection_interval": DEFAULT_METRICS_COLLECTION_INTERVAL,
                "alert_retention_hours": DEFAULT_ALERT_RETENTION_HOURS,
                "performance_thresholds": {
                    "cpu_usage_warning": DEFAULT_CPU_WARNING_THRESHOLD,
                    "cpu_usage_critical": DEFAULT_CPU_CRITICAL_THRESHOLD,
                    "memory_usage_warning": DEFAULT_MEMORY_WARNING_THRESHOLD,
                    "memory_usage_critical": DEFAULT_MEMORY_CRITICAL_THRESHOLD,
                    "error_rate_warning": DEFAULT_ERROR_RATE_WARNING,
                    "error_rate_critical": DEFAULT_ERROR_RATE_CRITICAL,
                    "heartbeat_timeout": DEFAULT_HEARTBEAT_TIMEOUT,
                    "win_rate_threshold": DEFAULT_WIN_RATE_THRESHOLD,
                    "max_metrics_history": DEFAULT_MAX_METRICS_HISTORY,
                },
            }
        except Exception as e:
            self._logger.warning(f"Failed to load monitoring configuration: {e}")
            return {}

    @with_error_context(component="BotMonitor", operation="startup")
    async def _do_start(self) -> None:
        """
        Start the bot monitoring system with service layer dependencies.

        Raises:
            ServiceError: If startup fails
        """
        if self.is_running:
            self._logger.warning("Bot monitor is already running")
            return

        self._logger.info("Starting bot monitor with service layer")

        # Load configuration from config service
        try:
            config_service = self.resolve_dependency("Config")
            bot_monitor_config = config_service.bot_management.get("monitoring", {})

            # Update intervals from config
            self.monitoring_interval = bot_monitor_config.get(
                "monitoring_interval", self.monitoring_interval
            )
            self.health_check_interval = bot_monitor_config.get(
                "health_check_interval", self.health_check_interval
            )
            self.metrics_collection_interval = bot_monitor_config.get(
                "metrics_collection_interval", self.metrics_collection_interval
            )
            self.alert_retention_hours = bot_monitor_config.get(
                "alert_retention_hours", self.alert_retention_hours
            )

            # Update thresholds from config
            thresholds_config = bot_monitor_config.get("thresholds", {})
            self.performance_thresholds.update(thresholds_config)

        except Exception as e:
            self._logger.warning(f"Failed to load configuration, using defaults: {e}")

        # Resolve service dependencies
        self._bot_service = self.resolve_dependency("BotService")
        self._state_service = self.resolve_dependency("StateService")

        # Risk service is optional - handle gracefully if not available
        try:
            self._risk_service = self.resolve_dependency("RiskService")
        except Exception as e:
            self._logger.warning(
                f"RiskService not available, monitoring will continue without risk metrics: {e}"
            )
            self._risk_service = None

        self._metrics_collector = self.resolve_dependency("MetricsCollector")
        self._config_service = self.resolve_dependency("ConfigService")

        # Verify core dependencies are resolved (risk service is optional)
        if not all([self._bot_service, self._state_service]):
            raise ServiceError("Failed to resolve all required service dependencies")

        # Validate MetricsCollector is available
        if not self._metrics_collector:
            self._logger.warning("MetricsCollector not available - monitoring will be limited")

        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())

        # is_running is managed by BaseComponent

        self._logger.info(
            "Bot monitor started successfully with service dependencies",
            dependencies=["BotService", "StateService", "RiskService"],
        )

    @with_error_context(component="BotMonitor", operation="shutdown")
    async def _do_stop(self) -> None:
        """
        Stop the bot monitoring system.

        Raises:
            ServiceError: If shutdown fails
        """
        if not self.is_running:
            self._logger.warning("Bot monitor is not running")
            return

        self._logger.info("Stopping bot monitor")
        # is_running is managed by BaseComponent

        # Stop monitoring tasks
        tasks = [self.monitoring_task, self.health_check_task, self.metrics_collection_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clear local state (services will handle their own cleanup)
        self.monitored_bots.clear()
        self.bot_health_status.clear()
        self.active_alerts.clear()
        self.metrics_history.clear()

        self._logger.info("Bot monitor stopped successfully")

    @log_calls
    @with_error_context(component="BotMonitor", operation="register_bot")
    async def register_bot(self, bot_id: str) -> None:
        """
        Register a bot for monitoring.

        Args:
            bot_id: Bot identifier

        Raises:
            ValidationError: If registration is invalid
        """
        if not bot_id or not isinstance(bot_id, str):
            raise ValidationError("Bot ID must be a non-empty string")

        if bot_id in self.monitored_bots:
            self._logger.warning("Bot already registered for monitoring", bot_id=bot_id)
            return

        # Initialize monitoring data
        self.monitored_bots[bot_id] = {
            "registered_at": datetime.now(timezone.utc),
            "last_health_check": None,
            "last_metrics_collection": None,
            "consecutive_failures": 0,
            "total_health_checks": 0,
            "health_check_failures": 0,
        }

        # Initialize health status
        self.bot_health_status[bot_id] = {
            "status": "unknown",
            "last_heartbeat": None,
            "health_score": 0.0,
            "issues": [],
            "last_updated": datetime.now(timezone.utc),
        }

        # Initialize performance baseline
        self.performance_baselines[bot_id] = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "trade_frequency": 0.0,
            "error_rate": 0.0,
            "baseline_established": False,
        }

        # Initialize alert tracking
        self.active_alerts[bot_id] = []

        self.monitoring_stats["bots_monitored"] += 1

        self._logger.info("Bot registered for monitoring", bot_id=bot_id)

    @log_calls
    @with_error_context(component="BotMonitor", operation="unregister_bot")
    async def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot from monitoring.

        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self.monitored_bots:
            self._logger.warning("Bot not registered for monitoring", bot_id=bot_id)
            return

        # Remove from all tracking
        del self.monitored_bots[bot_id]
        del self.bot_health_status[bot_id]
        del self.performance_baselines[bot_id]
        del self.active_alerts[bot_id]

        self.monitoring_stats["bots_monitored"] -= 1

        self._logger.info("Bot unregistered from monitoring", bot_id=bot_id)

    @log_calls
    @with_error_context(component="BotMonitor", operation="update_bot_metrics")
    async def update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None:
        """
        Update metrics for a monitored bot.

        Args:
            bot_id: Bot identifier
            metrics: Current bot metrics
        """
        if not metrics:
            raise ValidationError("Metrics cannot be None")

        if bot_id not in self.monitored_bots:
            await self.register_bot(bot_id)

        current_time = datetime.now(timezone.utc)

        # Update monitoring tracking
        self.monitored_bots[bot_id]["last_metrics_collection"] = current_time

        # Store metrics in history
        if bot_id not in self.metrics_history:
            self.metrics_history[bot_id] = []
        self.metrics_history[bot_id].append(metrics)

        # Keep only recent history (limit to prevent memory bloat)
        max_metrics_history = self.performance_thresholds.get("max_metrics_history", 1000)
        if len(self.metrics_history[bot_id]) > max_metrics_history:
            self.metrics_history[bot_id] = self.metrics_history[bot_id][-max_metrics_history:]

        # Update health status
        await self._update_bot_health_status(bot_id, metrics)

        # Store metrics in InfluxDB
        await self._store_metrics(bot_id, metrics)

        # Check for performance anomalies
        await self._check_performance_anomalies(bot_id, metrics)

        # Update performance baseline
        await self._update_performance_baseline(bot_id, metrics)

    @log_calls
    @with_error_context(component="BotMonitor", operation="check_bot_health")
    @with_retry(max_attempts=2, base_delay=1.0, exceptions=(NetworkError, ServiceError))
    async def check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]:
        """
        Perform comprehensive health check for a bot.

        Args:
            bot_id: Bot identifier
            bot_status: Current bot status

        Returns:
            dict: Health check results
        """
        if bot_id not in self.monitored_bots:
            return {"error": "Bot not registered for monitoring"}

        # Update monitoring statistics
        current_time = await self._update_monitoring_stats(bot_id)

        # Initialize health results structure
        health_results = await self._initialize_health_results(bot_id, current_time)

        # Perform all health checks
        await self._perform_health_checks(bot_id, bot_status, health_results)

        # Calculate and update health scores
        await self._calculate_and_update_health_score(bot_id, bot_status, health_results)

        # Finalize health results
        await self._finalize_health_results(bot_id, health_results, current_time)

        return health_results

    async def _update_monitoring_stats(self, bot_id: str) -> datetime:
        """Update monitoring statistics for bot health check."""
        current_time = datetime.now(timezone.utc)
        monitoring_data = self.monitored_bots[bot_id]

        # Update monitoring statistics
        monitoring_data["total_health_checks"] += 1
        monitoring_data["last_health_check"] = current_time
        self.monitoring_stats["total_checks"] += 1

        # Record health check in monitoring system with error handling
        if self._metrics_collector:
            try:
                await self._metrics_collector.increment(
                    "bot_health_checks_total", labels={"bot_id": bot_id}
                )
            except Exception as e:
                self._logger.debug(f"Failed to record health check metric: {e}")

        return current_time

    async def _initialize_health_results(
        self, bot_id: str, current_time: datetime
    ) -> dict[str, Any]:
        """Initialize health results structure."""
        return {
            "bot_id": bot_id,
            "timestamp": current_time.isoformat(),
            "overall_health": "healthy",
            "health_score": 1.0,
            "checks": {},
            "issues": [],
            "recommendations": [],
        }

    async def _perform_health_checks(
        self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]
    ) -> None:
        """Perform all individual health checks."""
        # Status check
        status_healthy = await self._check_bot_status(bot_id, bot_status)
        health_results["checks"]["status"] = status_healthy

        # Heartbeat check
        heartbeat_healthy = await self._check_bot_heartbeat(bot_id)
        health_results["checks"]["heartbeat"] = heartbeat_healthy

        # Resource usage check
        resource_healthy = await self._check_resource_usage(bot_id)
        health_results["checks"]["resources"] = resource_healthy

        # Performance check
        performance_healthy = await self._check_performance_health(bot_id)
        health_results["checks"]["performance"] = performance_healthy

        # Error rate check
        error_rate_healthy = await self._check_error_rate(bot_id)
        health_results["checks"]["error_rate"] = error_rate_healthy

        # Risk health check
        risk_healthy = await self._check_risk_health(bot_id)
        health_results["checks"]["risk"] = risk_healthy

    async def _calculate_and_update_health_score(
        self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]
    ) -> None:
        """Calculate and update overall health score."""
        # Calculate overall health score using dedicated method or fallback
        if self.metrics_history.get(bot_id):
            latest_metrics = self.metrics_history[bot_id][-1]
            health_results["health_score"] = await self._calculate_health_score(
                bot_id, bot_status, latest_metrics
            )
        else:
            health_results["health_score"] = await self._calculate_fallback_health_score(
                health_results
            )

        # Record health score in monitoring with error handling
        if self._metrics_collector:
            try:
                self._metrics_collector.gauge(
                    "bot_health_score", health_results["health_score"], labels={"bot_id": bot_id}
                )
            except Exception as e:
                self._logger.debug(f"Failed to record health score metric: {e}")

        # Determine overall health
        await self._determine_overall_health_status(health_results)

    async def _calculate_fallback_health_score(self, health_results: dict[str, Any]) -> float:
        """Calculate fallback health score from check scores."""
        check_scores = [
            health_results["checks"]["status"].get("score", 0.0),
            health_results["checks"]["heartbeat"].get("score", 0.0),
            health_results["checks"]["resources"].get("score", 0.0),
            health_results["checks"]["performance"].get("score", 0.0),
            health_results["checks"]["error_rate"].get("score", 0.0),
        ]
        return sum(check_scores) / len(check_scores) if check_scores else 0.0

    async def _determine_overall_health_status(self, health_results: dict[str, Any]) -> None:
        """Determine overall health status based on health score."""
        if health_results["health_score"] >= 0.8:
            health_results["overall_health"] = "healthy"
        elif health_results["health_score"] >= 0.6:
            health_results["overall_health"] = "warning"
        else:
            health_results["overall_health"] = "critical"

    async def _finalize_health_results(
        self, bot_id: str, health_results: dict[str, Any], current_time: datetime
    ) -> None:
        """Finalize health results by collecting issues and updating bot health status."""
        # Get issues from bot health status that was updated by update_bot_metrics
        if bot_id in self.bot_health_status:
            bot_issues = self.bot_health_status[bot_id].get("issues", [])
            health_results["issues"].extend(bot_issues)

        # Collect issues and recommendations from individual checks
        for _check_name, check_result in health_results["checks"].items():
            if check_result.get("issues"):
                health_results["issues"].extend(check_result["issues"])
            if check_result.get("recommendations"):
                health_results["recommendations"].extend(check_result["recommendations"])

        # Update bot health status
        self.bot_health_status[bot_id].update(
            {
                "status": health_results["overall_health"],
                "health_score": health_results["health_score"],
                "issues": health_results["issues"],
                "last_updated": current_time,
            }
        )

        # Generate alerts if needed
        await self._generate_health_alerts(bot_id, health_results)

        return health_results

    @with_error_context(component="BotMonitor", operation="get_monitoring_summary")
    async def get_monitoring_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary for all bots."""

        # Aggregate health statistics
        health_summary = {"healthy": 0, "warning": 0, "critical": 0, "error": 0, "unknown": 0}

        # Aggregate performance statistics
        performance_summary = {
            "total_cpu_usage": 0.0,
            "total_memory_usage": 0.0,
            "average_health_score": 0.0,
            "bots_with_issues": 0,
        }

        # Process each monitored bot
        bot_summaries = {}
        total_health_score = 0.0

        for bot_id, health_status in self.bot_health_status.items():
            status = health_status["status"]
            health_summary[status] = health_summary.get(status, 0) + 1

            health_score = health_status["health_score"]
            total_health_score += health_score

            if health_status["issues"]:
                performance_summary["bots_with_issues"] += 1

            bot_summaries[bot_id] = {
                "status": status,
                "health_score": health_score,
                "last_updated": health_status["last_updated"].isoformat(),
                "active_alerts": len(self.active_alerts.get(bot_id, [])),
                "issues_count": len(health_status["issues"]),
            }

        # Calculate averages
        bot_count = len(self.bot_health_status)
        if bot_count > 0:
            performance_summary["average_health_score"] = total_health_score / bot_count

        # Count active alerts
        total_active_alerts = sum(len(alerts) for alerts in self.active_alerts.values())
        critical_alerts = sum(
            1
            for alerts in self.active_alerts.values()
            for alert in alerts
            if alert.get("severity") == "critical"
        )

        monitoring_data = {
            "monitoring_overview": {
                "monitored_bots": self.monitoring_stats["bots_monitored"],
                "is_running": self.is_running,
                "total_checks_performed": self.monitoring_stats["total_checks"],
                "active_alerts": total_active_alerts,
                "critical_alerts": critical_alerts,
                "last_monitoring_cycle": (
                    self.monitoring_stats["last_monitoring_time"].isoformat()
                    if self.monitoring_stats["last_monitoring_time"]
                    else None
                ),
            },
            "bot_health_summary": health_summary,
            "alert_summary": {
                "total_alerts": self.monitoring_stats["alerts_generated"],
                "critical_alerts_total": self.monitoring_stats["critical_alerts"],
                "warning_alerts_total": self.monitoring_stats["warning_alerts"],
            },
            "performance_overview": performance_summary,
            "system_health": {
                "average_health_score": performance_summary["average_health_score"],
                "bots_with_issues": performance_summary["bots_with_issues"],
            },
            "bot_summaries": bot_summaries,
        }

        # Apply consistent data transformation for monitoring data communication
        return CoreDataTransformer.apply_cross_module_consistency(
            CoreDataTransformer.transform_for_request_reply_pattern(
                "MONITORING_SUMMARY",
                monitoring_data,
                metadata={
                    "target_modules": ["analytics", "web_interface", "risk_management"],
                    "processing_priority": "normal",
                    "data_type": "monitoring_summary"
                }
            ),
            target_module="analytics",
            source_module="bot_management"
        )

    async def get_bot_health_details(self, bot_id: str) -> dict[str, Any] | None:
        """
        Get detailed health information for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Detailed health information or None if not found
        """
        if bot_id not in self.bot_health_status:
            return None

        health_status = self.bot_health_status[bot_id]
        monitoring_data = self.monitored_bots.get(bot_id, {})
        baseline_data = self.performance_baselines.get(bot_id, {})
        active_alerts = self.active_alerts.get(bot_id, [])

        return {
            "bot_id": bot_id,
            "health_status": {
                "status": health_status["status"],
                "health_score": health_status["health_score"],
                "last_heartbeat": (
                    health_status["last_heartbeat"].isoformat()
                    if health_status["last_heartbeat"]
                    else None
                ),
                "last_updated": health_status["last_updated"].isoformat(),
                "issues": health_status["issues"],
            },
            "monitoring_statistics": {
                "registered_at": (
                    monitoring_data.get("registered_at").isoformat()
                    if monitoring_data.get("registered_at")
                    else None
                ),
                "total_health_checks": monitoring_data.get("total_health_checks", 0),
                "health_check_failures": monitoring_data.get("health_check_failures", 0),
                "consecutive_failures": monitoring_data.get("consecutive_failures", 0),
                "last_health_check": (
                    monitoring_data.get("last_health_check").isoformat()
                    if monitoring_data.get("last_health_check")
                    else None
                ),
            },
            "performance_baseline": baseline_data,
            "active_alerts": active_alerts,
            "alert_history_24h": [
                alert
                for alert in self.alert_history
                if (
                    alert["bot_id"] == bot_id
                    and (
                        datetime.now(timezone.utc) - datetime.fromisoformat(alert["timestamp"])
                    ).total_seconds()
                    < 86400
                )
            ],
        }

    @with_error_context(component="BotMonitor", operation="health_check_loop")
    async def _health_check_loop(self) -> None:
        """Health check loop for all monitored bots."""
        try:
            while self.is_running:
                try:
                    # Perform health checks for all bots
                    for bot_id in list(self.monitored_bots.keys()):
                        try:
                            # Note: In a real implementation, we would get actual bot status
                            # For now, assume bot is running if recently registered
                            await self.check_bot_health(bot_id, BotStatus.RUNNING)
                        except (NetworkError, ServiceError) as e:
                            await self.error_handler.handle_error(
                                e,
                                context={"operation": "health_check", "bot_id": bot_id},
                                severity=ErrorSeverity.MEDIUM.value,
                            )
                        except ValidationError as e:
                            await self.error_handler.handle_error(
                                e,
                                context={"operation": "health_check", "bot_id": bot_id},
                                severity=ErrorSeverity.LOW.value,
                            )

                    # Wait for next cycle
                    await asyncio.sleep(self.health_check_interval)

                except (NetworkError, ServiceError) as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "health_check_loop"},
                        severity=ErrorSeverity.MEDIUM.value,
                    )
                    await asyncio.sleep(30)
                except Exception as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "health_check_loop"},
                        severity=ErrorSeverity.HIGH.value,
                    )
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            self._logger.info("Health check loop cancelled")

    @with_error_context(component="BotMonitor", operation="metrics_collection_loop")
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        try:
            while self.is_running:
                try:
                    # Collect system-wide metrics
                    await self._collect_system_metrics()

                    # Collect risk metrics for all bots
                    for bot_id in list(self.monitored_bots.keys()):
                        try:
                            await self._collect_risk_metrics(bot_id)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to collect risk metrics for bot {bot_id}: {e}"
                            )

                    # Update performance baselines
                    await self._update_all_baselines()

                    # Wait for next cycle
                    await asyncio.sleep(self.metrics_collection_interval)

                except ServiceError as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "metrics_collection_loop"},
                        severity=ErrorSeverity.HIGH.value,
                    )
                    await asyncio.sleep(60)  # Longer delay for service issues
                except NetworkError as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "metrics_collection_loop"},
                        severity=ErrorSeverity.MEDIUM.value,
                    )
                    await asyncio.sleep(30)
                except Exception as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "metrics_collection_loop"},
                        severity=ErrorSeverity.HIGH.value,
                    )
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            self._logger.info("Metrics collection loop cancelled")

    async def _update_bot_health_status(self, bot_id: str, metrics: BotMetrics) -> None:
        """Update bot health status based on metrics."""
        current_time = datetime.now(timezone.utc)

        if bot_id not in self.bot_health_status:
            return

        health_status = self.bot_health_status[bot_id]

        # Update heartbeat
        if metrics.last_heartbeat:
            health_status["last_heartbeat"] = metrics.last_heartbeat

        # Check for performance issues
        issues = await self._check_performance_issues(metrics)

        health_status["issues"] = issues
        health_status["last_updated"] = current_time

    async def _check_performance_issues(self, metrics: BotMetrics) -> list[str]:
        """Check for performance issues in bot metrics."""
        issues: list[dict[str, Any]] = []

        # CPU usage check
        await self._check_cpu_usage_issues(metrics, issues)

        # Memory usage check
        await self._check_memory_usage_issues(metrics, issues)

        # Error rate check
        await self._check_error_rate_issues(metrics, issues)

        # Win rate check
        await self._check_win_rate_issues(metrics, issues)

        return issues

    async def _check_cpu_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None:
        """Check for CPU usage issues."""
        if metrics.cpu_usage > self.performance_thresholds["cpu_usage_critical"]:
            issues.append(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > self.performance_thresholds["cpu_usage_warning"]:
            issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")

    async def _check_memory_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None:
        """Check for memory usage issues."""
        if metrics.memory_usage > self.performance_thresholds["memory_usage_critical"]:
            issues.append(f"Critical memory usage: {metrics.memory_usage:.1f} MB")
        elif metrics.memory_usage > self.performance_thresholds["memory_usage_warning"]:
            issues.append(f"High memory usage: {metrics.memory_usage:.1f} MB")

    async def _check_error_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None:
        """Check for error rate issues."""
        if metrics.total_trades > 0:
            error_rate = metrics.error_count / metrics.total_trades
            if error_rate > self.performance_thresholds["error_rate_critical"]:
                issues.append(f"Critical error rate: {error_rate:.1%}")
            elif error_rate > self.performance_thresholds["error_rate_warning"]:
                issues.append(f"High error rate: {error_rate:.1%}")

    async def _check_win_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None:
        """Check for win rate issues."""
        if metrics.win_rate < self.performance_thresholds["win_rate_threshold"]:
            issues.append(f"Low win rate: {metrics.win_rate:.1%}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_error_context(component="BotMonitor", operation="store_metrics")
    async def _store_metrics(self, bot_id: str, metrics: BotMetrics) -> None:
        """Store metrics using state service - proper service layer abstraction."""
        try:
            if not self._state_service:
                self._logger.warning("StateService not available for metrics storage")
                return

            # Create metrics record for storage through state service
            metrics_record = {
                "bot_id": bot_id,
                "total_trades": metrics.total_trades,
                "profitable_trades": metrics.profitable_trades,
                "losing_trades": metrics.losing_trades,
                "total_pnl": str(metrics.total_pnl),
                "unrealized_pnl": str(metrics.unrealized_pnl),
                "win_rate": str(metrics.win_rate),
                "average_trade_pnl": str(metrics.average_trade_pnl),
                "max_drawdown": str(metrics.max_drawdown),
                "uptime_percentage": metrics.uptime_percentage,
                "error_count": metrics.error_count,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "api_calls_count": metrics.api_calls_made,
                "timestamp": datetime.now(timezone.utc),
            }

            # Store through state service layer using proper service abstraction
            await self._state_service.store_metrics(bot_id, metrics_record)

            self._logger.debug(
                "Stored metrics through state service",
                bot_id=bot_id,
                metrics_count=len(metrics_record),
            )

        except Exception as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "store_metrics", "bot_id": bot_id},
                severity=ErrorSeverity.MEDIUM.value,
            )

    async def _check_performance_anomalies(self, bot_id: str, metrics: BotMetrics) -> None:
        """Check for performance anomalies against baseline."""
        if bot_id not in self.performance_baselines:
            return

        baseline = self.performance_baselines[bot_id]

        if not baseline.get("baseline_established"):
            return

        # Check for significant deviations
        anomalies: list[dict[str, Any]] = []

        # CPU usage anomaly - handle both dict and float baseline formats
        cpu_baseline = baseline.get("cpu_usage", 0)
        if isinstance(cpu_baseline, dict):
            cpu_baseline = cpu_baseline.get("mean", 0)

        cpu_deviation = abs(metrics.cpu_usage - cpu_baseline) / max(cpu_baseline, CPU_DEVIATION_THRESHOLD)
        if cpu_deviation > 0.5:  # 50% deviation
            anomalies.append(
                {
                    "type": "cpu_anomaly",
                    "current": metrics.cpu_usage,
                    "baseline": cpu_baseline,
                    "deviation": cpu_deviation,
                }
            )

        # Memory usage anomaly - handle both dict and float baseline formats
        memory_baseline = baseline.get("memory_usage", 0)
        if isinstance(memory_baseline, dict):
            memory_baseline = memory_baseline.get("mean", 0)

        memory_deviation = abs(metrics.memory_usage - memory_baseline) / max(memory_baseline, MEMORY_DEVIATION_THRESHOLD)
        if memory_deviation > 0.5:  # 50% deviation
            anomalies.append(
                {
                    "type": "memory_anomaly",
                    "current": metrics.memory_usage,
                    "baseline": memory_baseline,
                    "deviation": memory_deviation,
                }
            )

        # Error rate anomaly
        current_error_rate = metrics.error_count / max(metrics.total_trades, 1)
        error_rate_deviation = abs(current_error_rate - baseline["error_rate"])
        if error_rate_deviation > 0.1:  # 10% absolute deviation
            anomalies.append(
                {
                    "type": "error_rate_anomaly",
                    "current": current_error_rate,
                    "baseline": baseline["error_rate"],
                    "deviation": error_rate_deviation,
                }
            )

        # Generate alerts for anomalies
        if anomalies:
            await self._generate_anomaly_alerts(bot_id, anomalies)

    async def _update_performance_baseline(self, bot_id: str, metrics: BotMetrics) -> None:
        """Update performance baseline with exponential moving average."""
        if bot_id not in self.performance_baselines:
            return

        baseline = self.performance_baselines[bot_id]
        alpha = 0.1  # Smoothing factor for exponential moving average

        if not baseline["baseline_established"]:
            # Initialize baseline
            baseline.update(
                {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "error_rate": metrics.error_count / max(metrics.total_trades, 1),
                    "baseline_established": True,
                }
            )
        else:
            # Update with exponential moving average - handle both dict and float formats
            cpu_current = baseline.get("cpu_usage", 0)
            if isinstance(cpu_current, dict):
                cpu_current = cpu_current.get("mean", 0)
            baseline["cpu_usage"] = (1 - alpha) * cpu_current + alpha * metrics.cpu_usage

            memory_current = baseline.get("memory_usage", 0)
            if isinstance(memory_current, dict):
                memory_current = memory_current.get("mean", 0)
            baseline["memory_usage"] = (1 - alpha) * memory_current + alpha * metrics.memory_usage

            current_error_rate = metrics.error_count / max(metrics.total_trades, 1)
            baseline["error_rate"] = (1 - alpha) * baseline.get(
                "error_rate", 0
            ) + alpha * current_error_rate

    async def _check_bot_status(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]:
        """Check bot status health."""
        score = 1.0
        issues: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        if bot_status == BotStatus.ERROR:
            score = 0.0
            issues.append("Bot is in error state")
            recommendations.append("Investigate error logs and restart bot")
        elif bot_status == BotStatus.STOPPED:
            score = 0.2
            issues.append("Bot is stopped")
            recommendations.append("Start bot if it should be running")
        elif bot_status == BotStatus.PAUSED:
            score = 0.5
            issues.append("Bot is paused")
            recommendations.append("Resume bot if pause was unintended")
        elif bot_status == BotStatus.INITIALIZING:
            score = 0.7
            issues.append("Bot is still starting")
        elif bot_status == BotStatus.STOPPING:
            score = 0.3
            issues.append("Bot is stopping")

        return {
            "score": score,
            "status": bot_status.value,
            "issues": issues,
            "recommendations": recommendations,
        }

    async def _check_bot_heartbeat(self, bot_id: str) -> dict[str, Any]:
        """Check bot heartbeat health."""
        if bot_id not in self.bot_health_status:
            return {"score": 0.0, "issues": ["Bot not in health status tracking"]}

        health_status = self.bot_health_status[bot_id]
        last_heartbeat = health_status.get("last_heartbeat")

        if not last_heartbeat:
            return {
                "score": 0.0,
                "issues": ["No heartbeat received"],
                "recommendations": ["Check bot connectivity and health monitoring"],
            }

        current_time = datetime.now(timezone.utc)
        heartbeat_age = (current_time - last_heartbeat).total_seconds()

        if heartbeat_age > self.performance_thresholds["heartbeat_timeout"]:
            return {
                "score": 0.0,
                "issues": [f"Heartbeat timeout: {heartbeat_age:.0f}s ago"],
                "recommendations": ["Check bot connectivity and restart if necessary"],
            }
        elif heartbeat_age > self.performance_thresholds["heartbeat_timeout"] / 2:
            return {
                "score": 0.5,
                "issues": [f"Stale heartbeat: {heartbeat_age:.0f}s ago"],
                "recommendations": ["Monitor bot connectivity"],
            }
        else:
            return {"score": 1.0, "last_heartbeat_age": heartbeat_age}

    @with_error_context(component="BotMonitor", operation="check_resource_usage")
    async def _check_resource_usage(self, bot_id: str) -> dict[str, Any]:
        """Check resource usage health."""
        # In a real implementation, this would get actual resource usage
        # For now, simulate based on system resources
        try:
            process = psutil.Process()
            cpu_usage = process.cpu_percent()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            score = 1.0
            issues: list[dict[str, Any]] = []
            recommendations: list[dict[str, Any]] = []

            # CPU check
            if cpu_usage > self.performance_thresholds["cpu_usage_critical"]:
                score *= 0.2
                issues.append(f"Critical CPU usage: {cpu_usage:.1f}%")
                recommendations.append("Optimize bot algorithms or reduce trading frequency")
            elif cpu_usage > self.performance_thresholds["cpu_usage_warning"]:
                score *= 0.6
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                recommendations.append("Monitor CPU usage trends")

            # Memory check
            if memory_usage > self.performance_thresholds["memory_usage_critical"]:
                score *= 0.2
                issues.append(f"Critical memory usage: {memory_usage:.1f} MB")
                recommendations.append("Check for memory leaks and restart bot")
            elif memory_usage > self.performance_thresholds["memory_usage_warning"]:
                score *= 0.6
                issues.append(f"High memory usage: {memory_usage:.1f} MB")
                recommendations.append("Monitor memory usage trends")

            return {
                "score": score,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "issues": issues,
                "recommendations": recommendations,
            }

        except OSError as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "resource_check", "bot_id": bot_id},
                severity=ErrorSeverity.MEDIUM.value,
            )
            return {
                "score": 0.5,
                "issues": [f"System resource check failed: {e}"],
                "recommendations": ["Manual resource monitoring required"],
            }
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "resource_check", "bot_id": bot_id},
                severity=ErrorSeverity.LOW.value,
            )
            return {
                "score": 0.5,
                "issues": [f"Resource check failed: {e}"],
                "recommendations": ["Manual resource monitoring required"],
            }

    async def _check_performance_health(self, bot_id: str) -> dict[str, Any]:
        """Check trading performance health."""
        # This would integrate with actual bot metrics
        # For now, return baseline health check
        return {"score": 0.8, "issues": [], "recommendations": []}

    async def _check_error_rate(self, bot_id: str) -> dict[str, Any]:
        """Check error rate health."""
        # This would integrate with actual error tracking
        # For now, return baseline health check
        return {"score": 0.9, "issues": [], "recommendations": []}

    async def _check_risk_health(self, bot_id: str) -> dict[str, Any]:
        """Check risk health metrics."""
        score = 1.0
        issues: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []

        if not self._risk_service:
            return {
                "score": 0.5,
                "issues": ["Risk service not available"],
                "recommendations": ["Enable risk service for proper risk monitoring"],
            }

        try:
            # Skip risk metrics - method doesn't exist in RiskService
            # Risk metrics collection skipped - RiskService method not available
            self._logger.debug(
                "Risk metrics collection skipped - RiskService method not available", bot_id=bot_id
            )
            # Use placeholder values to maintain health score logic
            risk_metrics = None

            # Skip risk checks since we don't have metrics
            if risk_metrics is not None:
                # Check max drawdown
                if risk_metrics.max_drawdown > 0.2:  # 20% drawdown threshold
                    score *= 0.6
                    issues.append(f"High max drawdown: {risk_metrics.max_drawdown:.1%}")
                    recommendations.append("Implement stricter stop-loss controls")

        except Exception as e:
            self._logger.warning(f"Failed to check risk health for bot {bot_id}: {e}")
            score = 0.5
            issues.append(f"Risk health check failed: {e!s}")

        return {"score": score, "issues": issues, "recommendations": recommendations}

    async def _generate_health_alerts(self, bot_id: str, health_results: dict[str, Any]) -> None:
        """Generate alerts based on health check results."""
        if health_results["overall_health"] in ["warning", "critical"]:
            severity = "critical" if health_results["overall_health"] == "critical" else "warning"

            alert = {
                "alert_id": str(uuid.uuid4()),
                "bot_id": bot_id,
                "type": "health_check",
                "severity": severity,
                "message": f"Bot health check failed: {health_results['overall_health']}",
                "health_score": health_results["health_score"],
                "issues": health_results["issues"],
                "recommendations": health_results["recommendations"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "acknowledged": False,
            }

            # Add to active alerts
            self.active_alerts[bot_id].append(alert)
            self.alert_history.append(alert)

            # Update statistics
            self.monitoring_stats["alerts_generated"] += 1
            if severity == "critical":
                self.monitoring_stats["critical_alerts"] += 1
            else:
                self.monitoring_stats["warning_alerts"] += 1

            self._logger.warning(
                "Health alert generated for bot",
                bot_id=bot_id,
                severity=severity,
                health_score=health_results["health_score"],
                issues_count=len(health_results["issues"]),
            )

    async def _generate_anomaly_alerts(self, bot_id: str, anomalies: list[dict[str, Any]]) -> None:
        """Generate alerts for performance anomalies."""
        for anomaly in anomalies:
            alert = {
                "alert_id": str(uuid.uuid4()),
                "bot_id": bot_id,
                "type": "performance_anomaly",
                "severity": "warning",
                "message": f"Performance anomaly detected: {anomaly['type']}",
                "anomaly_data": anomaly,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "acknowledged": False,
            }

            self.active_alerts[bot_id].append(alert)
            self.alert_history.append(alert)
            self.monitoring_stats["alerts_generated"] += 1
            self.monitoring_stats["warning_alerts"] += 1

            self._logger.warning(
                "Performance anomaly alert generated",
                bot_id=bot_id,
                anomaly_type=anomaly["type"],
                deviation=anomaly["deviation"],
            )

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.alert_retention_hours)

        # Clean up alert history
        self.alert_history = [
            alert
            for alert in self.alert_history
            if (
                isinstance(alert, dict)
                and "timestamp" in alert
                and datetime.fromisoformat(alert["timestamp"]) > cutoff_time
            )
        ]

        # Clean up active alerts for acknowledged ones older than retention period
        for bot_id in self.active_alerts:
            self.active_alerts[bot_id] = [
                alert
                for alert in self.active_alerts[bot_id]
                if (
                    not alert.get("acknowledged")
                    or datetime.fromisoformat(alert["timestamp"]) > cutoff_time
                )
            ]

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations for unacknowledged critical alerts."""
        # Implementation would handle alert escalation logic
        # For now, just log critical alerts that need attention

        critical_alerts = []
        for bot_id, alerts in self.active_alerts.items():
            for alert in alerts:
                if (
                    isinstance(alert, dict)
                    and alert.get("severity") == "critical"
                    and not alert.get("acknowledged")
                ):
                    critical_alerts.append((bot_id, alert))

        if critical_alerts:
            self._logger.critical(
                "Unacknowledged critical alerts requiring attention", count=len(critical_alerts)
            )

    async def _update_monitoring_statistics(self) -> None:
        """Update monitoring statistics."""
        # Update bot count
        self.monitoring_stats["bots_monitored"] = len(self.monitored_bots)

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_error_context(component="BotMonitor", operation="collect_system_metrics")
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics using service layer abstraction."""
        try:
            # Collect system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Create system metrics record for state service
            system_metrics = {
                "component": "bot_monitor",
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "monitored_bots": len(self.monitored_bots),
                "active_alerts": sum(len(alerts) for alerts in self.active_alerts.values()),
                "timestamp": datetime.now(timezone.utc),
            }

            # Push to monitoring system with error handling
            if self._metrics_collector:
                try:
                    self._metrics_collector.gauge(
                        "bot_monitor_cpu_usage_percent",
                        cpu_percent,
                        labels={"component": "bot_monitor"},
                    )
                    self._metrics_collector.gauge(
                        "bot_monitor_memory_usage_percent",
                        memory.percent,
                        labels={"component": "bot_monitor"},
                    )
                    self._metrics_collector.gauge(
                        "bot_monitor_monitored_bots_count",
                        len(self.monitored_bots),
                        labels={"component": "bot_monitor"},
                    )
                    self._metrics_collector.gauge(
                        "bot_monitor_active_alerts_count",
                        sum(len(alerts) for alerts in self.active_alerts.values()),
                        labels={"component": "bot_monitor"},
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to push system metrics: {e}")

            # Store through state service layer using proper service abstraction
            # We'll use bot_id="system" to indicate system-wide metrics
            if self._state_service:
                await self._state_service.store_metrics(
                    "system_monitor", system_metrics
                )

            self._logger.debug(
                "Collected and stored system metrics through state service",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                monitored_bots=len(self.monitored_bots),
            )

        except OSError as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "collect_system_metrics"},
                severity=ErrorSeverity.MEDIUM.value,
            )
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "collect_system_metrics"},
                severity=ErrorSeverity.MEDIUM.value,
            )

    async def _update_all_baselines(self) -> None:
        """Update performance baselines for all bots."""
        try:
            for bot_id in list(self.bot_baselines.keys()):
                await self._establish_performance_baseline(bot_id)

            self._logger.debug(f"Updated baselines for {len(self.bot_baselines)} bots")

        except Exception as e:
            self._logger.error(f"Failed to update performance baselines: {e}")

    @with_error_context(component="BotMonitor", operation="collect_risk_metrics")
    async def _collect_risk_metrics(self, bot_id: str) -> None:
        """
        Collect risk metrics from RiskService for a bot.

        Args:
            bot_id: Bot identifier
        """
        if not self._risk_service:
            return

        try:
            # Skip risk metrics collection - method doesn't exist in RiskService
            # Risk metrics collection skipped - RiskService method not available
            self._logger.debug(
                "Risk metrics collection skipped - RiskService method not available", bot_id=bot_id
            )
            risk_metrics = None

            if risk_metrics and self._metrics_collector:
                # This block won't execute until get_risk_metrics is available
                try:
                    self._metrics_collector.gauge(
                        "bot_risk_var",
                        str(risk_metrics.value_at_risk),
                        labels={"bot_id": bot_id},
                    )
                    self._metrics_collector.gauge(
                        "bot_risk_sharpe_ratio",
                        str(risk_metrics.sharpe_ratio),
                        labels={"bot_id": bot_id},
                    )
                    self._metrics_collector.gauge(
                        "bot_risk_max_drawdown",
                        str(risk_metrics.max_drawdown),
                        labels={"bot_id": bot_id},
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to record risk metrics: {e}")

                # Check risk thresholds and generate alerts
                if risk_metrics.max_drawdown > self.performance_thresholds.get(
                    "max_drawdown_threshold", 0.2
                ):
                    await self._generate_alert(
                        bot_id,
                        "high_drawdown",
                        "critical",
                        f"Max drawdown is {risk_metrics.max_drawdown:.1%}",
                    )

                if risk_metrics.sharpe_ratio < self.performance_thresholds.get(
                    "min_sharpe_ratio", 0.5
                ):
                    await self._generate_alert(
                        bot_id,
                        "low_sharpe_ratio",
                        "warning",
                        f"Sharpe ratio is {risk_metrics.sharpe_ratio:.2f}",
                    )

        except Exception as e:
            await self.error_handler.handle_error(
                e,
                context={"operation": "collect_risk_metrics", "bot_id": bot_id},
                severity=ErrorSeverity.MEDIUM.value,
            )

    # Missing methods expected by tests

    @with_error_context(component="BotMonitor", operation="check_alert_conditions")
    async def _check_alert_conditions(self, bot_id: str) -> None:
        """
        Check alert conditions for a bot and generate alerts if needed.

        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self.monitored_bots:
            return

        # Get latest metrics from history
        if bot_id not in self.metrics_history or not self.metrics_history[bot_id]:
            return

        latest_metrics = self.metrics_history[bot_id][-1]

        # Check CPU threshold
        if latest_metrics.cpu_usage > self.performance_thresholds["cpu_usage_warning"]:
            severity = (
                "critical"
                if latest_metrics.cpu_usage > self.performance_thresholds["cpu_usage_critical"]
                else "warning"
            )
            await self._generate_alert(
                bot_id,
                "high_cpu_usage",
                severity,
                f"CPU usage is {latest_metrics.cpu_usage:.1f}%",
            )

        # Check memory threshold
        if latest_metrics.memory_usage > self.performance_thresholds["memory_usage_warning"]:
            severity = (
                "critical"
                if latest_metrics.memory_usage
                > self.performance_thresholds["memory_usage_critical"]
                else "warning"
            )
            await self._generate_alert(
                bot_id,
                "high_memory_usage",
                severity,
                f"Memory usage is {latest_metrics.memory_usage:.1f} MB",
            )

        # Check error rate
        if latest_metrics.total_trades > 0:
            error_rate = latest_metrics.error_count / latest_metrics.total_trades
            if error_rate > self.performance_thresholds["error_rate_warning"]:
                severity = (
                    "critical"
                    if error_rate > self.performance_thresholds["error_rate_critical"]
                    else "warning"
                )
                await self._generate_alert(
                    bot_id, "high_error_rate", severity, f"Error rate is {error_rate:.1%}"
                )

    @with_error_context(component="BotMonitor", operation="generate_alert")
    async def _generate_alert(
        self, bot_id: str, alert_type: str, severity: str, message: str
    ) -> None:
        """
        Generate an alert for a bot.

        Args:
            bot_id: Bot identifier
            alert_type: Type of alert
            severity: Alert severity (warning/critical)
            message: Alert message
        """
        # Check for rate limiting - don't generate duplicate alerts too frequently
        recent_alerts = [
            alert
            for alert in self.alert_history
            if (
                isinstance(alert, dict)
                and "bot_id" in alert
                and "alert_type" in alert
                and "timestamp" in alert
                and alert["bot_id"] == bot_id
                and alert["alert_type"] == alert_type
                and (
                    datetime.now(timezone.utc) - datetime.fromisoformat(alert["timestamp"])
                ).total_seconds()
                < 300  # 5 minutes
            )
        ]

        if len(recent_alerts) >= 3:  # Rate limit: max 3 alerts of same type per 5 minutes
            return

        alert = {
            "alert_id": str(uuid.uuid4()),
            "bot_id": bot_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "acknowledged": False,
        }

        # Add to active alerts and history
        if bot_id not in self.active_alerts:
            self.active_alerts[bot_id] = []

        self.active_alerts[bot_id].append(alert)
        self.alert_history.append(alert)

        # Update statistics
        self.monitoring_stats["alerts_generated"] += 1
        if severity == "critical":
            self.monitoring_stats["critical_alerts"] += 1
        else:
            self.monitoring_stats["warning_alerts"] += 1

        self._logger.warning(
            "Alert generated for bot", bot_id=bot_id, alert_type=alert_type, severity=severity
        )

    @with_error_context(component="BotMonitor", operation="establish_performance_baseline")
    async def _establish_performance_baseline(self, bot_id: str) -> None:
        """
        Establish performance baseline for a bot.

        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self.metrics_history or len(self.metrics_history[bot_id]) < 5:
            return  # Need at least 5 data points

        metrics_list = self.metrics_history[bot_id][-10:]  # Use last 10 data points

        # Calculate averages for baseline
        cpu_values = [m.cpu_usage for m in metrics_list if hasattr(m, "cpu_usage")]
        memory_values = [m.memory_usage for m in metrics_list if hasattr(m, "memory_usage")]

        if cpu_values and memory_values:
            baseline = {
                "cpu_usage": {
                    "mean": sum(cpu_values) / len(cpu_values),
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                },
                "memory_usage": {
                    "mean": sum(memory_values) / len(memory_values),
                    "min": min(memory_values),
                    "max": max(memory_values),
                },
                "established_at": datetime.now(timezone.utc),
                "sample_count": len(metrics_list),
            }

            self.performance_baselines[bot_id].update(baseline)
            self.performance_baselines[bot_id]["baseline_established"] = True

            self._logger.info("Performance baseline established for bot", bot_id=bot_id)

    @with_error_context(component="BotMonitor", operation="detect_anomalies")
    async def _detect_anomalies(self, bot_id: str, metrics: BotMetrics) -> list[dict[str, Any]]:
        """
        Detect performance anomalies against baseline.

        Args:
            bot_id: Bot identifier
            metrics: Current metrics

        Returns:
            List of detected anomalies
        """
        anomalies: list[dict[str, Any]] = []

        if bot_id not in self.performance_baselines:
            return anomalies

        baseline = self.performance_baselines[bot_id]
        if not baseline.get("baseline_established"):
            return anomalies

        # Check CPU anomaly
        if "cpu_usage" in baseline and hasattr(metrics, "cpu_usage"):
            cpu_baseline = baseline["cpu_usage"]
            if isinstance(cpu_baseline, dict):
                cpu_mean = cpu_baseline.get("mean", 0)
            else:
                cpu_mean = cpu_baseline

            cpu_deviation = abs(metrics.cpu_usage - cpu_mean) / max(cpu_mean, CPU_DEVIATION_THRESHOLD)

            if cpu_deviation > 0.5:  # 50% deviation from baseline
                anomalies.append(
                    {
                        "metric": "cpu_usage",
                        "current": metrics.cpu_usage,
                        "baseline": cpu_mean,
                        "deviation": cpu_deviation,
                        "severity": "high" if cpu_deviation > CPU_DEVIATION_THRESHOLD else "medium",
                    }
                )

        # Check memory anomaly
        if "memory_usage" in baseline and hasattr(metrics, "memory_usage"):
            memory_baseline = baseline["memory_usage"]
            if isinstance(memory_baseline, dict):
                memory_mean = memory_baseline.get("mean", 0)
            else:
                memory_mean = memory_baseline
            memory_deviation = abs(metrics.memory_usage - memory_mean) / max(memory_mean, MEMORY_DEVIATION_THRESHOLD)

            if memory_deviation > 0.5:  # 50% deviation from baseline
                anomalies.append(
                    {
                        "metric": "memory_usage",
                        "current": metrics.memory_usage,
                        "baseline": memory_mean,
                        "deviation": memory_deviation,
                        "severity": (
                            "high" if memory_deviation > MEMORY_DEVIATION_THRESHOLD else "medium"
                        ),
                    }
                )

        return anomalies

    @with_error_context(component="BotMonitor", operation="get_bot_health_history")
    async def get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get bot health history.

        Args:
            bot_id: Bot identifier
            hours: Number of hours to look back

        Returns:
            List of historical health records
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # For this implementation, we'll create a simple history based on metrics
        history = []

        if bot_id in self.metrics_history:
            for _i, metrics in enumerate(self.metrics_history[bot_id]):
                # Filter by cutoff time
                if (
                    hasattr(metrics, "metrics_updated_at")
                    and metrics.metrics_updated_at < cutoff_time
                ):
                    continue
                try:
                    # Create a health record for each metrics point
                    health_score = await self._calculate_health_score(
                        bot_id, BotStatus.RUNNING, metrics
                    )

                    history.append(
                        {
                            "timestamp": metrics.metrics_updated_at.isoformat(),
                            "health_score": health_score,
                            "cpu_usage": metrics.cpu_usage,
                            "memory_usage": metrics.memory_usage,
                            "error_count": metrics.error_count,
                            "total_trades": metrics.total_trades,
                        }
                    )
                except ValidationError as e:
                    await self.error_handler.handle_error(
                        e,
                        context={"operation": "get_health_history", "bot_id": bot_id},
                        severity=ErrorSeverity.LOW.value,
                    )
                    continue  # Skip this metrics entry

        return history

    @with_error_context(component="BotMonitor", operation="get_alert_history")
    async def get_alert_history(
        self, bot_id: str | None = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Get alert history, optionally filtered by bot ID.

        Args:
            bot_id: Optional bot identifier to filter by
            hours: Number of hours to look back

        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered_alerts = []
        for alert in self.alert_history:
            try:
                # Skip invalid alert objects
                if not isinstance(alert, dict) or "timestamp" not in alert or "bot_id" not in alert:
                    continue

                # Handle both string and datetime timestamps
                timestamp = alert["timestamp"]
                if isinstance(timestamp, str):
                    alert_time = datetime.fromisoformat(timestamp)
                else:
                    alert_time = timestamp

                if alert_time >= cutoff_time:
                    if bot_id is None or alert["bot_id"] == bot_id:
                        filtered_alerts.append(alert)
            except (ValueError, TypeError, KeyError) as e:
                await self.error_handler.handle_error(
                    e,
                    context={
                        "operation": "parse_alert_timestamp",
                        "alert_id": alert.get("alert_id"),
                    },
                    severity=ErrorSeverity.LOW.value,
                )
                continue  # Skip this alert

        def sort_key(alert):
            """Extract timestamp for sorting alerts."""
            timestamp = alert["timestamp"]
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp)
            return timestamp

        return sorted(filtered_alerts, key=sort_key, reverse=True)

    @with_error_context(component="BotMonitor", operation="calculate_health_score")
    async def _calculate_health_score(
        self, bot_id: str, bot_status: BotStatus, metrics: BotMetrics
    ) -> float:
        """
        Calculate a health score for a bot using extracted utilities.
        """
        # Use extracted health check utilities
        return HealthCheckUtils.calculate_health_score(bot_status, metrics)

    # REMOVED: Individual score calculation methods - now using HealthCheckUtils
    # These methods have been consolidated into src.utils.bot_health_utils

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self._logger.info("Starting monitoring loop")

        try:
            while self.is_running:
                try:
                    self.monitoring_stats["last_monitoring_time"] = datetime.now(timezone.utc)

                    # Clean up old alerts
                    await self._cleanup_old_alerts()

                    # Clean up old metrics
                    await self._cleanup_old_metrics()

                    # Process alert escalations
                    await self._process_alert_escalations()

                    # Update monitoring statistics
                    await self._update_monitoring_statistics()

                    # Wait for next cycle
                    await asyncio.sleep(self.monitoring_interval)

                except Exception as e:
                    self._logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self._logger.info("Monitoring loop cancelled")

    @with_error_context(component="BotMonitor", operation="cleanup_old_metrics")
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory bloat."""
        max_metrics_per_bot = 1000  # Keep last 1000 metrics per bot

        for bot_id in list(self.metrics_history.keys()):
            try:
                metrics_list = self.metrics_history[bot_id]
                if len(metrics_list) > max_metrics_per_bot:
                    # Keep only the most recent metrics
                    self.metrics_history[bot_id] = metrics_list[-max_metrics_per_bot:]
                    self._logger.debug("Cleaned up old metrics for bot", bot_id=bot_id)
            except (KeyError, IndexError) as e:
                await self.error_handler.handle_error(
                    e,
                    context={"operation": "cleanup_metrics", "bot_id": bot_id},
                    severity=ErrorSeverity.LOW.value,
                )
                continue

    async def _export_metrics_to_influxdb(self, bot_id: str, metrics: BotMetrics) -> None:
        """
        Export metrics to InfluxDB if enabled.

        Args:
            bot_id: Bot identifier
            metrics: Bot metrics to export
        """
        try:
            config = self._config_service.get_config() if self._config_service else {}
            if not config.get("monitoring", {}).get("influxdb_enabled", False):
                return

            # This would integrate with actual InfluxDB client
            # For now, just log that we would export
            self._logger.debug("Would export metrics to InfluxDB", bot_id=bot_id)

        except Exception as e:
            self._logger.warning(f"Failed to export metrics to InfluxDB: {e}", bot_id=bot_id)

    @with_error_context(component="BotMonitor", operation="detect_performance_degradation")
    async def _detect_performance_degradation(
        self, bot_id: str, metrics: BotMetrics
    ) -> dict[str, Any]:
        """
        Detect performance degradation compared to baseline.

        Args:
            bot_id: Bot identifier
            metrics: Current metrics

        Returns:
            Dictionary with degradation analysis
        """
        if bot_id not in self.performance_baselines:
            return {"is_degraded": False, "reason": "No baseline established"}

        baseline = self.performance_baselines[bot_id]
        if not baseline.get("baseline_established"):
            return {"is_degraded": False, "reason": "Baseline not established"}

        degraded_metrics = []

        # Check CPU degradation
        if "cpu_usage" in baseline and hasattr(metrics, "cpu_usage"):
            cpu_baseline_data = baseline["cpu_usage"]
            if isinstance(cpu_baseline_data, dict):
                cpu_baseline = cpu_baseline_data.get("mean", 0)
            else:
                cpu_baseline = cpu_baseline_data

            if metrics.cpu_usage > cpu_baseline * 1.5:  # 50% increase
                degraded_metrics.append(
                    {
                        "metric": "cpu_usage",
                        "current": metrics.cpu_usage,
                        "baseline": cpu_baseline,
                        "increase_pct": (
                            (metrics.cpu_usage - cpu_baseline)
                            / max(cpu_baseline, CPU_DEVIATION_THRESHOLD)
                        )
                        * 100,
                    }
                )

        # Check memory degradation
        if "memory_usage" in baseline and hasattr(metrics, "memory_usage"):
            memory_baseline_data = baseline["memory_usage"]
            if isinstance(memory_baseline_data, dict):
                memory_baseline = memory_baseline_data.get("mean", 0)
            else:
                memory_baseline = memory_baseline_data

            if metrics.memory_usage > memory_baseline * 1.5:  # 50% increase
                degraded_metrics.append(
                    {
                        "metric": "memory_usage",
                        "current": metrics.memory_usage,
                        "baseline": memory_baseline,
                        "increase_pct": (
                            (metrics.memory_usage - memory_baseline)
                            / max(memory_baseline, MEMORY_DEVIATION_THRESHOLD)
                        )
                        * 100,
                    }
                )

        return {
            "is_degraded": len(degraded_metrics) > 0,
            "degraded_metrics": degraded_metrics,
            "severity": (
                "high" if len(degraded_metrics) >= 2 else "medium" if degraded_metrics else "none"
            ),
        }

    @with_error_context(component="BotMonitor", operation="get_resource_usage_summary")
    async def get_resource_usage_summary(self, bot_id: str) -> dict[str, Any]:
        """
        Get resource usage summary for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Resource usage summary
        """
        if bot_id not in self.metrics_history or not self.metrics_history[bot_id]:
            return {"error": "No metrics available"}

        recent_metrics = self.metrics_history[bot_id][-10:]  # Last 10 metrics

        # Calculate trends
        cpu_values = [m.cpu_usage for m in recent_metrics if hasattr(m, "cpu_usage")]
        memory_values = [m.memory_usage for m in recent_metrics if hasattr(m, "memory_usage")]

        cpu_trend = "stable"
        memory_trend = "stable"

        if len(cpu_values) >= 3:
            if cpu_values[-1] > cpu_values[0] * 1.2:
                cpu_trend = "increasing"
            elif cpu_values[-1] < cpu_values[0] * 0.8:
                cpu_trend = "decreasing"

        if len(memory_values) >= 3:
            if memory_values[-1] > memory_values[0] * 1.2:
                memory_trend = "increasing"
            elif memory_values[-1] < memory_values[0] * 0.8:
                memory_trend = "decreasing"

        latest = recent_metrics[-1]

        return {
            "cpu_usage": {
                "current": latest.cpu_usage if hasattr(latest, "cpu_usage") else 0,
                "trend": cpu_trend,
                "average_recent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            },
            "memory_usage": {
                "current": latest.memory_usage if hasattr(latest, "memory_usage") else 0,
                "trend": memory_trend,
                "average_recent": (sum(memory_values) / len(memory_values) if memory_values else 0),
            },
            "timestamp": latest.metrics_updated_at.isoformat(),
        }

    async def _generate_predictive_alerts(self, bot_id: str) -> list[dict[str, Any]]:
        """
        Generate predictive alerts based on trends.

        Args:
            bot_id: Bot identifier

        Returns:
            List of predictive alerts
        """
        predictions: list[dict[str, Any]] = []

        try:
            if bot_id not in self.metrics_history or len(self.metrics_history[bot_id]) < 5:
                return predictions

            recent_metrics = self.metrics_history[bot_id][-10:]  # Last 10 data points

            # Check CPU usage trend
            cpu_values = [m.cpu_usage for m in recent_metrics if hasattr(m, "cpu_usage")]
            if len(cpu_values) >= 5:
                # Simple linear trend detection
                trend_slope = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
                if trend_slope > CPU_TREND_WARNING:  # Increasing trend detection
                    predictions.append(
                        {
                            "type": "cpu_trend_alert",
                            "message": f"CPU trending up (slope: {trend_slope:.1f}%/cycle)",
                            "severity": "warning",
                            "projected_threshold_breach": (
                                "within_30_minutes"
                                if trend_slope > CPU_TREND_CRITICAL
                                else "within_2_hours"
                            ),
                        }
                    )

            # Check memory usage trend
            memory_values = [m.memory_usage for m in recent_metrics if hasattr(m, "memory_usage")]
            if len(memory_values) >= 5:
                trend_slope = (memory_values[-1] - memory_values[0]) / len(memory_values)
                if trend_slope > MEMORY_TREND_WARNING:  # Memory increasing trend detection
                    predictions.append(
                        {
                            "type": "memory_trend_alert",
                            "message": f"Memory trending up (slope: {trend_slope:.1f} MB/cycle)",
                            "severity": "warning",
                            "projected_threshold_breach": (
                                "within_1_hour"
                                if trend_slope > MEMORY_TREND_CRITICAL
                                else "within_4_hours"
                            ),
                        }
                    )

        except Exception as e:
            self._logger.warning(f"Failed to generate predictive alerts: {e}", bot_id=bot_id)

        return predictions

    @with_error_context(component="BotMonitor", operation="compare_bot_performance")
    async def compare_bot_performance(self) -> dict[str, Any]:
        """
        Compare performance across all monitored bots.

        Returns:
            Dictionary with bot performance comparison
        """
        if not self.monitored_bots:
            return {"rankings": [], "performance_gaps": []}

        bot_performances = []

        for bot_id in self.monitored_bots.keys():
            if self.metrics_history.get(bot_id):
                latest_metrics = self.metrics_history[bot_id][-1]
                health_status = self.bot_health_status.get(bot_id, {})

                performance_score = health_status.get("health_score", 0.0)

                bot_performances.append(
                    {
                        "bot_id": bot_id,
                        "health_score": performance_score,
                        "cpu_usage": getattr(latest_metrics, "cpu_usage", 0),
                        "memory_usage": getattr(latest_metrics, "memory_usage", 0),
                        "error_count": getattr(latest_metrics, "error_count", 0),
                        "total_trades": getattr(latest_metrics, "total_trades", 0),
                    }
                )

        # Sort by health score (descending)
        bot_performances.sort(key=lambda x: x["health_score"], reverse=True)

        # Identify performance gaps
        performance_gaps = []
        if len(bot_performances) >= 2:
            best_performance = bot_performances[0]["health_score"]
            worst_performance = bot_performances[-1]["health_score"]

            if best_performance - worst_performance > 0.3:  # 30% gap
                performance_gaps.append(
                    {
                        "type": "health_score_gap",
                        "best_bot": bot_performances[0]["bot_id"],
                        "worst_bot": bot_performances[-1]["bot_id"],
                        "gap": best_performance - worst_performance,
                    }
                )

        return {
            "rankings": bot_performances,
            "performance_gaps": performance_gaps,
            "total_bots_compared": len(bot_performances),
            "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
        }
