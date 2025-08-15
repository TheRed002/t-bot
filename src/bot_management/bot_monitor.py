"""
Bot health and performance monitoring system.

This module implements the BotMonitor class that provides comprehensive
monitoring of bot instances, including health checks, performance metrics
collection, alert generation, and error pattern detection.

CRITICAL: This integrates with P-002 (database), P-002A (error handling),
and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil

from src.core.config import Config
from src.core.exceptions import ExecutionError
from src.core.logging import get_logger
from src.core.types import BotMetrics, BotStatus

# MANDATORY: Import from P-002 (database)
from src.database.connection import DatabaseConnectionManager, get_influxdb_client

# MANDATORY: Import from P-002A (error handling)
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls


class BotMonitor:
    """
    Comprehensive bot health and performance monitoring system.
    
    This class provides:
    - Bot status monitoring and health checks
    - Performance metrics collection and analysis
    - Alert generation for anomalies and issues
    - Resource usage tracking per bot
    - Error pattern detection and reporting
    - Historical performance analysis
    """

    def __init__(self, config: Config):
        """
        Initialize bot monitor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.BotMonitor")
        self.error_handler = ErrorHandler(config.error_handling)

        # Database connections for metrics storage
        self.db_connection = DatabaseConnectionManager(config)
        self.influxdb_client = None  # Will be initialized when needed

        # Bot monitoring state
        self.monitored_bots: dict[str, dict[str, Any]] = {}
        self.bot_health_status: dict[str, dict[str, Any]] = {}
        self.performance_baselines: dict[str, dict[str, float]] = {}

        # Monitor state
        self.is_running = False
        self.monitoring_task = None
        self.health_check_task = None
        self.metrics_collection_task = None

        # Alert tracking
        self.active_alerts: dict[str, list[dict[str, Any]]] = {}
        self.alert_history: list[dict[str, Any]] = []

        # Configuration
        self.monitoring_interval = config.bot_management.get("monitoring_interval", 30)
        self.health_check_interval = config.bot_management.get("health_check_interval", 60)
        self.metrics_collection_interval = config.bot_management.get("metrics_collection_interval", 10)
        self.alert_retention_hours = config.bot_management.get("alert_retention_hours", 24)

        # Performance thresholds
        self.performance_thresholds = {
            "cpu_usage_warning": config.bot_management.get("cpu_usage_warning", 70.0),
            "cpu_usage_critical": config.bot_management.get("cpu_usage_critical", 90.0),
            "memory_usage_warning": config.bot_management.get("memory_usage_warning", 500.0),  # MB
            "memory_usage_critical": config.bot_management.get("memory_usage_critical", 1000.0),  # MB
            "error_rate_warning": config.bot_management.get("error_rate_warning", 0.05),  # 5%
            "error_rate_critical": config.bot_management.get("error_rate_critical", 0.15),  # 15%
            "heartbeat_timeout": config.bot_management.get("heartbeat_timeout", 120),  # seconds
            "win_rate_threshold": config.bot_management.get("win_rate_threshold", 0.30)  # 30%
        }

        # Monitoring statistics
        self.monitoring_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "bots_monitored": 0,
            "last_monitoring_time": None
        }

        self.logger.info("Bot monitor initialized")

    @log_calls
    async def start(self) -> None:
        """
        Start the bot monitoring system.
        
        Raises:
            ExecutionError: If startup fails
        """
        try:
            if self.is_running:
                self.logger.warning("Bot monitor is already running")
                return

            self.logger.info("Starting bot monitor")

            # Initialize database connections
            await self.db_connection.initialize()

            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())

            self.is_running = True
            self.logger.info("Bot monitor started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start bot monitor: {e}")
            raise ExecutionError(f"Bot monitor startup failed: {e}")

    @log_calls
    async def stop(self) -> None:
        """
        Stop the bot monitoring system.
        
        Raises:
            ExecutionError: If shutdown fails
        """
        try:
            if not self.is_running:
                self.logger.warning("Bot monitor is not running")
                return

            self.logger.info("Stopping bot monitor")
            self.is_running = False

            # Stop monitoring tasks
            tasks = [self.monitoring_task, self.health_check_task, self.metrics_collection_task]
            for task in tasks:
                if task:
                    task.cancel()

            # Close database connections
            await self.db_connection.close()

            # Clear state
            self.monitored_bots.clear()
            self.bot_health_status.clear()
            self.active_alerts.clear()

            self.logger.info("Bot monitor stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop bot monitor: {e}")
            raise ExecutionError(f"Bot monitor shutdown failed: {e}")

    @log_calls
    async def register_bot(self, bot_id: str) -> None:
        """
        Register a bot for monitoring.
        
        Args:
            bot_id: Bot identifier
            
        Raises:
            ValidationError: If registration is invalid
        """
        try:
            if bot_id in self.monitored_bots:
                self.logger.warning("Bot already registered for monitoring", bot_id=bot_id)
                return

            # Initialize monitoring data
            self.monitored_bots[bot_id] = {
                "registered_at": datetime.now(timezone.utc),
                "last_health_check": None,
                "last_metrics_collection": None,
                "consecutive_failures": 0,
                "total_health_checks": 0,
                "health_check_failures": 0
            }

            # Initialize health status
            self.bot_health_status[bot_id] = {
                "status": "unknown",
                "last_heartbeat": None,
                "health_score": 0.0,
                "issues": [],
                "last_updated": datetime.now(timezone.utc)
            }

            # Initialize performance baseline
            self.performance_baselines[bot_id] = {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "trade_frequency": 0.0,
                "error_rate": 0.0,
                "baseline_established": False
            }

            # Initialize alert tracking
            self.active_alerts[bot_id] = []

            self.monitoring_stats["bots_monitored"] += 1

            self.logger.info("Bot registered for monitoring", bot_id=bot_id)

        except Exception as e:
            self.logger.error(f"Failed to register bot for monitoring: {e}", bot_id=bot_id)
            raise

    @log_calls
    async def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot from monitoring.
        
        Args:
            bot_id: Bot identifier
        """
        try:
            if bot_id not in self.monitored_bots:
                self.logger.warning("Bot not registered for monitoring", bot_id=bot_id)
                return

            # Remove from all tracking
            del self.monitored_bots[bot_id]
            del self.bot_health_status[bot_id]
            del self.performance_baselines[bot_id]
            del self.active_alerts[bot_id]

            self.monitoring_stats["bots_monitored"] -= 1

            self.logger.info("Bot unregistered from monitoring", bot_id=bot_id)

        except Exception as e:
            self.logger.error(f"Failed to unregister bot from monitoring: {e}", bot_id=bot_id)

    @log_calls
    async def update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None:
        """
        Update metrics for a monitored bot.
        
        Args:
            bot_id: Bot identifier
            metrics: Current bot metrics
        """
        try:
            if bot_id not in self.monitored_bots:
                await self.register_bot(bot_id)

            current_time = datetime.now(timezone.utc)

            # Update monitoring tracking
            self.monitored_bots[bot_id]["last_metrics_collection"] = current_time

            # Update health status
            await self._update_bot_health_status(bot_id, metrics)

            # Store metrics in InfluxDB
            await self._store_metrics(bot_id, metrics)

            # Check for performance anomalies
            await self._check_performance_anomalies(bot_id, metrics)

            # Update performance baseline
            await self._update_performance_baseline(bot_id, metrics)

        except Exception as e:
            self.logger.error(f"Failed to update bot metrics: {e}", bot_id=bot_id)

    @log_calls
    async def check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]:
        """
        Perform comprehensive health check for a bot.
        
        Args:
            bot_id: Bot identifier
            bot_status: Current bot status
            
        Returns:
            dict: Health check results
        """
        try:
            if bot_id not in self.monitored_bots:
                return {"error": "Bot not registered for monitoring"}

            current_time = datetime.now(timezone.utc)
            monitoring_data = self.monitored_bots[bot_id]

            # Update monitoring statistics
            monitoring_data["total_health_checks"] += 1
            monitoring_data["last_health_check"] = current_time
            self.monitoring_stats["total_checks"] += 1

            health_results = {
                "bot_id": bot_id,
                "timestamp": current_time.isoformat(),
                "overall_health": "healthy",
                "health_score": 1.0,
                "checks": {},
                "issues": [],
                "recommendations": []
            }

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

            # Calculate overall health score
            check_scores = [
                status_healthy.get("score", 0.0),
                heartbeat_healthy.get("score", 0.0),
                resource_healthy.get("score", 0.0),
                performance_healthy.get("score", 0.0),
                error_rate_healthy.get("score", 0.0)
            ]

            health_results["health_score"] = sum(check_scores) / len(check_scores)

            # Determine overall health
            if health_results["health_score"] >= 0.8:
                health_results["overall_health"] = "healthy"
            elif health_results["health_score"] >= 0.6:
                health_results["overall_health"] = "warning"
            else:
                health_results["overall_health"] = "critical"

            # Collect issues and recommendations
            for check_name, check_result in health_results["checks"].items():
                if check_result.get("issues"):
                    health_results["issues"].extend(check_result["issues"])
                if check_result.get("recommendations"):
                    health_results["recommendations"].extend(check_result["recommendations"])

            # Update bot health status
            self.bot_health_status[bot_id].update({
                "status": health_results["overall_health"],
                "health_score": health_results["health_score"],
                "issues": health_results["issues"],
                "last_updated": current_time
            })

            # Generate alerts if needed
            await self._generate_health_alerts(bot_id, health_results)

            return health_results

        except Exception as e:
            self.logger.error(f"Bot health check failed: {e}", bot_id=bot_id)
            monitoring_data["health_check_failures"] += 1
            monitoring_data["consecutive_failures"] += 1

            return {
                "bot_id": bot_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_health": "error",
                "health_score": 0.0,
                "error": str(e)
            }

    async def get_monitoring_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary for all bots."""
        try:
            current_time = datetime.now(timezone.utc)

            # Aggregate health statistics
            health_summary = {
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "error": 0,
                "unknown": 0
            }

            # Aggregate performance statistics
            performance_summary = {
                "total_cpu_usage": 0.0,
                "total_memory_usage": 0.0,
                "average_health_score": 0.0,
                "bots_with_issues": 0
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
                    "issues_count": len(health_status["issues"])
                }

            # Calculate averages
            bot_count = len(self.bot_health_status)
            if bot_count > 0:
                performance_summary["average_health_score"] = total_health_score / bot_count

            # Count active alerts
            total_active_alerts = sum(len(alerts) for alerts in self.active_alerts.values())
            critical_alerts = sum(
                1 for alerts in self.active_alerts.values()
                for alert in alerts
                if alert.get("severity") == "critical"
            )

            return {
                "monitoring_overview": {
                    "is_running": self.is_running,
                    "bots_monitored": self.monitoring_stats["bots_monitored"],
                    "total_checks_performed": self.monitoring_stats["total_checks"],
                    "active_alerts": total_active_alerts,
                    "critical_alerts": critical_alerts,
                    "last_monitoring_cycle": self.monitoring_stats["last_monitoring_time"].isoformat() if self.monitoring_stats["last_monitoring_time"] else None
                },
                "health_summary": health_summary,
                "performance_summary": performance_summary,
                "bot_summaries": bot_summaries,
                "alert_statistics": {
                    "total_alerts_generated": self.monitoring_stats["alerts_generated"],
                    "critical_alerts_total": self.monitoring_stats["critical_alerts"],
                    "warning_alerts_total": self.monitoring_stats["warning_alerts"]
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate monitoring summary: {e}")
            return {"error": str(e)}

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
                "last_heartbeat": health_status["last_heartbeat"].isoformat() if health_status["last_heartbeat"] else None,
                "last_updated": health_status["last_updated"].isoformat(),
                "issues": health_status["issues"]
            },
            "monitoring_statistics": {
                "registered_at": monitoring_data.get("registered_at").isoformat() if monitoring_data.get("registered_at") else None,
                "total_health_checks": monitoring_data.get("total_health_checks", 0),
                "health_check_failures": monitoring_data.get("health_check_failures", 0),
                "consecutive_failures": monitoring_data.get("consecutive_failures", 0),
                "last_health_check": monitoring_data.get("last_health_check").isoformat() if monitoring_data.get("last_health_check") else None
            },
            "performance_baseline": baseline_data,
            "active_alerts": active_alerts,
            "alert_history_24h": [
                alert for alert in self.alert_history
                if (alert["bot_id"] == bot_id and
                    (datetime.now(timezone.utc) - datetime.fromisoformat(alert["timestamp"])).total_seconds() < 86400)
            ]
        }

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_running:
                try:
                    self.monitoring_stats["last_monitoring_time"] = datetime.now(timezone.utc)

                    # Clean up old alerts
                    await self._cleanup_old_alerts()

                    # Process alert escalations
                    await self._process_alert_escalations()

                    # Update monitoring statistics
                    await self._update_monitoring_statistics()

                    # Wait for next cycle
                    await asyncio.sleep(self.monitoring_interval)

                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")

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
                        except Exception as e:
                            self.logger.warning(f"Health check failed for bot: {e}", bot_id=bot_id)

                    # Wait for next cycle
                    await asyncio.sleep(self.health_check_interval)

                except Exception as e:
                    self.logger.error(f"Health check loop error: {e}")
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            self.logger.info("Health check loop cancelled")

    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        try:
            while self.is_running:
                try:
                    # Collect system-wide metrics
                    await self._collect_system_metrics()

                    # Update performance baselines
                    await self._update_all_baselines()

                    # Wait for next cycle
                    await asyncio.sleep(self.metrics_collection_interval)

                except Exception as e:
                    self.logger.error(f"Metrics collection loop error: {e}")
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            self.logger.info("Metrics collection loop cancelled")

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
        issues = []

        # CPU usage check
        if metrics.cpu_usage > self.performance_thresholds["cpu_usage_critical"]:
            issues.append(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > self.performance_thresholds["cpu_usage_warning"]:
            issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")

        # Memory usage check
        if metrics.memory_usage > self.performance_thresholds["memory_usage_critical"]:
            issues.append(f"Critical memory usage: {metrics.memory_usage:.1f} MB")
        elif metrics.memory_usage > self.performance_thresholds["memory_usage_warning"]:
            issues.append(f"High memory usage: {metrics.memory_usage:.1f} MB")

        # Error rate check
        if metrics.total_trades > 0:
            error_rate = metrics.error_count / metrics.total_trades
            if error_rate > self.performance_thresholds["error_rate_critical"]:
                issues.append(f"Critical error rate: {error_rate:.1%}")
            elif error_rate > self.performance_thresholds["error_rate_warning"]:
                issues.append(f"High error rate: {error_rate:.1%}")

        # Win rate check
        if metrics.win_rate < self.performance_thresholds["win_rate_threshold"]:
            issues.append(f"Low win rate: {metrics.win_rate:.1%}")

        health_status["issues"] = issues
        health_status["last_updated"] = current_time

    async def _store_metrics(self, bot_id: str, metrics: BotMetrics) -> None:
        """Store metrics in InfluxDB."""
        try:
            # Prepare metrics data for InfluxDB
            metrics_data = {
                "measurement": "bot_metrics",
                "tags": {"bot_id": bot_id},
                "fields": {
                    "total_trades": metrics.total_trades,
                    "profitable_trades": metrics.profitable_trades,
                    "losing_trades": metrics.losing_trades,
                    "total_pnl": float(metrics.total_pnl),
                    "unrealized_pnl": float(metrics.unrealized_pnl),
                    "win_rate": metrics.win_rate,
                    "average_trade_pnl": float(metrics.average_trade_pnl),
                    "max_drawdown": float(metrics.max_drawdown),
                    "uptime_percentage": metrics.uptime_percentage,
                    "error_count": metrics.error_count,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "api_calls_count": metrics.api_calls_count
                },
                "timestamp": datetime.now(timezone.utc)
            }

            if self.config.monitoring.get("influxdb_enabled", False):
                try:
                    influxdb_client = get_influxdb_client()
                    # Write metrics to InfluxDB (implementation depends on client interface)
                    # This would need to be implemented based on actual InfluxDB client API
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to write metrics to InfluxDB: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to store metrics: {e}", bot_id=bot_id)

    async def _check_performance_anomalies(self, bot_id: str, metrics: BotMetrics) -> None:
        """Check for performance anomalies against baseline."""
        if bot_id not in self.performance_baselines:
            return

        baseline = self.performance_baselines[bot_id]

        if not baseline.get("baseline_established"):
            return

        # Check for significant deviations
        anomalies = []

        # CPU usage anomaly
        cpu_deviation = abs(metrics.cpu_usage - baseline["cpu_usage"]) / max(baseline["cpu_usage"], 1.0)
        if cpu_deviation > 0.5:  # 50% deviation
            anomalies.append({
                "type": "cpu_anomaly",
                "current": metrics.cpu_usage,
                "baseline": baseline["cpu_usage"],
                "deviation": cpu_deviation
            })

        # Memory usage anomaly
        memory_deviation = abs(metrics.memory_usage - baseline["memory_usage"]) / max(baseline["memory_usage"], 1.0)
        if memory_deviation > 0.5:  # 50% deviation
            anomalies.append({
                "type": "memory_anomaly",
                "current": metrics.memory_usage,
                "baseline": baseline["memory_usage"],
                "deviation": memory_deviation
            })

        # Error rate anomaly
        current_error_rate = metrics.error_count / max(metrics.total_trades, 1)
        error_rate_deviation = abs(current_error_rate - baseline["error_rate"])
        if error_rate_deviation > 0.1:  # 10% absolute deviation
            anomalies.append({
                "type": "error_rate_anomaly",
                "current": current_error_rate,
                "baseline": baseline["error_rate"],
                "deviation": error_rate_deviation
            })

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
            baseline.update({
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "error_rate": metrics.error_count / max(metrics.total_trades, 1),
                "baseline_established": True
            })
        else:
            # Update with exponential moving average
            baseline["cpu_usage"] = (1 - alpha) * baseline["cpu_usage"] + alpha * metrics.cpu_usage
            baseline["memory_usage"] = (1 - alpha) * baseline["memory_usage"] + alpha * metrics.memory_usage

            current_error_rate = metrics.error_count / max(metrics.total_trades, 1)
            baseline["error_rate"] = (1 - alpha) * baseline["error_rate"] + alpha * current_error_rate

    async def _check_bot_status(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]:
        """Check bot status health."""
        score = 1.0
        issues = []
        recommendations = []

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
        elif bot_status == BotStatus.STARTING:
            score = 0.7
            issues.append("Bot is still starting")
        elif bot_status == BotStatus.STOPPING:
            score = 0.3
            issues.append("Bot is stopping")

        return {
            "score": score,
            "status": bot_status.value,
            "issues": issues,
            "recommendations": recommendations
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
                "recommendations": ["Check bot connectivity and health monitoring"]
            }

        current_time = datetime.now(timezone.utc)
        heartbeat_age = (current_time - last_heartbeat).total_seconds()

        if heartbeat_age > self.performance_thresholds["heartbeat_timeout"]:
            return {
                "score": 0.0,
                "issues": [f"Heartbeat timeout: {heartbeat_age:.0f}s ago"],
                "recommendations": ["Check bot connectivity and restart if necessary"]
            }
        elif heartbeat_age > self.performance_thresholds["heartbeat_timeout"] / 2:
            return {
                "score": 0.5,
                "issues": [f"Stale heartbeat: {heartbeat_age:.0f}s ago"],
                "recommendations": ["Monitor bot connectivity"]
            }
        else:
            return {"score": 1.0, "last_heartbeat_age": heartbeat_age}

    async def _check_resource_usage(self, bot_id: str) -> dict[str, Any]:
        """Check resource usage health."""
        # In a real implementation, this would get actual resource usage
        # For now, simulate based on system resources
        try:
            process = psutil.Process()
            cpu_usage = process.cpu_percent()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            score = 1.0
            issues = []
            recommendations = []

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
                "recommendations": recommendations
            }

        except Exception as e:
            return {
                "score": 0.5,
                "issues": [f"Resource check failed: {e}"],
                "recommendations": ["Manual resource monitoring required"]
            }

    async def _check_performance_health(self, bot_id: str) -> dict[str, Any]:
        """Check trading performance health."""
        # This would integrate with actual bot metrics
        # For now, return baseline health check
        return {
            "score": 0.8,
            "issues": [],
            "recommendations": []
        }

    async def _check_error_rate(self, bot_id: str) -> dict[str, Any]:
        """Check error rate health."""
        # This would integrate with actual error tracking
        # For now, return baseline health check
        return {
            "score": 0.9,
            "issues": [],
            "recommendations": []
        }

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
                "acknowledged": False
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

            self.logger.warning(
                "Health alert generated for bot",
                bot_id=bot_id,
                severity=severity,
                health_score=health_results["health_score"],
                issues_count=len(health_results["issues"])
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
                "acknowledged": False
            }

            self.active_alerts[bot_id].append(alert)
            self.alert_history.append(alert)
            self.monitoring_stats["alerts_generated"] += 1
            self.monitoring_stats["warning_alerts"] += 1

            self.logger.warning(
                "Performance anomaly alert generated",
                bot_id=bot_id,
                anomaly_type=anomaly["type"],
                deviation=anomaly["deviation"]
            )

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.alert_retention_hours)

        # Clean up alert history
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        # Clean up active alerts for acknowledged ones older than retention period
        for bot_id in self.active_alerts:
            self.active_alerts[bot_id] = [
                alert for alert in self.active_alerts[bot_id]
                if (not alert.get("acknowledged") or
                    datetime.fromisoformat(alert["timestamp"]) > cutoff_time)
            ]

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations for unacknowledged critical alerts."""
        # Implementation would handle alert escalation logic
        # For now, just log critical alerts that need attention

        critical_alerts = []
        for bot_id, alerts in self.active_alerts.items():
            for alert in alerts:
                if (alert["severity"] == "critical" and
                    not alert.get("acknowledged")):
                    critical_alerts.append((bot_id, alert))

        if critical_alerts:
            self.logger.critical(
                "Unacknowledged critical alerts requiring attention",
                count=len(critical_alerts)
            )

    async def _update_monitoring_statistics(self) -> None:
        """Update monitoring statistics."""
        # Update bot count
        self.monitoring_stats["bots_monitored"] = len(self.monitored_bots)

    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics."""
        try:
            # Collect system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Store system metrics
            system_metrics = {
                "measurement": "system_metrics",
                "tags": {"component": "bot_monitor"},
                "fields": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "disk_usage_percent": disk.percent,
                    "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                    "monitored_bots": len(self.monitored_bots),
                    "active_alerts": sum(len(alerts) for alerts in self.active_alerts.values())
                },
                "timestamp": datetime.now(timezone.utc)
            }

            if self.config.monitoring.get("influxdb_enabled", False):
                try:
                    influxdb_client = get_influxdb_client()
                    # Write system metrics to InfluxDB
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to write system metrics to InfluxDB: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")

    async def _update_all_baselines(self) -> None:
        """Update performance baselines for all bots."""
        # This would be called periodically to refresh baselines
        # Implementation would recalculate baselines based on recent performance
        pass
