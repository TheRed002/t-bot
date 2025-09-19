"""
Bot Monitoring Service Implementation.

This service handles bot health monitoring, metrics collection,
and alert generation following the service layer pattern.
"""

from typing import Any, TYPE_CHECKING

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotMetrics

from .interfaces import IBotMonitoringService

if TYPE_CHECKING:
    from .instance_service import BotInstanceService


class BotMonitoringService(BaseService, IBotMonitoringService):
    """
    Service for monitoring bot health and performance.

    This service provides comprehensive monitoring capabilities including
    health checks, metrics collection, and alert generation.
    """

    def __init__(
        self,
        bot_instance_service: "BotInstanceService" = None,
        name: str = "BotMonitoringService",
        config: dict[str, Any] = None,
    ):
        """Initialize bot monitoring service."""
        super().__init__(name=name, config=config)
        self._bot_instance_service = bot_instance_service
        self._logger = get_logger(__name__)
        self._health_data: dict[str, dict[str, Any]] = {}
        self._metrics_cache: dict[str, BotMetrics] = {}
        self._alert_conditions: list[dict[str, Any]] = []

    async def get_bot_health(self, bot_id: str) -> dict[str, Any]:
        """
        Get bot health status.

        Args:
            bot_id: Bot ID

        Returns:
            Health status information
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Get cached health data or perform health check
            health_data = self._health_data.get(bot_id)

            if not health_data:
                health_data = await self._perform_health_check(bot_id)
                self._health_data[bot_id] = health_data

            return health_data

        except Exception as e:
            self._logger.error(f"Failed to get bot health for {bot_id}: {e}")
            raise ServiceError(f"Failed to get bot health: {e}") from e

    async def get_bot_metrics(self, bot_id: str) -> BotMetrics:
        """
        Get bot metrics.

        Args:
            bot_id: Bot ID

        Returns:
            Bot metrics
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Check cache first
            if bot_id in self._metrics_cache:
                return self._metrics_cache[bot_id]

            # Get fresh metrics
            metrics = await self._collect_bot_metrics(bot_id)
            self._metrics_cache[bot_id] = metrics

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to get bot metrics for {bot_id}: {e}")
            raise ServiceError(f"Failed to get bot metrics: {e}") from e

    async def get_system_health(self) -> dict[str, Any]:
        """
        Get overall system health.

        Returns:
            System health information
        """
        try:
            # Get active bot IDs
            active_bot_ids = []
            if self._bot_instance_service:
                active_bot_ids = self._bot_instance_service.get_active_bot_ids()

            # Collect health data for all bots
            bot_health_summary = {}
            healthy_bots = 0
            total_bots = len(active_bot_ids)

            for bot_id in active_bot_ids:
                try:
                    health = await self.get_bot_health(bot_id)
                    bot_health_summary[bot_id] = health
                    if health.get("status") == "healthy":
                        healthy_bots += 1
                except Exception as e:
                    self._logger.warning(f"Failed to get health for bot {bot_id}: {e}")
                    bot_health_summary[bot_id] = {"status": "error", "error": str(e)}

            # Calculate system health metrics
            system_health = {
                "overall_status": "healthy" if healthy_bots == total_bots else "degraded",
                "total_bots": total_bots,
                "healthy_bots": healthy_bots,
                "unhealthy_bots": total_bots - healthy_bots,
                "health_percentage": (healthy_bots / total_bots * 100) if total_bots > 0 else 100,
                "bot_health_summary": bot_health_summary,
                "timestamp": self._get_current_timestamp(),
            }

            # Check for system-wide issues
            system_health["issues"] = await self._check_system_issues()

            return system_health

        except Exception as e:
            self._logger.error(f"Failed to get system health: {e}")
            raise ServiceError(f"Failed to get system health: {e}") from e

    async def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Performance summary information
        """
        try:
            # Get active bot IDs
            active_bot_ids = []
            if self._bot_instance_service:
                active_bot_ids = self._bot_instance_service.get_active_bot_ids()

            # Collect performance data
            performance_data = {}
            total_pnl = 0.0
            total_trades = 0
            winning_trades = 0

            for bot_id in active_bot_ids:
                try:
                    metrics = await self.get_bot_metrics(bot_id)
                    bot_performance = {
                        "pnl": float(metrics.total_pnl) if hasattr(metrics, 'total_pnl') else 0.0,
                        "trades": int(metrics.total_trades) if hasattr(metrics, 'total_trades') else 0,
                        "win_rate": float(metrics.win_rate) if hasattr(metrics, 'win_rate') else 0.0,
                    }
                    performance_data[bot_id] = bot_performance

                    total_pnl += bot_performance["pnl"]
                    total_trades += bot_performance["trades"]
                    if bot_performance["win_rate"] > 0.5:
                        winning_trades += bot_performance["trades"]

                except Exception as e:
                    self._logger.warning(f"Failed to get performance for bot {bot_id}: {e}")
                    performance_data[bot_id] = {"error": str(e)}

            # Calculate summary metrics
            average_win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

            performance_summary = {
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "average_win_rate": average_win_rate,
                "bot_performance": performance_data,
                "timestamp": self._get_current_timestamp(),
            }

            return performance_summary

        except Exception as e:
            self._logger.error(f"Failed to get performance summary: {e}")
            raise ServiceError(f"Failed to get performance summary: {e}") from e

    async def check_alert_conditions(self) -> list[dict[str, Any]]:
        """
        Check for alert conditions.

        Returns:
            List of active alerts
        """
        try:
            alerts = []

            # Check bot-specific alerts
            if self._bot_instance_service:
                active_bot_ids = self._bot_instance_service.get_active_bot_ids()

                for bot_id in active_bot_ids:
                    bot_alerts = await self._check_bot_alerts(bot_id)
                    alerts.extend(bot_alerts)

            # Check system-wide alerts
            system_alerts = await self._check_system_alerts()
            alerts.extend(system_alerts)

            return alerts

        except Exception as e:
            self._logger.error(f"Failed to check alert conditions: {e}")
            raise ServiceError(f"Failed to check alert conditions: {e}") from e

    async def _perform_health_check(self, bot_id: str) -> dict[str, Any]:
        """Perform health check for a bot."""
        try:
            # Basic health check implementation
            health_data = {
                "bot_id": bot_id,
                "status": "healthy",
                "checks": {
                    "connectivity": True,
                    "resource_usage": True,
                    "error_rate": True,
                },
                "timestamp": self._get_current_timestamp(),
            }

            # Perform actual checks
            if self._bot_instance_service:
                try:
                    # Check if bot exists and is responsive
                    bot_state = await self._bot_instance_service.get_bot_state(bot_id)
                    health_data["bot_state"] = bot_state.value if hasattr(bot_state, 'value') else str(bot_state)
                except Exception as e:
                    health_data["status"] = "unhealthy"
                    health_data["checks"]["connectivity"] = False
                    health_data["error"] = str(e)

            return health_data

        except Exception as e:
            return {
                "bot_id": bot_id,
                "status": "error",
                "error": str(e),
                "timestamp": self._get_current_timestamp(),
            }

    async def _collect_bot_metrics(self, bot_id: str) -> BotMetrics:
        """Collect metrics for a bot."""
        try:
            # Create default metrics if bot not found
            from decimal import Decimal

            metrics = BotMetrics(
                bot_id=bot_id,
                timestamp=self._get_current_timestamp(),
                total_pnl=Decimal("0"),
                total_trades=0,
                win_rate=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_count=0,
            )

            # Collect real metrics if bot instance service available
            if self._bot_instance_service:
                # This would be implemented to get real metrics from the bot
                pass

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to collect metrics for bot {bot_id}: {e}")
            # Return default metrics on error
            from decimal import Decimal
            return BotMetrics(
                bot_id=bot_id,
                timestamp=self._get_current_timestamp(),
                total_pnl=Decimal("0"),
                total_trades=0,
                win_rate=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_count=1,  # Mark error in metrics
            )

    async def _check_bot_alerts(self, bot_id: str) -> list[dict[str, Any]]:
        """Check alerts for a specific bot."""
        alerts = []

        try:
            # Get bot health and metrics
            health = await self.get_bot_health(bot_id)
            metrics = await self.get_bot_metrics(bot_id)

            # Check for various alert conditions
            if health.get("status") != "healthy":
                alerts.append({
                    "type": "bot_health",
                    "severity": "high",
                    "bot_id": bot_id,
                    "message": f"Bot {bot_id} is unhealthy",
                    "details": health,
                })

            # Check error rate
            if hasattr(metrics, 'error_count') and metrics.error_count > 10:
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "medium",
                    "bot_id": bot_id,
                    "message": f"Bot {bot_id} has high error rate",
                    "error_count": metrics.error_count,
                })

            # Check resource usage
            if hasattr(metrics, 'cpu_usage') and metrics.cpu_usage > 90:
                alerts.append({
                    "type": "high_cpu_usage",
                    "severity": "medium",
                    "bot_id": bot_id,
                    "message": f"Bot {bot_id} has high CPU usage",
                    "cpu_usage": metrics.cpu_usage,
                })

        except Exception as e:
            self._logger.error(f"Failed to check alerts for bot {bot_id}: {e}")

        return alerts

    async def _check_system_alerts(self) -> list[dict[str, Any]]:
        """Check system-wide alerts."""
        alerts = []

        try:
            # Check system health
            system_health = await self.get_system_health()

            if system_health.get("health_percentage", 100) < 80:
                alerts.append({
                    "type": "system_health",
                    "severity": "high",
                    "message": "System health is degraded",
                    "health_percentage": system_health.get("health_percentage"),
                })

        except Exception as e:
            self._logger.error(f"Failed to check system alerts: {e}")

        return alerts

    async def _check_system_issues(self) -> list[str]:
        """Check for system-wide issues."""
        issues = []

        try:
            # Placeholder for system issue detection
            pass

        except Exception as e:
            self._logger.error(f"Failed to check system issues: {e}")
            issues.append(f"Error checking system issues: {e}")

        return issues

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)