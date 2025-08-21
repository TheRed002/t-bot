"""
Cache monitoring and health management for the T-Bot trading system.

This module provides comprehensive monitoring capabilities for the Redis caching
infrastructure, including health checks, performance metrics, alerting, and
cache optimization recommendations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.base import BaseComponent

from .cache_manager import get_cache_manager
from .cache_metrics import get_cache_metrics


class CacheHealthStatus(Enum):
    """Cache health status enumeration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class CacheAlert:
    """Cache alert definition."""

    alert_id: str
    alert_type: str
    severity: str
    message: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    namespace: str = ""
    acknowledged: bool = False


@dataclass
class CacheHealthReport:
    """Comprehensive cache health report."""

    overall_status: CacheHealthStatus = CacheHealthStatus.UNKNOWN
    connectivity_status: CacheHealthStatus = CacheHealthStatus.UNKNOWN
    performance_status: CacheHealthStatus = CacheHealthStatus.UNKNOWN
    memory_status: CacheHealthStatus = CacheHealthStatus.UNKNOWN

    # Performance metrics
    response_time_avg: float = 0.0
    hit_rate_overall: float = 0.0
    miss_rate_overall: float = 0.0
    operations_per_second: float = 0.0

    # Resource usage
    memory_used: str = "0B"
    memory_usage_percent: float = 0.0
    connected_clients: int = 0
    total_operations: int = 0

    # Health indicators
    alerts: list[CacheAlert] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Namespace-specific metrics
    namespace_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.utcnow)


class CacheMonitor(BaseComponent):
    """
    Comprehensive cache monitoring and health management.

    Features:
    - Real-time health monitoring
    - Performance metrics tracking
    - Alerting and notifications
    - Optimization recommendations
    - Historical trend analysis
    - Proactive issue detection
    """

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config

        # Get cache manager and metrics instances
        self.cache_manager = get_cache_manager(config=config)
        self.cache_metrics = get_cache_metrics()

        # Alert thresholds (configurable via config)
        self.alert_thresholds = {
            "hit_rate_low": 0.80,  # Alert if hit rate < 80%
            "response_time_high": 0.100,  # Alert if avg response > 100ms
            "memory_usage_high": 0.90,  # Alert if memory usage > 90%
            "error_rate_high": 0.05,  # Alert if error rate > 5%
            "operations_low": 10,  # Alert if ops/sec < 10
        }

        # Active alerts
        self._active_alerts: dict[str, CacheAlert] = {}
        self._alert_history: list[CacheAlert] = []

        # Monitoring state
        self._monitoring_active = False
        self._last_health_check = None
        self._health_check_interval = 30  # seconds

        # Historical data for trend analysis
        self._performance_history: list[dict[str, Any]] = []
        self._max_history_points = 1440  # 24 hours at 1-minute intervals

    async def start_monitoring(self) -> None:
        """Start continuous cache monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        self.logger.info("Cache monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop cache monitoring."""
        self._monitoring_active = False
        self.logger.info("Cache monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self._health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in cache monitoring loop: {e}")
                await asyncio.sleep(10)  # Back off on error

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            health_data = await self.cache_manager.health_check()

            # Update performance history
            self._update_performance_history(health_data)

            # Check for alerts
            await self._check_alerts(health_data)

            self._last_health_check = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _update_performance_history(self, health_data: dict[str, Any]) -> None:
        """Update performance history for trend analysis."""
        history_point = {
            "timestamp": datetime.utcnow(),
            "ping_time": health_data.get("ping_time", 0),
            "memory_used": health_data.get("used_memory_human", "0B"),
            "connected_clients": health_data.get("connected_clients", 0),
            "keyspace_hits": health_data.get("keyspace_hits", 0),
            "keyspace_misses": health_data.get("keyspace_misses", 0),
        }

        self._performance_history.append(history_point)

        # Limit history size
        if len(self._performance_history) > self._max_history_points:
            self._performance_history.pop(0)

    async def _check_alerts(self, health_data: dict[str, Any]) -> None:
        """Check for alert conditions."""
        cache_stats = health_data.get("cache_stats", {})

        # Check hit rate
        total_stats = cache_stats.get("total", {})
        hit_rate = total_stats.get("hit_rate", 1.0)
        if hit_rate < self.alert_thresholds["hit_rate_low"]:
            await self._create_alert(
                "hit_rate_low",
                "warning",
                f"Cache hit rate is low: {hit_rate:.2%}",
                self.alert_thresholds["hit_rate_low"],
                hit_rate,
            )

        # Check response time
        avg_response_time = total_stats.get("avg_response_time", 0)
        if avg_response_time > self.alert_thresholds["response_time_high"]:
            await self._create_alert(
                "response_time_high",
                "warning",
                f"Cache response time is high: {avg_response_time:.3f}s",
                self.alert_thresholds["response_time_high"],
                avg_response_time,
            )

        # Check error rate
        error_rate = total_stats.get("error_rate", 0)
        if error_rate > self.alert_thresholds["error_rate_high"]:
            await self._create_alert(
                "error_rate_high",
                "critical",
                f"Cache error rate is high: {error_rate:.2%}",
                self.alert_thresholds["error_rate_high"],
                error_rate,
            )

    async def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        threshold: float,
        current_value: float,
        namespace: str = "",
    ) -> None:
        """Create and track an alert."""
        alert_id = f"{alert_type}_{namespace}_{int(time.time())}"

        # Don't create duplicate alerts
        if alert_id in self._active_alerts:
            return

        alert = CacheAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold_value=threshold,
            current_value=current_value,
            timestamp=datetime.utcnow(),
            namespace=namespace,
        )

        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)

        self.logger.warning(f"Cache alert created: {message}", alert_id=alert_id)

    async def get_health_report(self) -> CacheHealthReport:
        """Generate comprehensive health report."""
        try:
            # Get current health data
            health_data = await self.cache_manager.health_check()

            # Get cache metrics
            cache_stats = self.cache_metrics.get_stats()
            total_stats = cache_stats.get("total", {})

            # Calculate overall health status
            overall_status = self._calculate_overall_health(health_data, total_stats)

            # Generate recommendations
            recommendations = self._generate_recommendations(health_data, total_stats)

            # Create namespace-specific metrics
            namespace_metrics = {}
            for namespace, stats in cache_stats.items():
                if namespace != "total":
                    namespace_metrics[namespace] = {
                        "hit_rate": stats.get("hit_rate", 0),
                        "miss_rate": stats.get("miss_rate", 0),
                        "operations": stats.get("hits", 0)
                        + stats.get("misses", 0)
                        + stats.get("sets", 0),
                        "errors": stats.get("errors", 0),
                        "avg_response_time": stats.get("avg_response_time", 0),
                    }

            return CacheHealthReport(
                overall_status=overall_status,
                connectivity_status=(
                    CacheHealthStatus.HEALTHY
                    if health_data.get("status") == "healthy"
                    else CacheHealthStatus.CRITICAL
                ),
                performance_status=self._assess_performance_status(total_stats),
                memory_status=self._assess_memory_status(health_data),
                response_time_avg=total_stats.get("avg_response_time", 0),
                hit_rate_overall=total_stats.get("hit_rate", 0),
                miss_rate_overall=total_stats.get("miss_rate", 0),
                operations_per_second=self._calculate_ops_per_second(),
                memory_used=health_data.get("used_memory_human", "0B"),
                memory_usage_percent=self._calculate_memory_usage_percent(health_data),
                connected_clients=health_data.get("connected_clients", 0),
                total_operations=total_stats.get("hits", 0)
                + total_stats.get("misses", 0)
                + total_stats.get("sets", 0),
                alerts=list(self._active_alerts.values()),
                recommendations=recommendations,
                namespace_metrics=namespace_metrics,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate health report: {e}")
            return CacheHealthReport(overall_status=CacheHealthStatus.UNKNOWN)

    def _calculate_overall_health(
        self, health_data: dict[str, Any], cache_stats: dict[str, Any]
    ) -> CacheHealthStatus:
        """Calculate overall health status."""
        if health_data.get("status") != "healthy":
            return CacheHealthStatus.CRITICAL

        # Check critical metrics
        hit_rate = cache_stats.get("hit_rate", 1.0)
        error_rate = cache_stats.get("error_rate", 0)
        avg_response_time = cache_stats.get("avg_response_time", 0)

        if error_rate > self.alert_thresholds["error_rate_high"]:
            return CacheHealthStatus.CRITICAL

        if (
            hit_rate < self.alert_thresholds["hit_rate_low"]
            or avg_response_time > self.alert_thresholds["response_time_high"]
        ):
            return CacheHealthStatus.WARNING

        return CacheHealthStatus.HEALTHY

    def _assess_performance_status(self, cache_stats: dict[str, Any]) -> CacheHealthStatus:
        """Assess performance status."""
        hit_rate = cache_stats.get("hit_rate", 1.0)
        avg_response_time = cache_stats.get("avg_response_time", 0)

        if hit_rate < 0.60 or avg_response_time > 0.200:  # Very poor performance
            return CacheHealthStatus.CRITICAL
        elif (
            hit_rate < self.alert_thresholds["hit_rate_low"]
            or avg_response_time > self.alert_thresholds["response_time_high"]
        ):
            return CacheHealthStatus.WARNING
        else:
            return CacheHealthStatus.HEALTHY

    def _assess_memory_status(self, health_data: dict[str, Any]) -> CacheHealthStatus:
        """Assess memory usage status."""
        memory_usage_percent = self._calculate_memory_usage_percent(health_data)

        if memory_usage_percent > 95:
            return CacheHealthStatus.CRITICAL
        elif memory_usage_percent > self.alert_thresholds["memory_usage_high"] * 100:
            return CacheHealthStatus.WARNING
        else:
            return CacheHealthStatus.HEALTHY

    def _calculate_memory_usage_percent(self, health_data: dict[str, Any]) -> float:
        """Calculate memory usage percentage (simplified)."""
        # This would need Redis maxmemory configuration to be accurate
        # For now, return a reasonable estimate based on used memory
        used_memory_str = health_data.get("used_memory_human", "0B")

        # Parse memory string (e.g., "1.23M" -> MB)
        if "G" in used_memory_str:
            return 70.0  # Assume higher usage for GB range
        elif "M" in used_memory_str:
            mb_value = float(used_memory_str.replace("M", ""))
            return min(mb_value / 10.0, 100.0)  # Rough estimate
        else:
            return 10.0  # Low usage for KB range

    def _calculate_ops_per_second(self) -> float:
        """Calculate operations per second from recent history."""
        if len(self._performance_history) < 2:
            return 0.0

        recent = self._performance_history[-1]
        previous = self._performance_history[-2]

        time_diff = (recent["timestamp"] - previous["timestamp"]).total_seconds()
        if time_diff <= 0:
            return 0.0

        hits_diff = recent.get("keyspace_hits", 0) - previous.get("keyspace_hits", 0)
        misses_diff = recent.get("keyspace_misses", 0) - previous.get("keyspace_misses", 0)

        return (hits_diff + misses_diff) / time_diff

    def _generate_recommendations(
        self, health_data: dict[str, Any], cache_stats: dict[str, Any]
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        hit_rate = cache_stats.get("hit_rate", 1.0)
        avg_response_time = cache_stats.get("avg_response_time", 0)

        if hit_rate < 0.70:
            recommendations.append(
                "Consider reviewing cache TTL settings - low hit rate suggests data expires too quickly"
            )

        if hit_rate < 0.60:
            recommendations.append(
                "Implement cache warming strategies for frequently accessed data"
            )

        if avg_response_time > 0.050:
            recommendations.append(
                "Consider Redis connection pooling optimization or server resources"
            )

        if avg_response_time > 0.100:
            recommendations.append(
                "Review Redis server configuration and consider upgrading hardware"
            )

        error_rate = cache_stats.get("error_rate", 0)
        if error_rate > 0.01:
            recommendations.append("Investigate cache errors and consider implementing retry logic")

        # Memory-based recommendations
        memory_usage_percent = self._calculate_memory_usage_percent(health_data)
        if memory_usage_percent > 80:
            recommendations.append(
                "Consider increasing Redis maxmemory or implementing eviction policies"
            )

        if len(self._active_alerts) > 0:
            recommendations.append("Address active alerts to improve cache stability")

        return recommendations

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    async def clear_acknowledged_alerts(self) -> int:
        """Clear all acknowledged alerts."""
        cleared_count = 0
        alerts_to_remove = []

        for alert_id, alert in self._active_alerts.items():
            if alert.acknowledged:
                alerts_to_remove.append(alert_id)
                cleared_count += 1

        for alert_id in alerts_to_remove:
            del self._active_alerts[alert_id]

        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} acknowledged alerts")

        return cleared_count

    async def get_performance_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get performance trends over specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filter history to requested timeframe
        filtered_history = [
            point for point in self._performance_history if point["timestamp"] >= cutoff_time
        ]

        if not filtered_history:
            return {"error": "No data available for requested timeframe"}

        # Calculate trends
        trends = {
            "timeframe_hours": hours,
            "data_points": len(filtered_history),
            "response_times": [point.get("ping_time", 0) for point in filtered_history],
            "memory_usage_trend": self._analyze_memory_trend(filtered_history),
            "operation_rate_trend": self._analyze_operation_trend(filtered_history),
            "client_connection_trend": [
                point.get("connected_clients", 0) for point in filtered_history
            ],
            "timestamps": [point["timestamp"].isoformat() for point in filtered_history],
        }

        return trends

    def _analyze_memory_trend(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze memory usage trend."""
        memory_values = [point.get("memory_used", "0B") for point in history]

        return {
            "current": memory_values[-1] if memory_values else "0B",
            "trend": "stable",  # Simplified - would need proper parsing
            "peak": max(memory_values, key=len) if memory_values else "0B",
        }

    def _analyze_operation_trend(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze operation rate trend."""
        if len(history) < 2:
            return {"trend": "insufficient_data"}

        # Calculate operation rates
        rates = []
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]

            time_diff = (curr["timestamp"] - prev["timestamp"]).total_seconds()
            if time_diff > 0:
                hits_diff = curr.get("keyspace_hits", 0) - prev.get("keyspace_hits", 0)
                misses_diff = curr.get("keyspace_misses", 0) - prev.get("keyspace_misses", 0)
                rate = (hits_diff + misses_diff) / time_diff
                rates.append(rate)

        if not rates:
            return {"trend": "no_data"}

        avg_rate = sum(rates) / len(rates)
        recent_rate = sum(rates[-5:]) / min(5, len(rates))  # Last 5 data points

        trend = "stable"
        if recent_rate > avg_rate * 1.2:
            trend = "increasing"
        elif recent_rate < avg_rate * 0.8:
            trend = "decreasing"

        return {
            "average_ops_per_second": avg_rate,
            "recent_ops_per_second": recent_rate,
            "trend": trend,
        }


# Global monitor instance
_cache_monitor: CacheMonitor | None = None


def get_cache_monitor(config: Any | None = None) -> CacheMonitor:
    """Get or create global cache monitor instance."""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CacheMonitor(config)
    return _cache_monitor
