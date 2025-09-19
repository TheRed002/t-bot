"""
Monitoring service for web interface business logic.

This service handles all monitoring-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError
from src.web_interface.interfaces import WebMonitoringServiceInterface


class WebMonitoringService(BaseComponent):
    """Service handling monitoring business logic for web interface."""

    def __init__(self, monitoring_facade=None):
        super().__init__()
        self.monitoring_facade = monitoring_facade

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web monitoring service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web monitoring service cleaned up")

    async def get_system_health_summary(self) -> dict[str, Any]:
        """Get system health summary with web-specific formatting."""
        try:
            if self.monitoring_facade:
                # Get health data from monitoring system
                health_data = await self.monitoring_facade.get_system_health()
            else:
                # Mock data for development
                health_data = {
                    "overall_status": "healthy",
                    "components": {
                        "database": {"status": "healthy", "response_time_ms": 12.3},
                        "redis": {"status": "healthy", "memory_usage": "45%"},
                        "exchange_connections": {"status": "healthy", "active_connections": 3},
                        "trading_engine": {"status": "healthy", "orders_per_minute": 15.2},
                        "data_feeds": {"status": "degraded", "feed_lag_ms": 250},
                    },
                    "metrics": {
                        "uptime_seconds": 86400 * 5,  # 5 days
                        "total_requests": 125000,
                        "error_rate": 0.02,
                        "cpu_usage": 35.5,
                        "memory_usage": 67.8,
                        "disk_usage": 23.4,
                    },
                    "alerts": {
                        "active": 2,
                        "resolved_today": 8,
                    },
                }

            # Business logic: calculate health scores and format data
            overall_health_score = self._calculate_overall_health_score(health_data)
            formatted_components = self._format_component_health(health_data.get("components", {}))

            # Business logic: format uptime
            uptime_seconds = health_data.get("metrics", {}).get("uptime_seconds", 0)
            formatted_uptime = self._format_uptime(uptime_seconds)

            return {
                "overall_status": health_data.get("overall_status", "unknown"),
                "health_score": overall_health_score,
                "uptime": formatted_uptime,
                "uptime_seconds": uptime_seconds,
                "components": formatted_components,
                "performance": {
                    "cpu_usage": health_data.get("metrics", {}).get("cpu_usage", 0),
                    "memory_usage": health_data.get("metrics", {}).get("memory_usage", 0),
                    "disk_usage": health_data.get("metrics", {}).get("disk_usage", 0),
                    "error_rate": health_data.get("metrics", {}).get("error_rate", 0),
                    "total_requests": health_data.get("metrics", {}).get("total_requests", 0),
                },
                "alerts": health_data.get("alerts", {"active": 0, "resolved_today": 0}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting system health summary: {e}")
            raise ServiceError(f"Failed to get system health summary: {e}")

    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]:
        """Get performance metrics with web-specific processing."""
        try:
            if self.monitoring_facade:
                metrics_data = await self.monitoring_facade.get_performance_metrics(component)
            else:
                # Mock data for development
                metrics_data = {
                    "trading": {
                        "orders_per_second": 25.5,
                        "execution_latency_ms": 12.3,
                        "fill_rate": 0.95,
                        "slippage_avg": 0.0012,
                    },
                    "data": {
                        "feed_latency_ms": 45.2,
                        "messages_per_second": 1250,
                        "data_quality_score": 0.98,
                        "cache_hit_rate": 0.85,
                    },
                    "system": {
                        "cpu_cores_used": 4.2,
                        "memory_gb_used": 8.5,
                        "network_mbps": 125.8,
                        "disk_iops": 850,
                    },
                    "database": {
                        "queries_per_second": 450,
                        "avg_response_time_ms": 8.7,
                        "connection_pool_usage": 0.65,
                        "slow_queries": 3,
                    },
                }

            # Business logic: filter by component if specified
            if component:
                filtered_metrics = metrics_data.get(component, {})
                if not filtered_metrics:
                    raise ServiceError(f"Component '{component}' not found")
                metrics_data = {component: filtered_metrics}

            # Business logic: calculate performance scores for each component
            processed_metrics = {}
            for comp_name, comp_metrics in metrics_data.items():
                performance_score = self._calculate_performance_score(comp_name, comp_metrics)
                processed_metrics[comp_name] = {
                    **comp_metrics,
                    "performance_score": performance_score,
                    "status": "good"
                    if performance_score >= 0.8
                    else "warning"
                    if performance_score >= 0.6
                    else "critical",
                }

            return {
                "metrics": processed_metrics,
                "component_filter": component,
                "summary": {
                    "components_monitored": len(processed_metrics),
                    "avg_performance_score": sum(
                        m["performance_score"] for m in processed_metrics.values()
                    )
                    / len(processed_metrics)
                    if processed_metrics
                    else 0,
                    "critical_components": sum(
                        1 for m in processed_metrics.values() if m["status"] == "critical"
                    ),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            raise ServiceError(f"Failed to get performance metrics: {e}")

    async def get_error_summary(self, time_range: str = "24h") -> dict[str, Any]:
        """Get error summary with web-specific analysis."""
        try:
            # Business logic: parse time range
            hours = self._parse_time_range(time_range)
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            end_time = datetime.now(timezone.utc)

            if self.monitoring_facade:
                error_data = await self.monitoring_facade.get_error_summary(start_time, end_time)
            else:
                # Mock data for development
                error_data = {
                    "total_errors": 45,
                    "error_rate": 0.018,
                    "errors_by_type": {
                        "network_errors": 18,
                        "validation_errors": 12,
                        "exchange_errors": 8,
                        "database_errors": 4,
                        "unknown_errors": 3,
                    },
                    "errors_by_severity": {
                        "critical": 5,
                        "high": 12,
                        "medium": 18,
                        "low": 10,
                    },
                    "recent_errors": [
                        {
                            "timestamp": (
                                datetime.now(timezone.utc) - timedelta(minutes=15)
                            ).isoformat(),
                            "type": "network_error",
                            "severity": "medium",
                            "message": "Connection timeout to exchange API",
                            "component": "exchange_client",
                        },
                        {
                            "timestamp": (
                                datetime.now(timezone.utc) - timedelta(hours=2)
                            ).isoformat(),
                            "type": "validation_error",
                            "severity": "low",
                            "message": "Invalid order size parameter",
                            "component": "trading_api",
                        },
                    ],
                }

            # Business logic: analyze error patterns and trends
            error_analysis = self._analyze_error_patterns(error_data)

            return {
                "time_range": time_range,
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "summary": {
                    "total_errors": error_data.get("total_errors", 0),
                    "error_rate": error_data.get("error_rate", 0),
                    "trend": error_analysis["trend"],
                    "most_common_type": error_analysis["most_common_type"],
                },
                "breakdown": {
                    "by_type": error_data.get("errors_by_type", {}),
                    "by_severity": error_data.get("errors_by_severity", {}),
                },
                "recent_errors": error_data.get("recent_errors", [])[:10],  # Limit to 10 recent
                "analysis": error_analysis,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting error summary: {e}")
            raise ServiceError(f"Failed to get error summary: {e}")

    async def get_alert_dashboard_data(self) -> dict[str, Any]:
        """Get alert dashboard data with web-specific formatting."""
        try:
            if self.monitoring_facade:
                alert_data = await self.monitoring_facade.get_alerts()
            else:
                # Mock data for development
                alert_data = {
                    "active_alerts": [
                        {
                            "id": "alert_001",
                            "type": "high_latency",
                            "severity": "medium",
                            "component": "data_feed",
                            "message": "Data feed latency above threshold",
                            "triggered_at": (
                                datetime.now(timezone.utc) - timedelta(minutes=30)
                            ).isoformat(),
                            "threshold": 100,
                            "current_value": 250,
                        },
                        {
                            "id": "alert_002",
                            "type": "error_rate",
                            "severity": "high",
                            "component": "trading_engine",
                            "message": "Error rate spike detected",
                            "triggered_at": (
                                datetime.now(timezone.utc) - timedelta(hours=1)
                            ).isoformat(),
                            "threshold": 0.05,
                            "current_value": 0.08,
                        },
                    ],
                    "resolved_alerts": [
                        {
                            "id": "alert_003",
                            "type": "disk_space",
                            "severity": "low",
                            "component": "database",
                            "message": "Disk space usage warning",
                            "triggered_at": (
                                datetime.now(timezone.utc) - timedelta(hours=6)
                            ).isoformat(),
                            "resolved_at": (
                                datetime.now(timezone.utc) - timedelta(hours=4)
                            ).isoformat(),
                            "resolution": "Log rotation completed",
                        },
                    ],
                }

            # Business logic: categorize and prioritize alerts
            active_alerts = alert_data.get("active_alerts", [])
            alert_summary = self._categorize_alerts(active_alerts)

            return {
                "summary": {
                    "total_active": len(active_alerts),
                    "critical": alert_summary["critical"],
                    "high": alert_summary["high"],
                    "medium": alert_summary["medium"],
                    "low": alert_summary["low"],
                    "resolved_today": len(alert_data.get("resolved_alerts", [])),
                },
                "active_alerts": sorted(
                    active_alerts,
                    key=lambda x: self._get_severity_priority(x["severity"]),
                    reverse=True,
                ),
                "recent_resolved": alert_data.get("resolved_alerts", [])[:5],  # Last 5 resolved
                "alert_trends": {
                    "most_common_type": alert_summary["most_common_type"],
                    "most_affected_component": alert_summary["most_affected_component"],
                    "avg_resolution_time_minutes": alert_summary["avg_resolution_time"],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting alert dashboard data: {e}")
            raise ServiceError(f"Failed to get alert dashboard data: {e}")

    def _calculate_overall_health_score(self, health_data: dict[str, Any]) -> float:
        """Calculate overall system health score (business logic)."""
        try:
            components = health_data.get("components", {})
            if not components:
                return 0.5  # Neutral score if no data

            total_score = 0
            component_count = 0

            for comp_name, comp_data in components.items():
                status = comp_data.get("status", "unknown").lower()
                if status == "healthy":
                    score = 1.0
                elif status == "degraded":
                    score = 0.6
                elif status == "critical":
                    score = 0.2
                else:
                    score = 0.4  # Unknown status

                total_score += score
                component_count += 1

            return total_score / component_count if component_count > 0 else 0.5

        except Exception as e:
            logger.warning(f"Error calculating overall health score: {e}")
            return 0.5  # Return neutral score on error

    def _format_component_health(self, components: dict[str, Any]) -> dict[str, Any]:
        """Format component health data (business logic)."""
        formatted = {}
        for comp_name, comp_data in components.items():
            formatted[comp_name] = {
                **comp_data,
                "display_name": comp_name.replace("_", " ").title(),
                "status_icon": self._get_status_icon(comp_data.get("status", "unknown")),
            }
        return formatted

    def _format_uptime(self, uptime_seconds: int) -> str:
        """Format uptime in human-readable format (business logic)."""
        if uptime_seconds < 60:
            return f"{uptime_seconds} seconds"
        elif uptime_seconds < 3600:
            return f"{uptime_seconds // 60} minutes"
        elif uptime_seconds < 86400:
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            return f"{hours} hours, {minutes} minutes"
        else:
            days = uptime_seconds // 86400
            hours = (uptime_seconds % 86400) // 3600
            return f"{days} days, {hours} hours"

    def _calculate_performance_score(self, component: str, metrics: dict[str, Any]) -> float:
        """Calculate performance score for a component (business logic)."""
        try:
            score = 1.0

            # Component-specific scoring logic
            if component == "trading":
                latency = metrics.get("execution_latency_ms", 0)
                if latency > 100:
                    score -= 0.3
                elif latency > 50:
                    score -= 0.1

                fill_rate = metrics.get("fill_rate", 1.0)
                if fill_rate < 0.8:
                    score -= 0.4
                elif fill_rate < 0.9:
                    score -= 0.2

            elif component == "data":
                feed_latency = metrics.get("feed_latency_ms", 0)
                if feed_latency > 200:
                    score -= 0.4
                elif feed_latency > 100:
                    score -= 0.2

                quality = metrics.get("data_quality_score", 1.0)
                score *= quality

            elif component == "system":
                cpu_usage = metrics.get("cpu_cores_used", 0)
                if cpu_usage > 8:
                    score -= 0.3
                elif cpu_usage > 6:
                    score -= 0.1

            return max(0.0, min(1.0, score))  # Clamp between 0 and 1

        except Exception as e:
            logger.warning(f"Error calculating component health score: {e}")
            return 0.5  # Return neutral score on error

    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to hours (business logic)."""
        time_range = time_range.lower()
        if time_range.endswith("h"):
            return int(time_range[:-1])
        elif time_range.endswith("d"):
            return int(time_range[:-1]) * 24
        elif time_range.endswith("m"):
            return int(time_range[:-1]) // 60
        else:
            return 24  # Default to 24 hours

    def _analyze_error_patterns(self, error_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze error patterns and trends (business logic)."""
        errors_by_type = error_data.get("errors_by_type", {})
        most_common_type = (
            max(errors_by_type.keys(), key=lambda k: errors_by_type[k])
            if errors_by_type
            else "none"
        )

        total_errors = error_data.get("total_errors", 0)
        if total_errors > 100:
            trend = "high"
        elif total_errors > 50:
            trend = "elevated"
        elif total_errors > 10:
            trend = "normal"
        else:
            trend = "low"

        return {
            "trend": trend,
            "most_common_type": most_common_type,
            "pattern_analysis": "Network errors are the most frequent, suggesting connectivity issues.",
        }

    def _categorize_alerts(self, alerts: list[dict[str, Any]]) -> dict[str, Any]:
        """Categorize alerts by severity and component (business logic)."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        component_counts = {}
        type_counts = {}

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            if severity in severity_counts:
                severity_counts[severity] += 1

            component = alert.get("component", "unknown")
            component_counts[component] = component_counts.get(component, 0) + 1

            alert_type = alert.get("type", "unknown")
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1

        most_affected_component = (
            max(component_counts.keys(), key=lambda k: component_counts[k])
            if component_counts
            else "none"
        )
        most_common_type = (
            max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else "none"
        )

        return {
            **severity_counts,
            "most_affected_component": most_affected_component,
            "most_common_type": most_common_type,
            "avg_resolution_time": 45,  # Mock average resolution time in minutes
        }

    def _get_severity_priority(self, severity: str) -> int:
        """Get severity priority for sorting (business logic)."""
        priorities = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return priorities.get(severity.lower(), 0)

    def _get_status_icon(self, status: str) -> str:
        """Get status icon for display (business logic)."""
        icons = {"healthy": "✅", "degraded": "⚠️", "critical": "❌", "unknown": "❓"}
        return icons.get(status.lower(), "❓")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "service": "WebMonitoringService",
            "status": "healthy",
            "monitoring_facade_available": self.monitoring_facade is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebMonitoringService",
            "description": "Web monitoring service handling system monitoring business logic",
            "capabilities": [
                "system_health_monitoring",
                "performance_metrics_analysis",
                "error_pattern_analysis",
                "alert_dashboard_management",
            ],
            "version": "1.0.0",
        }
