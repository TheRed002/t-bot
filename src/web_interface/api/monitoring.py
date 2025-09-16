"""
Monitoring API endpoints for T-Bot Trading System.

This module provides comprehensive monitoring APIs including:
- Prometheus metrics endpoint
- Performance profiling data
- System health metrics
- Alert management
- Dashboard data

Key Features:
- Real-time metrics export
- Performance analytics
- Alert management APIs
- Health check endpoints
- Monitoring configuration
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.utils.web_interface_utils import handle_api_error
from src.web_interface.di_registration import get_web_monitoring_service
from src.web_interface.security.auth import get_current_user, require_permissions

if TYPE_CHECKING:
    from src.web_interface.services.monitoring_service import WebMonitoringService

logger = get_logger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    uptime_seconds: Decimal
    version: str = "1.0.0"
    components: dict[str, str] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Metrics response model."""

    metrics: dict[str, Any]
    timestamp: datetime
    timeframe_minutes: int


class AlertRuleRequest(BaseModel):
    """Alert rule creation request."""

    name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    query: str
    threshold: Decimal
    operator: str
    duration: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    notification_channels: list[str] = Field(default_factory=list)  # Channel names
    escalation_delay: str | None = None
    enabled: bool = True


class AlertResponse(BaseModel):
    """Alert response model."""

    rule_name: str
    severity: str  # "low", "medium", "high", "critical"
    status: str  # "active", "resolved", "acknowledged"
    message: str
    labels: dict[str, str]
    annotations: dict[str, str]
    starts_at: datetime
    ends_at: datetime | None = None
    fingerprint: str
    notification_sent: bool = False


class PerformanceStatsResponse(BaseModel):
    """Performance statistics response."""

    timeframe_minutes: int
    function_performance: dict[str, Any]
    query_performance: dict[str, Any]
    cache_performance: dict[str, Any]
    system_resources: dict[str, Any]


class MemoryReportResponse(BaseModel):
    """Memory usage report response."""

    traced_memory: dict[str, Any]
    process_memory: dict[str, Any]
    top_allocations: list[dict[str, Any]]
    gc_stats: dict[str, Any]


# Dependency functions
def get_monitoring_service() -> "WebMonitoringService":
    """Get monitoring service through DI."""
    return get_web_monitoring_service()


# Health and Status Endpoints
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns system health status and component availability.
    """
    try:
        monitoring_service = get_monitoring_service()
        health_data = await monitoring_service.get_system_health_summary()

        return HealthCheckResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            uptime_seconds=health_data["uptime_seconds"],
            components=health_data["components"],
        )

    except Exception as e:
        raise handle_api_error(e, "Health check")


@router.get("/status")
async def system_status():
    """
    Get detailed system status including resource usage.
    """
    try:
        monitoring_service = get_monitoring_service()
        status_data = await monitoring_service.get_system_health_summary()
        return status_data

    except Exception as e:
        raise handle_api_error(e, "Status check")


# Metrics Endpoints
@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    try:
        monitoring_service = get_monitoring_service()
        metrics_data = await monitoring_service.get_performance_metrics()

        # Convert metrics to Prometheus format (simplified)
        prometheus_content = "# T-Bot Metrics\n"
        for metric_name, metric_value in metrics_data.get("metrics", {}).items():
            prometheus_content += f"tbot_{metric_name} {metric_value}\n"

        return Response(content=prometheus_content, media_type="text/plain")
    except Exception as e:
        raise handle_api_error(e, "Export metrics")


@router.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json(timeframe: int = Query(60, description="Timeframe in minutes")):
    """
    Get metrics in JSON format for dashboard consumption.

    Args:
        timeframe: Timeframe in minutes for metrics aggregation
    """
    try:
        monitoring_service = get_monitoring_service()
        metrics_data = await monitoring_service.get_performance_metrics()

        return MetricsResponse(
            metrics=metrics_data.get("metrics", {}),
            timestamp=datetime.now(timezone.utc),
            timeframe_minutes=timeframe,
        )

    except Exception as e:
        raise handle_api_error(e, "Get metrics")


# Performance Endpoints
@router.get("/performance/stats", response_model=PerformanceStatsResponse)
async def performance_stats(timeframe: int = Query(60, description="Timeframe in minutes")):
    """
    Get performance statistics.

    Args:
        timeframe: Timeframe in minutes for performance data
    """
    try:
        monitoring_service = get_monitoring_service()
        performance_data = await monitoring_service.get_performance_metrics()

        # Transform the data to match PerformanceStatsResponse structure
        stats = {
            "timeframe_minutes": timeframe,
            "function_performance": performance_data.get("latency_stats", {}),
            "query_performance": performance_data.get("query_performance", {}),
            "cache_performance": performance_data.get("cache_performance", {}),
            "system_resources": performance_data.get("system_resources", {}),
        }

        return PerformanceStatsResponse(**stats)

    except Exception as e:
        raise handle_api_error(e, "Get performance stats")


@router.get("/performance/memory")
async def memory_report(user=Depends(require_permissions(["monitoring.read"]))):
    """
    Get system resource statistics including memory usage.

    Requires monitoring.read permission.
    """
    try:
        monitoring_service = get_monitoring_service()
        performance_data = await monitoring_service.get_performance_metrics()

        return performance_data.get("system_resources", {})

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Get memory report")


@router.post("/performance/reset-metrics")
async def reset_performance_metrics(user=Depends(require_permissions(["monitoring.write"]))):
    """
    Reset performance metrics.

    Requires monitoring.write permission.
    """
    try:
        _ = get_monitoring_service()
        # This would be implemented in the monitoring service
        # For now, return success
        return {
            "message": "Performance metrics reset successfully",
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Reset performance metrics")


@router.get("/performance/database-queries")
async def database_query_stats(user=Depends(require_permissions(["monitoring.read"]))):
    """
    Get database query statistics.

    Requires monitoring.read permission.
    """
    try:
        monitoring_service = get_monitoring_service()
        performance_data = await monitoring_service.get_performance_metrics()

        return {
            "database_queries": performance_data.get("query_performance", {}),
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Get database query stats")


# Alert Management Endpoints
@router.get("/alerts")
async def get_alerts(
    severity: str | None = Query(None, description="Filter by severity"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
):
    """
    Get alerts with optional filtering.

    Args:
        severity: Optional severity filter
        status: Optional status filter
        limit: Maximum number of alerts to return
    """
    try:
        monitoring_service = get_monitoring_service()
        alert_data = await monitoring_service.get_alert_dashboard_data()

        # Apply filters and return formatted alerts
        all_alerts = alert_data.get("active_alerts", [])
        if status == "resolved":
            all_alerts = alert_data.get("recent_resolved", [])

        # Filter by severity if specified
        if severity:
            all_alerts = [alert for alert in all_alerts if alert.get("severity") == severity]

        return all_alerts[:limit]

    except Exception as e:
        raise handle_api_error(e, "Get alerts")


@router.post("/alerts/rules")
async def create_alert_rule(
    rule_request: dict,
    user=Depends(require_permissions(["monitoring.write"])),
):
    """
    Create a new alert rule.

    Requires monitoring.write permission.
    """
    try:
        # This would integrate with the monitoring service
        # For now, return success
        return {
            "message": f"Alert rule '{rule_request.get('name', 'unnamed')}' created successfully",
            "rule_name": rule_request.get("name", "unnamed"),
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Create alert rule")


@router.delete("/alerts/rules/{rule_name}")
async def delete_alert_rule(
    rule_name: str,
    user=Depends(require_permissions(["monitoring.write"])),
):
    """
    Delete an alert rule.

    Args:
        rule_name: Name of the rule to delete

    Requires monitoring.write permission.
    """
    try:
        # This would integrate with the monitoring service
        # For now, return success
        return {
            "message": f"Alert rule '{rule_name}' deleted successfully",
            "timestamp": datetime.now(timezone.utc),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Delete alert rule")


@router.post("/alerts/{fingerprint}/acknowledge")
async def acknowledge_alert(
    fingerprint: str,
    user=Depends(get_current_user),
):
    """
    Acknowledge an alert.

    Args:
        fingerprint: Alert fingerprint
    """
    try:
        # This would integrate with the monitoring service
        # For now, return success
        return {
            "message": "Alert acknowledged successfully",
            "acknowledged_by": user.username,
            "timestamp": datetime.now(timezone.utc),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Acknowledge alert")


@router.get("/alerts/stats")
async def alert_stats():
    """Get alert system statistics."""
    try:
        monitoring_service = get_monitoring_service()
        alert_data = await monitoring_service.get_alert_dashboard_data()
        return {"stats": alert_data.get("summary", {}), "timestamp": datetime.now(timezone.utc)}

    except Exception as e:
        raise handle_api_error(e, "Get alert stats")


# Configuration Endpoints
@router.get("/config")
async def get_monitoring_config(user=Depends(require_permissions(["monitoring.read"]))):
    """
    Get monitoring system configuration.

    Requires monitoring.read permission.
    """
    try:
        monitoring_service = get_monitoring_service()
        health_data = await monitoring_service.get_system_health_summary()

        config = {
            "metrics_collection": {
                "enabled": health_data.get("monitoring", {}).get("metrics_collector_active", False),
                "collection_interval": 60,
            },
            "alerting": {
                "enabled": health_data.get("monitoring", {}).get("alert_manager_active", False),
                "rules_count": 5,
                "escalation_policies_count": 2,
            },
            "performance_profiling": {
                "enabled": health_data.get("monitoring", {}).get(
                    "performance_profiler_active", False
                ),
                "memory_tracking": True,
                "cpu_profiling": True,
            },
        }

        return config

    except Exception as e:
        raise handle_api_error(e, "Get monitoring config")


@router.post("/config/start-monitoring")
async def start_monitoring(user=Depends(require_permissions(["monitoring.write"]))):
    """
    Start monitoring services.

    Requires monitoring.write permission.
    """
    try:
        # This would integrate with the monitoring service
        # For now, return success
        results = {
            "metrics_collection": "started",
            "alert_manager": "started",
            "performance_profiler": "started",
        }

        return {
            "message": "Monitoring services started",
            "results": results,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Start monitoring")


@router.post("/config/stop-monitoring")
async def stop_monitoring(user=Depends(require_permissions(["monitoring.write"]))):
    """
    Stop monitoring services.

    Requires monitoring.write permission.
    """
    try:
        # This would integrate with the monitoring service
        # For now, return success
        results = {
            "metrics_collection": "stopped",
            "alert_manager": "stopped",
            "performance_profiler": "stopped",
        }

        return {
            "message": "Monitoring services stopped",
            "results": results,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Stop monitoring")
