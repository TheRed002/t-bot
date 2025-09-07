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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.monitoring import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    MetricsCollector,
    NotificationChannel,
    PerformanceProfiler,
    get_alert_manager,
    get_metrics_collector,
    get_performance_profiler,
    get_trading_tracer,
)
from src.utils import handle_api_error
from src.web_interface.security.auth import get_current_user, require_permissions

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
    severity: AlertSeverity
    query: str
    threshold: Decimal
    operator: str
    duration: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    notification_channels: list[NotificationChannel] = Field(default_factory=list)
    escalation_delay: str | None = None
    enabled: bool = True


class AlertResponse(BaseModel):
    """Alert response model."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
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
async def get_metrics_collector_dep() -> MetricsCollector:
    """Get metrics collector dependency."""
    try:
        collector = get_metrics_collector()
        if not collector:
            logger.warning("Metrics collector not initialized - service unavailable")
            raise HTTPException(
                status_code=503,
                detail="Metrics collector service is not available. Please check monitoring setup.",
            )
        return collector
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Get metrics collector")


async def get_alert_manager_dep() -> AlertManager:
    """Get alert manager dependency."""
    try:
        manager = get_alert_manager()
        if not manager:
            logger.warning("Alert manager not initialized - service unavailable")
            raise HTTPException(
                status_code=503,
                detail="Alert manager service is not available. Please check monitoring setup.",
            )
        return manager
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Get alert manager")


async def get_profiler_dep() -> PerformanceProfiler:
    """Get performance profiler dependency."""
    try:
        profiler = get_performance_profiler()
        if not profiler:
            logger.warning("Performance profiler not initialized - service unavailable")
            raise HTTPException(
                status_code=503,
                detail="Performance profiler service is not available. Please check monitoring setup.",
            )
        return profiler
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Get performance profiler")


# Health and Status Endpoints
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns system health status and component availability.
    """
    try:
        import time

        import psutil

        # Get system info
        process = psutil.Process()
        uptime = time.time() - process.create_time()

        # Check component health
        components = {}

        # Check metrics collector
        try:
            collector = get_metrics_collector()
            components["metrics_collector"] = "healthy" if collector else "unavailable"
        except Exception as e:
            logger.error(f"Failed to check metrics collector: {e}")
            components["metrics_collector"] = "error"

        # Check alert manager
        try:
            alert_manager = get_alert_manager()
            components["alert_manager"] = "healthy" if alert_manager else "unavailable"
        except Exception as e:
            logger.error(f"Failed to check alert manager: {e}")
            components["alert_manager"] = "error"

        # Check performance profiler
        try:
            profiler = get_performance_profiler()
            components["performance_profiler"] = "healthy" if profiler else "unavailable"
        except Exception as e:
            logger.error(f"Failed to check performance profiler: {e}")
            components["performance_profiler"] = "error"

        # Check trading tracer
        try:
            tracer = get_trading_tracer()
            components["trading_tracer"] = "healthy" if tracer else "unavailable"
        except Exception as e:
            logger.error(f"Failed to check trading tracer: {e}")
            components["trading_tracer"] = "error"

        # Determine overall status
        status = "healthy"
        if any(status == "error" for status in components.values()):
            status = "degraded"
        elif any(status == "unavailable" for status in components.values()):
            status = "partial"

        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=uptime,
            components=components,
        )

    except Exception as e:
        raise handle_api_error(e, "Health check")


@router.get("/status")
async def system_status():
    """
    Get detailed system status including resource usage.
    """
    try:
        import psutil

        # System resources
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()

        # Monitoring component status
        collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        profiler = get_performance_profiler()

        stats = {}
        if alert_manager:
            stats = alert_manager.get_alert_stats()

        return {
            "timestamp": datetime.now(timezone.utc),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "num_threads": process.num_threads(),
            },
            "monitoring": {
                "metrics_collector_active": collector is not None,
                "alert_manager_active": alert_manager is not None,
                "performance_profiler_active": profiler is not None,
            },
            "alerts": stats,
        }

    except Exception as e:
        raise handle_api_error(e, "Status check")


# Metrics Endpoints
@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(collector: MetricsCollector = Depends(get_metrics_collector_dep)):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    try:
        metrics_content = collector.export_metrics()
        return Response(content=metrics_content, media_type=collector.get_metrics_content_type())
    except Exception as e:
        raise handle_api_error(e, "Export metrics")


@router.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json(
    timeframe: int = Query(60, description="Timeframe in minutes"),
    profiler: PerformanceProfiler = Depends(get_profiler_dep),
):
    """
    Get metrics in JSON format for dashboard consumption.

    Args:
        timeframe: Timeframe in minutes for metrics aggregation
    """
    try:
        # Get performance summary (doesn't take timeframe parameter)
        metrics_data = profiler.get_performance_summary()

        return MetricsResponse(
            metrics=metrics_data, timestamp=datetime.now(timezone.utc), timeframe_minutes=timeframe
        )

    except Exception as e:
        raise handle_api_error(e, "Get metrics")


# Performance Endpoints
@router.get("/performance/stats", response_model=PerformanceStatsResponse)
async def performance_stats(
    timeframe: int = Query(60, description="Timeframe in minutes"),
    profiler: PerformanceProfiler = Depends(get_profiler_dep),
):
    """
    Get performance statistics.

    Args:
        timeframe: Timeframe in minutes for performance data
    """
    try:
        # Get the performance summary
        summary = profiler.get_performance_summary()

        # Transform the data to match PerformanceStatsResponse structure
        stats = {
            "timeframe_minutes": timeframe,
            "function_performance": summary.get("latency_stats", {}),
            "query_performance": {},  # Would need database-specific metrics
            "cache_performance": {},  # Would need cache-specific metrics
            "system_resources": summary.get("system_resources", {}),
        }

        return PerformanceStatsResponse(**stats)

    except Exception as e:
        raise handle_api_error(e, "Get performance stats")


@router.get("/performance/memory")
async def memory_report(
    user=Depends(require_permissions(["monitoring.read"])),
    profiler: PerformanceProfiler = Depends(get_profiler_dep),
):
    """
    Get system resource statistics including memory usage.

    Requires monitoring.read permission.
    """
    try:
        # Use actual existing method
        resource_stats = profiler.get_system_resource_stats()
        gc_stats = profiler.get_gc_stats()

        if not resource_stats:
            raise HTTPException(
                status_code=503,
                detail="Resource statistics not available. Monitoring may not be started.",
            )

        return {
            "memory": {
                "rss_mb": resource_stats.memory_rss / (1024 * 1024) if resource_stats else 0,
                "vms_mb": resource_stats.memory_vms / (1024 * 1024) if resource_stats else 0,
                "percent": resource_stats.memory_percent if resource_stats else 0,
            },
            "cpu": {
                "percent": resource_stats.cpu_percent if resource_stats else 0,
            },
            "gc_stats": (
                {
                    "count": gc_stats.count if gc_stats else {},
                    "collected": gc_stats.collected if gc_stats else {},
                    "uncollectable": gc_stats.uncollectable if gc_stats else {},
                    "collections_per_second": gc_stats.collections_per_second if gc_stats else {},
                }
                if gc_stats
                else None
            ),
            "timestamp": datetime.now(timezone.utc),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "Get memory report")


@router.post("/performance/reset-metrics")
async def reset_performance_metrics(
    user=Depends(require_permissions(["monitoring.write"])),
    profiler: PerformanceProfiler = Depends(get_profiler_dep),
):
    """
    Reset performance metrics.

    Requires monitoring.write permission.
    """
    try:
        profiler.reset_metrics()

        return {
            "message": "Performance metrics reset successfully",
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Reset performance metrics")


@router.get("/performance/database-queries")
async def database_query_stats(
    user=Depends(require_permissions(["monitoring.read"])),
    profiler: PerformanceProfiler = Depends(get_profiler_dep),
):
    """
    Get database query statistics.

    Requires monitoring.read permission.
    """
    try:
        # Get performance summary which includes database stats
        summary = profiler.get_performance_summary()

        database_stats = {}
        if "database" in summary:
            database_stats = summary["database"]

        return {
            "database_queries": database_stats,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Get database query stats")


# Alert Management Endpoints
@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    severity: AlertSeverity | None = Query(None, description="Filter by severity"),
    status: AlertStatus | None = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
    alert_manager: AlertManager = Depends(get_alert_manager_dep),
):
    """
    Get alerts with optional filtering.

    Args:
        severity: Optional severity filter
        status: Optional status filter
        limit: Maximum number of alerts to return
    """
    try:
        if status == AlertStatus.FIRING:
            alerts = alert_manager.get_active_alerts(severity)
        else:
            alerts = alert_manager.get_alert_history(limit)
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if status:
                alerts = [a for a in alerts if a.status == status]

        return [
            AlertResponse(
                rule_name=alert.rule_name,
                severity=alert.severity,
                status=alert.status,
                message=alert.message,
                labels=alert.labels,
                annotations=alert.annotations,
                starts_at=alert.starts_at,
                ends_at=alert.ends_at,
                fingerprint=alert.fingerprint,
                notification_sent=alert.notification_sent,
            )
            for alert in alerts[:limit]
        ]

    except Exception as e:
        raise handle_api_error(e, "Get alerts")


@router.post("/alerts/rules")
async def create_alert_rule(
    rule_request: AlertRuleRequest,
    user=Depends(require_permissions(["monitoring.write"])),
    alert_manager: AlertManager = Depends(get_alert_manager_dep),
):
    """
    Create a new alert rule.

    Requires monitoring.write permission.
    """
    try:
        alert_rule = AlertRule(
            name=rule_request.name,
            description=rule_request.description,
            severity=rule_request.severity,
            query=rule_request.query,
            threshold=rule_request.threshold,
            operator=rule_request.operator,
            duration=rule_request.duration,
            labels=rule_request.labels,
            annotations=rule_request.annotations,
            notification_channels=rule_request.notification_channels,
            escalation_delay=rule_request.escalation_delay,
            enabled=rule_request.enabled,
        )

        alert_manager.add_rule(alert_rule)

        return {
            "message": f"Alert rule '{rule_request.name}' created successfully",
            "rule_name": rule_request.name,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Create alert rule")


@router.delete("/alerts/rules/{rule_name}")
async def delete_alert_rule(
    rule_name: str,
    user=Depends(require_permissions(["monitoring.write"])),
    alert_manager: AlertManager = Depends(get_alert_manager_dep),
):
    """
    Delete an alert rule.

    Args:
        rule_name: Name of the rule to delete

    Requires monitoring.write permission.
    """
    try:
        success = alert_manager.remove_rule(rule_name)

        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")

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
    alert_manager: AlertManager = Depends(get_alert_manager_dep),
):
    """
    Acknowledge an alert.

    Args:
        fingerprint: Alert fingerprint
    """
    try:
        success = await alert_manager.acknowledge_alert(fingerprint, user.username)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

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
async def alert_stats(alert_manager: AlertManager = Depends(get_alert_manager_dep)):
    """Get alert system statistics."""
    try:
        stats = alert_manager.get_alert_stats()
        return {"stats": stats, "timestamp": datetime.now(timezone.utc)}

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
        collector = get_metrics_collector()
        alert_manager = get_alert_manager()
        profiler = get_performance_profiler()

        config = {
            "metrics_collection": {
                "enabled": collector is not None,
                "collection_interval": (
                    getattr(collector, "_collection_interval", None) if collector else None
                ),
            },
            "alerting": {
                "enabled": alert_manager is not None,
                "rules_count": len(getattr(alert_manager, "_rules", {})) if alert_manager else 0,
                "escalation_policies_count": (
                    len(getattr(alert_manager, "_escalation_policies", {})) if alert_manager else 0
                ),
            },
            "performance_profiling": {
                "enabled": profiler is not None,
                "memory_tracking": (
                    getattr(profiler, "enable_memory_tracking", False) if profiler else False
                ),
                "cpu_profiling": (
                    getattr(profiler, "enable_cpu_profiling", False) if profiler else False
                ),
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
        results = {}

        # Start metrics collection
        collector = get_metrics_collector()
        if collector:
            await collector.start_collection()
            results["metrics_collection"] = "started"
        else:
            results["metrics_collection"] = "not available"

        # Start alert manager
        alert_manager = get_alert_manager()
        if alert_manager:
            await alert_manager.start()
            results["alert_manager"] = "started"
        else:
            results["alert_manager"] = "not available"

        # Start performance profiler
        profiler = get_performance_profiler()
        if profiler:
            await profiler.start()
            results["performance_profiler"] = "started"
        else:
            results["performance_profiler"] = "not available"

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
        results = {}

        # Stop metrics collection
        collector = get_metrics_collector()
        if collector:
            await collector.stop_collection()
            results["metrics_collection"] = "stopped"
        else:
            results["metrics_collection"] = "not available"

        # Stop alert manager
        alert_manager = get_alert_manager()
        if alert_manager:
            await alert_manager.stop()
            results["alert_manager"] = "stopped"
        else:
            results["alert_manager"] = "not available"

        # Stop performance profiler
        profiler = get_performance_profiler()
        if profiler:
            await profiler.stop()
            results["performance_profiler"] = "stopped"
        else:
            results["performance_profiler"] = "not available"

        return {
            "message": "Monitoring services stopped",
            "results": results,
            "timestamp": datetime.now(timezone.utc),
        }

    except Exception as e:
        raise handle_api_error(e, "Stop monitoring")
