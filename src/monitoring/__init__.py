"""
Monitoring infrastructure package for T-Bot Trading System.

This package provides comprehensive monitoring, alerting, and performance optimization
tools for the trading system following clean architecture principles.

Service Layer Architecture:
- Service interfaces abstract business operations
- Dependency injection manages infrastructure concerns
- Clean separation between business logic and infrastructure

Components:
- P-030: Monitoring Infrastructure (Prometheus, Grafana, custom metrics)
- P-031: Alerting System (AlertManager, notifications, escalation)
- P-032: Performance Optimization (profiling, optimization, bottleneck detection)

Service Layer:
    services: Service interfaces for monitoring operations
    dependency_injection: DI container for managing dependencies

Infrastructure Layer:
    metrics: Prometheus metrics collection and custom trading metrics
    alerting: Alert rules, notification channels, escalation policies
    performance: Profiling tools, performance monitoring, optimization
    dashboards: Grafana dashboard configurations
    telemetry: OpenTelemetry instrumentation and tracing
"""

from .alerting import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationConfig,
)
from .dependency_injection import (
    DIContainer,
    get_container,
    setup_monitoring_dependencies,
)
from .metrics import (
    ExchangeMetrics,
    MetricDefinition,
    MetricsCollector,
    RiskMetrics,
    SystemMetrics,
    TradingMetrics,
    setup_prometheus_server,
)
from .performance import (
    PerformanceProfiler,
    profile_async,
    profile_sync,
)
from .services import (
    AlertRequest,
    AlertService,
    DefaultAlertService,
    DefaultMetricsService,
    DefaultPerformanceService,
    MetricRequest,
    MetricsService,
    MonitoringService,
    PerformanceService,
)
from .telemetry import (
    OpenTelemetryConfig,
    TradingTracer,
    get_tracer,
    get_trading_tracer,
    instrument_fastapi,
    setup_telemetry,
    trace_async_function,
    trace_function,
)

try:
    from .trace_wrapper import Status, StatusCode, trace
except ImportError:
    # Mock implementations if trace_wrapper is not available
    class Status:
        pass

    class StatusCode:
        OK = "ok"
        ERROR = "error"

    def trace(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


__all__ = [
    # Service Layer
    "MonitoringService",
    "AlertService",
    "MetricsService",
    "PerformanceService",
    "DefaultAlertService",
    "DefaultMetricsService",
    "DefaultPerformanceService",
    "AlertRequest",
    "MetricRequest",
    # Core Infrastructure
    "Alert",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "ExchangeMetrics",
    "MetricDefinition",
    "MetricsCollector",
    "NotificationChannel",
    "NotificationConfig",
    "OpenTelemetryConfig",
    "PerformanceProfiler",
    "RiskMetrics",
    "Status",
    "StatusCode",
    "SystemMetrics",
    "TradingMetrics",
    "TradingTracer",
    # Dependency Injection
    "DIContainer",
    "get_container",
    "setup_monitoring_dependencies",
    # Utilities
    "get_tracer",
    "get_trading_tracer",
    "instrument_fastapi",
    "profile_async",
    "profile_sync",
    "setup_prometheus_server",
    "setup_telemetry",
    "trace",
    "trace_async_function",
    "trace_function",
    # Main initialization function
    "initialize_monitoring_service",
]


def initialize_monitoring_service(
    notification_config: NotificationConfig | None = None,
    metrics_registry=None,
    telemetry_config: OpenTelemetryConfig | None = None,
    prometheus_port: int = 8001,
) -> MonitoringService:
    """
    Initialize comprehensive monitoring service using proper service layer.

    This function sets up the complete monitoring infrastructure following
    clean architecture principles with proper dependency injection.

    Args:
        notification_config: Alert notification configuration
        metrics_registry: Prometheus metrics registry
        telemetry_config: OpenTelemetry configuration
        prometheus_port: Port for Prometheus metrics server

    Returns:
        Initialized MonitoringService instance with all dependencies injected
    """
    # Initialize infrastructure components with dependency injection
    metrics_collector = MetricsCollector(metrics_registry)

    if notification_config is None:
        notification_config = NotificationConfig()
    alert_manager = AlertManager(notification_config)

    performance_profiler = PerformanceProfiler(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
    )

    # Initialize telemetry if configured
    if telemetry_config:
        setup_telemetry(telemetry_config)

    # Setup Prometheus server
    try:
        setup_prometheus_server(prometheus_port)
    except Exception as e:
        # Log warning but don't fail initialization
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to setup Prometheus server on port {prometheus_port}: {e}")

    # Create service layer with injected dependencies
    alert_service = DefaultAlertService(alert_manager)
    metrics_service = DefaultMetricsService(metrics_collector)
    performance_service = DefaultPerformanceService(performance_profiler)

    return MonitoringService(alert_service, metrics_service, performance_service)
