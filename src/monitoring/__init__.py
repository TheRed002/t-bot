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

# Core monitoring components
from typing import Union, Optional

from .alerting import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationConfig,
    get_alert_manager,
    set_global_alert_manager,
)
from .dependency_injection import (
    DIContainer,
    get_monitoring_container,
    setup_monitoring_dependencies,
)
from .di_registration import register_monitoring_services
from .interfaces import (
    AlertServiceInterface,
    DashboardServiceInterface,
    MetricsServiceInterface,
    MonitoringServiceInterface,
    PerformanceServiceInterface,
)
from .metrics import (
    ExchangeMetrics,
    MetricDefinition,
    MetricsCollector,
    RiskMetrics,
    SystemMetrics,
    TradingMetrics,
    get_metrics_collector,
    set_metrics_collector,
    setup_prometheus_server,
)
from .performance import (
    PerformanceProfiler,
    get_performance_profiler,
    profile_async,
    profile_sync,
    set_global_profiler,
)
from .services import (
    AlertRequest,
    DefaultAlertService,
    DefaultMetricsService,
    DefaultPerformanceService,
    MetricRequest,
    MonitoringService,
)

# Service aliases for backwards compatibility
AlertService = DefaultAlertService
MetricsService = DefaultMetricsService
PerformanceService = DefaultPerformanceService

from .telemetry import (
    OpenTelemetryConfig,
    TradingTracer,
    get_tracer,
    get_trading_tracer,
    instrument_fastapi,
    set_global_trading_tracer,
    setup_telemetry,
    trace_async_function,
    trace_function,
)

try:
    from .trace_wrapper import (
        Status as StatusImport,
        StatusCode as StatusCodeImport,
        trace as TraceImport,
    )

    Status = StatusImport
    StatusCode = StatusCodeImport
    trace = TraceImport
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
    # Core Infrastructure
    "Alert",
    "AlertManager",
    "AlertRequest",
    "AlertRule",
    "AlertSeverity",
    "AlertService",
    "AlertServiceInterface",
    "AlertStatus",
    "DashboardServiceInterface",
    "DIContainer",
    "DefaultAlertService",
    "DefaultMetricsService",
    "DefaultPerformanceService",
    "ExchangeMetrics",
    "MetricDefinition",
    "MetricRequest",
    "MetricsCollector",
    "MetricsService",
    "MetricsServiceInterface",
    "MonitoringServiceInterface",
    "NotificationChannel",
    "NotificationConfig",
    "OpenTelemetryConfig",
    "PerformanceProfiler",
    "PerformanceService",
    "PerformanceServiceInterface",
    "RiskMetrics",
    "Status",
    "StatusCode",
    "SystemMetrics",
    "TradingMetrics",
    "TradingTracer",
    # Service Layer
    "MonitoringService",
    # Functions
    "get_alert_manager",
    "get_monitoring_container",
    "get_metrics_collector",
    "get_performance_profiler",
    "get_tracer",
    "get_trading_tracer",
    "initialize_monitoring_service",
    "instrument_fastapi",
    "profile_async",
    "profile_sync",
    "register_monitoring_services",
    "set_global_alert_manager",
    "set_global_profiler",
    "set_global_trading_tracer",
    "set_metrics_collector",
    "setup_monitoring_dependencies",
    "setup_prometheus_server",
    "setup_telemetry",
    "trace",
    "trace_async_function",
    "trace_function",
]


def initialize_monitoring_service(
    notification_config: Optional[NotificationConfig] = None,
    metrics_registry=None,
    telemetry_config: Optional[OpenTelemetryConfig] = None,
    prometheus_port: int = 8001,  # Use METRICS_DEFAULT_PROMETHEUS_PORT from config
    use_dependency_injection: bool = True,
    injector=None,
) -> MonitoringService:
    """
    Initialize comprehensive monitoring service using proper dependency injection.

    This function sets up the complete monitoring infrastructure following
    clean architecture principles with proper dependency injection.

    Args:
        notification_config: Alert notification configuration
        metrics_registry: Prometheus metrics registry
        telemetry_config: OpenTelemetry configuration
        prometheus_port: Port for Prometheus metrics server
        use_dependency_injection: Whether to use DI container (recommended)
        injector: Dependency injector instance (optional)

    Returns:
        Initialized MonitoringService instance with all dependencies injected
    """
    if use_dependency_injection and injector is not None:
        try:
            # Use core dependency injection system
            from src.monitoring.di_registration import register_monitoring_services

            # Register monitoring services with injector
            register_monitoring_services(injector)

            # Initialize telemetry if configured
            if telemetry_config:
                setup_telemetry(telemetry_config)

            # Setup Prometheus server (skip during tests to avoid blocking)
            import os

            if not os.environ.get("TESTING"):
                try:
                    setup_prometheus_server(prometheus_port)
                except Exception as e:
                    # Log warning but don't fail initialization
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to setup Prometheus server on port {prometheus_port}: {e}"
                    )

            # Get monitoring service from DI container
            return injector.resolve("MonitoringServiceInterface")

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Core DI initialization failed, trying monitoring DI: {e}")

    if use_dependency_injection:
        try:
            # Use monitoring-specific dependency injection
            from src.monitoring.dependency_injection import (
                create_monitoring_service,
                setup_monitoring_dependencies,
            )

            # Set up DI container
            setup_monitoring_dependencies()

            # Initialize telemetry if configured
            if telemetry_config:
                setup_telemetry(telemetry_config)

            # Setup Prometheus server (skip during tests to avoid blocking)
            import os

            if not os.environ.get("TESTING"):
                try:
                    setup_prometheus_server(prometheus_port)
                except Exception as e:
                    # Log warning but don't fail initialization
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to setup Prometheus server on port {prometheus_port}: {e}"
                    )

            # Get monitoring service from DI container
            return create_monitoring_service()

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"DI initialization failed, falling back to manual wiring: {e}")

    # Fallback: Manual dependency wiring for backward compatibility
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
