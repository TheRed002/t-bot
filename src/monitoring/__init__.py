"""
Monitoring infrastructure package for T-Bot Trading System.

This package provides comprehensive monitoring, alerting, and performance optimization
tools for the trading system including:

- P-030: Monitoring Infrastructure (Prometheus, Grafana, custom metrics)
- P-031: Alerting System (AlertManager, notifications, escalation)
- P-032: Performance Optimization (profiling, optimization, bottleneck detection)

Components:
    metrics: Prometheus metrics collection and custom trading metrics
    alerting: Alert rules, notification channels, escalation policies
    performance: Profiling tools, performance monitoring, optimization
    dashboards: Grafana dashboard configurations
    telemetry: OpenTelemetry instrumentation and tracing
"""

from .alerting import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationConfig,
    get_alert_manager,
    set_global_alert_manager,
)
from .metrics import (
    ExchangeMetrics,
    MetricsCollector,
    RiskMetrics,
    SystemMetrics,
    TradingMetrics,
    get_metrics_collector,
    set_metrics_collector,
)
from .performance import (
    PerformanceProfiler,
    get_performance_profiler,
    set_global_profiler,
)
from .telemetry import (
    OpenTelemetryConfig,
    get_tracer,
    get_trading_tracer,
    instrument_fastapi,
    set_global_trading_tracer,
    setup_telemetry,
)
from .trace_wrapper import Status, StatusCode, trace

__all__ = [
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "ExchangeMetrics",
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
    "get_alert_manager",
    "get_metrics_collector",
    "get_performance_profiler",
    "get_tracer",
    "get_trading_tracer",
    "instrument_fastapi",
    "set_global_alert_manager",
    "set_global_profiler",
    "set_global_trading_tracer",
    "set_metrics_collector",
    "setup_telemetry",
    "trace",
]
