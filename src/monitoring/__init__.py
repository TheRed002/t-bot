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

from .metrics import (
    ExchangeMetrics,
    MetricsCollector,
    RiskMetrics,
    SystemMetrics,
    TradingMetrics,
)
from .telemetry import (
    OpenTelemetryConfig,
    get_tracer,
    instrument_fastapi,
    setup_telemetry,
)

__all__ = [
    "ExchangeMetrics",
    "MetricsCollector",
    "OpenTelemetryConfig",
    "RiskMetrics",
    "SystemMetrics",
    "TradingMetrics",
    "get_tracer",
    "instrument_fastapi",
    "setup_telemetry",
]
