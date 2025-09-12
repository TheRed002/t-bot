"""
Service interfaces for monitoring components.

This module defines the service interfaces following the service layer pattern.
It provides clean separation between service implementations and their dependencies.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from src.core.types import AlertSeverity

if TYPE_CHECKING:
    from src.monitoring.alerting import Alert, AlertRule, EscalationPolicy
    from src.monitoring.dashboards import Dashboard
    from src.monitoring.performance import LatencyStats, SystemResourceStats
    from src.monitoring.services import AlertRequest, MetricRequest


class MonitoringServiceInterface(ABC):
    """Interface for monitoring service operations."""

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start monitoring services."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        pass

    @abstractmethod
    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of all monitoring components."""
        pass


class AlertServiceInterface(ABC):
    """Interface for alert service operations."""

    @abstractmethod
    async def create_alert(self, request: "AlertRequest") -> str:
        """Create a new alert and return its fingerprint."""
        pass

    @abstractmethod
    async def resolve_alert(self, fingerprint: str) -> bool:
        """Resolve an active alert."""
        pass

    @abstractmethod
    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        pass

    @abstractmethod
    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list["Alert"]:
        """Get active alerts."""
        pass

    @abstractmethod
    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        pass

    @abstractmethod
    def add_rule(self, rule: "AlertRule") -> None:
        """Add an alert rule."""
        pass

    @abstractmethod
    def add_escalation_policy(self, policy: "EscalationPolicy") -> None:
        """Add an escalation policy."""
        pass


class MetricsServiceInterface(ABC):
    """Interface for metrics service operations."""

    @abstractmethod
    def record_counter(self, request: "MetricRequest") -> None:
        """Record a counter metric."""
        pass

    @abstractmethod
    def record_gauge(self, request: "MetricRequest") -> None:
        """Record a gauge metric."""
        pass

    @abstractmethod
    def record_histogram(self, request: "MetricRequest") -> None:
        """Record a histogram metric."""
        pass

    @abstractmethod
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        pass


class PerformanceServiceInterface(ABC):
    """Interface for performance monitoring service."""

    @abstractmethod
    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        pass

    @abstractmethod
    def record_order_execution(
        self,
        exchange: str,
        order_type: str,
        symbol: str,
        latency_ms: Decimal,
        fill_rate: Decimal,
        slippage_bps: Decimal,
    ) -> None:
        """Record order execution metrics."""
        pass

    @abstractmethod
    def record_market_data_processing(
        self,
        exchange: str,
        data_type: str,
        processing_time_ms: Decimal,
        message_count: int,
    ) -> None:
        """Record market data processing metrics."""
        pass

    @abstractmethod
    def get_latency_stats(self, metric_name: str) -> Optional["LatencyStats"]:
        """Get latency statistics for a metric."""
        pass

    @abstractmethod
    def get_system_resource_stats(self) -> Optional["SystemResourceStats"]:
        """Get system resource statistics."""
        pass


class DashboardServiceInterface(ABC):
    """Interface for dashboard management service."""

    @abstractmethod
    async def deploy_dashboard(self, dashboard: "Dashboard") -> bool:
        """Deploy a dashboard."""
        pass

    @abstractmethod
    async def deploy_all_dashboards(self) -> dict[str, bool]:
        """Deploy all dashboards."""
        pass

    @abstractmethod
    def export_dashboards_to_files(self, output_dir: str) -> None:
        """Export dashboards to files."""
        pass

    @abstractmethod
    def create_trading_overview_dashboard(self) -> "Dashboard":
        """Create trading overview dashboard."""
        pass

    @abstractmethod
    def create_system_performance_dashboard(self) -> "Dashboard":
        """Create system performance dashboard."""
        pass
