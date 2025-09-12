"""
Analytics Service Interfaces and Protocols.

This module defines the service layer interfaces for the analytics system,
ensuring proper abstraction and dependency inversion.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from src.analytics.types import (
    AnalyticsAlert,
    AnalyticsReport,
    OperationalMetrics,
    PortfolioMetrics,
    PositionMetrics,
    ReportType,
    RiskMetrics,
    StrategyMetrics,
)
from src.core.types import Order, Position, Trade


class AnalyticsServiceProtocol(Protocol):
    """Protocol defining the analytics service interface."""

    async def start(self) -> None:
        """Start the analytics service."""
        ...

    async def stop(self) -> None:
        """Stop the analytics service."""
        ...

    def update_position(self, position: Position) -> None:
        """Update position data."""
        ...

    def update_trade(self, trade: Trade) -> None:
        """Update trade data."""
        ...

    def update_order(self, order: Order) -> None:
        """Update order data."""
        ...

    async def get_portfolio_metrics(self) -> PortfolioMetrics | None:
        """Get current portfolio metrics."""
        ...

    async def get_risk_metrics(self) -> RiskMetrics:
        """Get risk metrics."""
        ...

    def record_risk_metrics(self, risk_metrics) -> None:
        """Record risk metrics for analytics."""
        ...

    def record_risk_alert(self, alert) -> None:
        """Record risk alert for analytics."""
        ...


class AlertServiceProtocol(Protocol):
    """Protocol for alert management service."""

    async def generate_alert(
        self,
        rule_name: str,
        severity: str,
        message: str,
        labels: dict[str, str],
        annotations: dict[str, str],
        **kwargs,
    ) -> AnalyticsAlert:
        """Generate a new alert with parameters aligned to monitoring AlertRequest."""
        ...

    def get_active_alerts(self) -> list[AnalyticsAlert]:
        """Get active alerts."""
        ...

    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert using consistent fingerprint parameter."""
        ...


class RiskServiceProtocol(Protocol):
    """Protocol for risk calculation service."""

    async def calculate_var(
        self, confidence_level: Decimal, time_horizon: int, method: str
    ) -> dict[str, Decimal]:
        """Calculate Value at Risk."""
        ...

    async def run_stress_test(
        self, scenario_name: str, scenario_params: dict[str, Any]
    ) -> dict[str, Decimal]:
        """Run stress test scenario."""
        ...

    async def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics."""
        ...


class PortfolioServiceProtocol(Protocol):
    """Protocol for portfolio analytics service."""

    def update_position(self, position: Position) -> None:
        """Update position data."""
        ...

    def update_trade(self, trade: Trade) -> None:
        """Update trade data."""
        ...

    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio metrics."""
        ...

    async def get_portfolio_composition(self) -> dict[str, Any]:
        """Get portfolio composition analysis."""
        ...

    async def calculate_correlation_matrix(self) -> Any | None:
        """Get correlation matrix."""
        ...


class ReportingServiceProtocol(Protocol):
    """Protocol for reporting service."""

    async def generate_performance_report(
        self,
        report_type: ReportType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AnalyticsReport:
        """Generate performance report."""
        ...


class ExportServiceProtocol(Protocol):
    """Protocol for data export service."""

    async def export_portfolio_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """Export portfolio data."""
        ...

    async def export_risk_data(self, format: str = "json", include_metadata: bool = True) -> str:
        """Export risk data."""
        ...


class OperationalServiceProtocol(Protocol):
    """Protocol for operational analytics service."""

    async def get_operational_metrics(self) -> OperationalMetrics:
        """Get operational metrics."""
        ...

    def record_strategy_event(
        self, strategy_name: str, event_type: str, success: bool = True, **kwargs
    ) -> None:
        """Record strategy event."""
        ...

    def record_system_error(
        self, component: str, error_type: str, error_message: str, **kwargs
    ) -> None:
        """Record system error."""
        ...


class RealtimeAnalyticsServiceProtocol(Protocol):
    """Protocol for realtime analytics service."""

    async def start(self) -> None:
        """Start realtime analytics."""
        ...

    async def stop(self) -> None:
        """Stop realtime analytics."""
        ...

    def update_position(self, position: Position) -> None:
        """Update position data."""
        ...

    def update_trade(self, trade: Trade) -> None:
        """Update trade data."""
        ...

    def update_order(self, order: Order) -> None:
        """Update order data."""
        ...

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price data."""
        ...

    async def get_portfolio_metrics(self) -> PortfolioMetrics | None:
        """Get portfolio metrics."""
        ...

    async def get_position_metrics(self, symbol: str | None = None) -> list[PositionMetrics]:
        """Get position metrics."""
        ...

    async def get_strategy_metrics(self, strategy: str | None = None) -> list[StrategyMetrics]:
        """Get strategy metrics."""
        ...


class AnalyticsDataRepository(ABC):
    """Abstract base class for analytics data repository."""

    @abstractmethod
    async def store_portfolio_metrics(self, metrics: PortfolioMetrics) -> None:
        """Store portfolio metrics."""
        pass

    @abstractmethod
    async def store_position_metrics(self, metrics: list[PositionMetrics]) -> None:
        """Store position metrics."""
        pass

    @abstractmethod
    async def store_risk_metrics(self, metrics: RiskMetrics) -> None:
        """Store risk metrics."""
        pass

    @abstractmethod
    async def get_historical_portfolio_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> list[PortfolioMetrics]:
        """Get historical portfolio metrics."""
        pass


class MetricsCalculationService(ABC):
    """Abstract base class for metrics calculation service."""

    @abstractmethod
    async def calculate_portfolio_metrics(
        self, positions: dict[str, Position], prices: dict[str, Decimal]
    ) -> PortfolioMetrics:
        """Calculate portfolio metrics from positions and prices."""
        pass

    @abstractmethod
    async def calculate_position_metrics(
        self, position: Position, current_price: Decimal
    ) -> PositionMetrics:
        """Calculate position metrics."""
        pass

    @abstractmethod
    async def calculate_strategy_metrics(
        self, strategy_name: str, positions: list[Position], trades: list[Trade]
    ) -> StrategyMetrics:
        """Calculate strategy performance metrics."""
        pass


class RiskCalculationService(ABC):
    """Abstract base class for risk calculation service."""

    @abstractmethod
    async def calculate_portfolio_var(
        self,
        positions: dict[str, Position],
        confidence_level: Decimal,
        time_horizon: int,
        method: str = "historical",
    ) -> dict[str, Decimal]:
        """Calculate portfolio Value at Risk."""
        pass

    @abstractmethod
    async def calculate_risk_metrics(
        self, positions: dict[str, Position], price_history: dict[str, list[Decimal]]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        pass


@runtime_checkable
class AnalyticsServiceFactoryProtocol(Protocol):
    """Protocol defining the analytics service factory interface."""

    def create_analytics_service(self, config=None) -> AnalyticsServiceProtocol:
        """Create main analytics service."""
        ...

    def create_portfolio_service(self) -> PortfolioServiceProtocol:
        """Create portfolio analytics service."""
        ...

    def create_risk_service(self) -> RiskServiceProtocol:
        """Create risk monitoring service."""
        ...

    def create_reporting_service(self) -> ReportingServiceProtocol:
        """Create performance reporting service."""
        ...

    def create_operational_service(self) -> OperationalServiceProtocol:
        """Create operational analytics service."""
        ...

    def create_alert_service(self) -> AlertServiceProtocol:
        """Create alert management service."""
        ...

    def create_export_service(self) -> ExportServiceProtocol:
        """Create data export service."""
        ...

    def create_realtime_analytics_service(self) -> RealtimeAnalyticsServiceProtocol:
        """Create realtime analytics service."""
        ...
