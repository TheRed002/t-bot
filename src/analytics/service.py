"""
Analytics Service Integration Layer.

This module provides the main service interface for the comprehensive analytics system,
integrating real-time trading analytics, portfolio analytics, performance reporting,
risk monitoring, and operational analytics.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.analytics.events import (
    AlertEventHandler,
    PortfolioEventHandler,
    RiskEventHandler,
    get_event_bus,
    publish_order_updated,
    publish_position_updated,
    publish_price_updated,
    publish_trade_executed,
)
from src.analytics.interfaces import (
    AlertServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RealtimeAnalyticsServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.types import (
    AnalyticsConfiguration,
    AnalyticsReport,
    BenchmarkData,
    OperationalMetrics,
    PortfolioMetrics,
    PositionMetrics,
    ReportType,
    RiskMetrics,
    StrategyMetrics,
)
from src.analytics.utils.error_handling import AnalyticsErrorHandler
from src.analytics.utils.metrics_helpers import MetricsHelper
from src.analytics.utils.task_management import TaskManager
from src.core.base.component import BaseComponent
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import Order, Position, Trade
from src.monitoring.metrics import get_metrics_collector
from src.utils.constants import (
    ANALYTICS_PERFORMANCE_THRESHOLDS,
    ANALYTICS_REPORT_VERSION,
    ANALYTICS_TIMING,
    VAR_CALCULATION_DEFAULTS,
)
from src.utils.datetime_utils import get_current_utc_timestamp


class AnalyticsService(BaseComponent):
    """
    Comprehensive analytics service orchestrator for institutional trading operations.

    This service provides a unified interface to all enhanced analytics capabilities:
    - Real-time trading analytics with WebSocket streaming and advanced VaR
    - Advanced portfolio analytics with Modern Portfolio Theory and factor models
    - Institutional-grade performance reporting with attribution analysis
    - Comprehensive risk monitoring with stress testing and scenario analysis
    - Multi-format data export with external system integrations
    - Operational analytics and health monitoring

    Enhanced Features:
    - Institutional-grade analytics accuracy and regulatory compliance
    - Real-time WebSocket streaming and live dashboards
    - Advanced factor models (Fama-French, Black-Litterman, Risk Parity)
    - Multi-methodology VaR calculations and stress testing
    - Comprehensive attribution analysis (Brinson, factor-based, transaction-level)
    - External integrations (Prometheus, InfluxDB, Kafka, REST APIs)
    - Regulatory reporting (CCAR, Basel III, MiFID II)
    - Performance persistence and skill analysis
    - Real-time calculation and monitoring
    - Comprehensive alert and threshold management
    - Export capabilities for external reporting
    - Integration with existing trading infrastructure
    """

    def __init__(
        self,
        config: AnalyticsConfiguration | None = None,
        realtime_analytics: RealtimeAnalyticsServiceProtocol | None = None,
        portfolio_service: PortfolioServiceProtocol | None = None,
        reporting_service: ReportingServiceProtocol | None = None,
        risk_service: RiskServiceProtocol | None = None,
        operational_service: OperationalServiceProtocol | None = None,
        alert_service: AlertServiceProtocol | None = None,
        export_service: ExportServiceProtocol | None = None,
        metrics_collector=None,
    ):
        """
        Initialize analytics service with dependency injection.

        Args:
            config: Analytics configuration (uses defaults if not provided)
            realtime_analytics: Realtime analytics engine (injected dependency)
            portfolio_service: Portfolio analytics service (injected dependency)
            reporting_service: Performance reporting service (injected dependency)
            risk_service: Risk monitoring service (injected dependency)
            operational_service: Operational analytics service (injected dependency)
            alert_service: Alert management service (injected dependency)
            export_service: Data export service (injected dependency)
            metrics_collector: Metrics collector (injected dependency)
        """
        super().__init__()
        self.config = config or AnalyticsConfiguration()
        self.metrics_collector = metrics_collector or get_metrics_collector()

        self.error_handler = AnalyticsErrorHandler()
        self.metrics_helper = MetricsHelper()
        self.task_manager = TaskManager()

        self.metrics_helper.setup_metrics_collector(self.metrics_collector)

        required_services = {
            "realtime_analytics": realtime_analytics,
            "portfolio_service": portfolio_service,
            "reporting_service": reporting_service,
            "risk_service": risk_service,
            "operational_service": operational_service,
            "alert_service": alert_service,
            "export_service": export_service,
        }

        for service_name, service_instance in required_services.items():
            if not service_instance:
                raise ComponentError(
                    f"{service_name} must be injected via dependency injection",
                    component="AnalyticsService",
                    operation="__init__",
                    context={"missing_service": service_name},
                )

        self.realtime_analytics = realtime_analytics
        self.portfolio_service = portfolio_service
        self.reporting_service = reporting_service
        self.risk_service = risk_service
        self.operational_service = operational_service
        self.alert_service = alert_service
        self.export_service = export_service

        self._running = False
        self._background_tasks: set = set()

        self.event_bus = get_event_bus()
        self._event_handlers: list[Callable] = []

        self._cache_enabled = True
        self._cache_ttl = timedelta(seconds=self.config.cache_ttl_seconds)
        self._cached_metrics: dict[str, dict[str, Any]] = {}

        self._setup_event_handlers()

        self.logger.info("AnalyticsService initialized with event-driven architecture")

    async def start(self) -> None:
        """Start the analytics service and all sub-engines."""
        if self._running:
            self.logger.warning("Analytics service already running")
            return

        self._running = True

        try:
            await self.event_bus.start()

            if self.realtime_analytics and hasattr(self.realtime_analytics, "start"):
                await self.realtime_analytics.start()
            if self.risk_service and hasattr(self.risk_service, "start"):
                await self.risk_service.start()
            if self.operational_service and hasattr(self.operational_service, "start"):
                await self.operational_service.start()
            if self.alert_service and hasattr(self.alert_service, "start"):
                await self.alert_service.start()

            tasks = [
                self._periodic_reporting_loop(),
                self._cache_maintenance_loop(),
            ]

            for task_coro in tasks:
                task = asyncio.create_task(task_coro)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            self.logger.info("Analytics service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start analytics service: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the analytics service and all sub-engines."""
        self._running = False

        try:
            # Stop background tasks
            for task in self._background_tasks:
                task.cancel()

            if self._background_tasks:
                # Cancel all tasks and wait for them to complete
                tasks_to_wait = list(self._background_tasks)
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
                self._background_tasks.clear()

            # Stop analytics engines
            if self.realtime_analytics and hasattr(self.realtime_analytics, "stop"):
                await self.realtime_analytics.stop()
            if self.risk_service and hasattr(self.risk_service, "stop"):
                await self.risk_service.stop()
            if self.operational_service and hasattr(self.operational_service, "stop"):
                await self.operational_service.stop()
            if self.alert_service and hasattr(self.alert_service, "stop"):
                await self.alert_service.stop()

            # Stop event bus last
            await self.event_bus.stop()

            self.logger.info("Analytics service stopped")

        except Exception as e:
            self.logger.error(f"Error stopping analytics service: {e}")

    def __del__(self):
        """Cleanup any remaining background tasks on garbage collection."""
        if hasattr(self, "_background_tasks") and self._background_tasks:
            # Cancel any remaining tasks to avoid warnings
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            self._background_tasks.clear()

    def update_position(self, position: Position) -> None:
        """
        Update position data across all analytics engines using consistent patterns.

        Args:
            position: Position update
        """
        try:
            # Use consistent async task creation pattern with proper task management
            if self._running:
                try:
                    # Check if event loop is running
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self._update_position_async(position))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    # No event loop running, skip async processing
                    self.logger.debug("No event loop running, skipping async position update")

        except Exception as e:
            self.error_handler.handle_analytics_error(
                e,
                "update_position",
                context={
                    "symbol": position.symbol,
                    "exchange": position.exchange,
                    "position_id": getattr(position, "position_id", "unknown"),
                },
            )

    async def _update_position_async(self, position: Position) -> None:
        """Async position update using event-driven pattern."""
        try:
            # Update realtime analytics directly (high frequency)
            if self.realtime_analytics:
                await self.realtime_analytics.update_position(position)

            # Update portfolio service directly
            if self.portfolio_service:
                await self.portfolio_service.update_position(position)

            # Publish event for other services to handle asynchronously
            await publish_position_updated(position, "AnalyticsService")

            self.logger.debug(f"Updated position: {position.symbol}")

        except Exception as e:
            self.error_handler.handle_analytics_error(
                e, "_update_position_async", context={"symbol": position.symbol}
            )

    def update_trade(self, trade: Trade) -> None:
        """
        Update trade data for analytics using consistent async patterns.

        Args:
            trade: Trade data
        """
        try:
            # Proper task management for trade updates
            if self._running:
                try:
                    # Check if event loop is running
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(self._update_trade_async(trade))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    # No event loop running, skip async processing
                    self.logger.debug("No event loop running, skipping async trade update")

        except Exception as e:
            raise ComponentError(
                f"Trade update failed for {trade.trade_id}",
                component="AnalyticsService",
                operation="update_trade",
                context={
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "exchange": trade.exchange,
                },
            ) from e

    async def _update_trade_async(self, trade: Trade) -> None:
        """Async trade update using event-driven pattern."""
        try:
            # Update realtime analytics directly (high frequency)
            if self.realtime_analytics:
                await self.realtime_analytics.update_trade(trade)

            # Update portfolio service directly
            if self.portfolio_service:
                await self.portfolio_service.update_trade(trade)

            # Publish event for other services to handle asynchronously
            await publish_trade_executed(trade, "AnalyticsService")

            # Record transaction cost data if available
            if (
                hasattr(trade, "fee")
                and trade.fee > 0
                and hasattr(self.reporting_service, "add_transaction_cost")
            ):
                if self.reporting_service:
                    await self.reporting_service.add_transaction_cost(
                        timestamp=trade.timestamp,
                        symbol=trade.symbol,
                        cost_type="commission",
                        cost_amount=trade.fee,
                        trade_value=trade.quantity * trade.price,
                    )

            self.logger.debug(f"Updated trade: {trade.trade_id}")

        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            raise

    def update_order(self, order: Order) -> None:
        """
        Update order data for analytics.

        Args:
            order: Order data
        """
        try:
            # Update realtime analytics directly (high frequency)
            if self.realtime_analytics:
                self.realtime_analytics.update_order(order)

            # Publish event for other services to handle asynchronously with task management
            if self._running:
                try:
                    # Check if event loop is running
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(publish_order_updated(order, "AnalyticsService"))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    # No event loop running, skip async publishing
                    self.logger.debug("No event loop running, skipping order update event")

            # Record order event for operational analytics
            if hasattr(self.operational_service, "record_order_update"):
                if self.operational_service:
                    self.operational_service.record_order_update(order)

            self.logger.debug(f"Updated order: {order.order_id}")

        except Exception as e:
            self.logger.error(f"Error updating order: {e}")
            raise

    def update_price(self, symbol: str, price: Decimal, timestamp: datetime | None = None) -> None:
        """
        Update price data for real-time analytics.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp
        """
        try:
            timestamp = timestamp or get_current_utc_timestamp()

            # Update realtime analytics directly (high frequency)
            if self.realtime_analytics:
                self.realtime_analytics.update_price(symbol, price)

            # Publish event for other services to handle asynchronously with task management
            if self._running:
                try:
                    # Check if event loop is running
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        publish_price_updated(symbol, price, timestamp, "AnalyticsService")
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                except RuntimeError:
                    # No event loop running, skip async publishing
                    self.logger.debug("No event loop running, skipping price update event")

            self.logger.debug(f"Updated price: {symbol} = {price}")

        except Exception as e:
            self.logger.error(f"Error updating price: {e}")
            raise

    def record_risk_metrics(self, risk_metrics) -> None:
        """Record risk metrics for analytics."""
        try:
            if self.risk_service:
                # Store risk metrics for analytics
                self.risk_service.store_risk_metrics(risk_metrics)
            self.logger.debug("Risk metrics recorded for analytics")
        except Exception as e:
            self.logger.error(f"Failed to record risk metrics: {e}")

    def record_risk_alert(self, alert) -> None:
        """Record risk alert for analytics."""
        try:
            if self.alert_service:
                # Store risk alert for analytics
                self.alert_service.store_risk_alert(alert)
            self.logger.debug("Risk alert recorded for analytics")
        except Exception as e:
            self.logger.error(f"Failed to record risk alert: {e}")

    def update_benchmark_data(self, benchmark_name: str, data: BenchmarkData) -> None:
        """
        Update benchmark data for performance comparison.

        Args:
            benchmark_name: Benchmark name
            data: Benchmark data
        """
        try:
            if hasattr(self.portfolio_service, "update_benchmark_data"):
                if self.portfolio_service:
                    self.portfolio_service.update_benchmark_data(benchmark_name, data)
            self.logger.debug(f"Updated benchmark: {benchmark_name}")

        except Exception as e:
            self.logger.error(f"Error updating benchmark data: {e}")
            raise

    async def get_portfolio_metrics(self) -> PortfolioMetrics | None:
        """
        Get current portfolio metrics.

        Returns:
            Portfolio metrics or None if not available
        """
        try:
            cache_key = "portfolio_metrics"

            # Check cache first
            if self._cache_enabled:
                cached = self._get_cached_result(cache_key)
                if cached is not None:
                    return cached

            if not self.realtime_analytics:
                return {}
            metrics = await self.realtime_analytics.get_portfolio_metrics()

            # Cache result
            if self._cache_enabled and metrics:
                self._cache_result(cache_key, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting portfolio metrics: {e}")
            return None

    async def get_position_metrics(self, symbol: str | None = None) -> list[PositionMetrics]:
        """
        Get position metrics.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position metrics
        """
        try:
            if not self.realtime_analytics:
                return None
            return await self.realtime_analytics.get_position_metrics(symbol)

        except Exception as e:
            self.logger.error(f"Error getting position metrics: {e}")
            return []

    async def get_strategy_metrics(self, strategy: str | None = None) -> list[StrategyMetrics]:
        """
        Get strategy performance metrics.

        Args:
            strategy: Optional strategy filter

        Returns:
            List of strategy metrics
        """
        try:
            if not self.realtime_analytics:
                return {}
            return await self.realtime_analytics.get_strategy_metrics(strategy)

        except Exception as e:
            self.logger.error(f"Error getting strategy metrics: {e}")
            return []

    async def get_risk_metrics(self) -> RiskMetrics:
        """
        Get comprehensive risk metrics.

        Returns:
            Risk metrics
        """
        try:
            cache_key = "risk_metrics"

            # Check cache first
            if self._cache_enabled:
                cached = self._get_cached_result(cache_key)
                if cached is not None:
                    return cached

            if not self.risk_service:
                return {}
            metrics = await self.risk_service.get_risk_metrics()

            # Cache result
            if self._cache_enabled:
                self._cache_result(cache_key, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return RiskMetrics(timestamp=get_current_utc_timestamp())

    async def get_operational_metrics(self) -> OperationalMetrics:
        """
        Get operational metrics.

        Returns:
            Operational metrics
        """
        try:
            if not self.operational_service:
                return self._default_operational_metrics()
            return await self.operational_service.get_operational_metrics()

        except Exception as e:
            self.logger.error(f"Error getting operational metrics: {e}")
            return OperationalMetrics(timestamp=get_current_utc_timestamp())

    # Risk Analysis Methods

    async def calculate_var(
        self,
        confidence_level: float = VAR_CALCULATION_DEFAULTS["confidence_level"],
        time_horizon: int = VAR_CALCULATION_DEFAULTS["time_horizon"],
        method: str = VAR_CALCULATION_DEFAULTS["method"],
    ) -> dict[str, float]:
        """
        Calculate Value at Risk.

        Args:
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
            method: Calculation method

        Returns:
            VaR calculation results
        """
        try:
            if not self.risk_service:
                return Decimal("0")
            return await self.risk_service.calculate_var(confidence_level, time_horizon, method)

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            raise

    async def run_stress_test(
        self, scenario_name: str, scenario_params: dict[str, Any]
    ) -> dict[str, float]:
        """
        Run stress test scenario.

        Args:
            scenario_name: Stress test scenario name
            scenario_params: Scenario parameters

        Returns:
            Stress test results
        """
        try:
            if not self.risk_service:
                return {}
            return await self.risk_service.run_stress_test(scenario_name, scenario_params)

        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            raise

    async def get_portfolio_composition(self) -> dict[str, Any]:
        """
        Get portfolio composition analysis.

        Returns:
            Portfolio composition metrics
        """
        try:
            if not self.portfolio_service:
                return {}
            return await self.portfolio_service.get_portfolio_composition()

        except Exception as e:
            self.logger.error(f"Error getting portfolio composition: {e}")
            raise

    async def get_correlation_matrix(self) -> Any | None:
        """
        Get correlation matrix for portfolio positions.

        Returns:
            Correlation matrix DataFrame or None
        """
        try:
            if not self.portfolio_service:
                return {}
            return await self.portfolio_service.calculate_correlation_matrix()

        except Exception as e:
            self.logger.error(f"Error getting correlation matrix: {e}")
            return None

    # Reporting Methods

    async def generate_performance_report(
        self,
        report_type: ReportType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AnalyticsReport:
        """
        Generate performance report.

        Args:
            report_type: Type of report to generate
            start_date: Report start date
            end_date: Report end date

        Returns:
            Complete analytics report
        """
        try:
            if not self.reporting_service:
                return {}
            return await self.reporting_service.generate_performance_report(
                report_type, start_date, end_date
            )

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise

    async def generate_risk_report(self) -> dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Risk report with all metrics and analysis
        """
        try:
            if hasattr(self.risk_service, "generate_risk_report"):
                if not self.risk_service:
                    return {}
                return await self.risk_service.generate_risk_report()
            else:
                # Fallback to basic risk metrics
                if not self.risk_service:
                    return {"risk_metrics": {}}
                return {"risk_metrics": await self.risk_service.get_risk_metrics()}

        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            raise

    async def generate_health_report(self) -> dict[str, Any]:
        """
        Generate system health report.

        Returns:
            System health report with status and recommendations
        """
        try:
            if hasattr(self.operational_service, "generate_health_report"):
                if not self.operational_service:
                    return {}
                return await self.operational_service.generate_health_report()
            else:
                # Fallback to basic operational metrics
                return {
                    "operational_metrics": await self.operational_service.get_operational_metrics()
                    if self.operational_service
                    else self._default_operational_metrics()
                }

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            raise

    # Alert and Monitoring Methods

    async def get_active_alerts(self) -> list[Any]:
        """
        Get all active analytics alerts.

        Returns:
            List of active alerts from all engines
        """
        try:
            alerts = []

            # Get alerts from real-time analytics
            if self.realtime_analytics:
                realtime_alerts = await self.realtime_analytics.get_active_alerts()
                alerts.extend(realtime_alerts)

            # Get alerts from alert manager
            if self.alert_service and hasattr(self.alert_service, "get_active_alerts"):
                alert_service_alerts = self.alert_service.get_active_alerts()
                alerts.extend(alert_service_alerts)

            return alerts

        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []

    # Export and Integration Methods

    async def export_metrics(self, format: str = "json") -> dict[str, Any]:
        """
        Export comprehensive metrics for external systems.

        Args:
            format: Export format ('json', 'csv', 'prometheus')

        Returns:
            Exported metrics data
        """
        try:
            export_data = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "portfolio_metrics": await self.get_portfolio_metrics(),
                "risk_metrics": await self.get_risk_metrics(),
                "operational_metrics": await self.get_operational_metrics(),
                "position_metrics": await self.get_position_metrics(),
                "strategy_metrics": await self.get_strategy_metrics(),
                "active_alerts": await self.get_active_alerts(),
            }

            # Convert Pydantic models to dict for serialization
            def convert_for_export(obj):
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                elif hasattr(obj, "dict"):
                    return obj.dict()
                elif isinstance(obj, list):
                    return [convert_for_export(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_for_export(v) for k, v in obj.items()}
                else:
                    return obj

            return convert_for_export(export_data)

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            raise

    async def export_portfolio_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """
        Export portfolio data in specified format.

        Args:
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported portfolio data
        """
        try:
            portfolio_metrics = await self.get_portfolio_metrics()
            if portfolio_metrics:
                return await self.export_service.export_portfolio_data(format, include_metadata)

            return ""

        except Exception as e:
            self.logger.error(f"Error exporting portfolio data: {e}")
            return ""

    async def export_risk_data(self, format: str = "json", include_metadata: bool = True) -> str:
        """
        Export risk data in specified format.

        Args:
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported risk data
        """
        try:
            return await self.export_service.export_risk_data(format, include_metadata)

        except Exception as e:
            self.logger.error(f"Error exporting risk data: {e}")
            return ""

    async def export_position_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """
        Export position data in specified format.

        Args:
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported position data
        """
        try:
            position_metrics = await self.get_position_metrics()
            if position_metrics:
                return await self.export_service.export_position_metrics(
                    position_metrics, format, include_metadata
                )
            return ""

        except Exception as e:
            self.logger.error(f"Error exporting position data: {e}")
            return ""

    async def export_strategy_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """
        Export strategy data in specified format.

        Args:
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported strategy data
        """
        try:
            strategy_metrics = await self.get_strategy_metrics()
            if strategy_metrics:
                return await self.export_service.export_strategy_metrics(
                    strategy_metrics, format, include_metadata
                )
            return ""

        except Exception as e:
            self.logger.error(f"Error exporting strategy data: {e}")
            return ""

    async def export_operational_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """
        Export operational data in specified format.

        Args:
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported operational data
        """
        try:
            operational_metrics = await self.get_operational_metrics()
            return await self.export_service.export_operational_metrics(
                operational_metrics, format, include_metadata
            )

        except Exception as e:
            self.logger.error(f"Error exporting operational data: {e}")
            return ""

    async def export_complete_report_data(
        self, report: AnalyticsReport, format: str = "json", include_charts: bool = False
    ) -> str:
        """
        Export complete analytics report.

        Args:
            report: Analytics report to export
            format: Export format ('json', 'excel')
            include_charts: Whether to include chart data

        Returns:
            Exported report data
        """
        try:
            return await self.export_service.export_complete_report(report, format, include_charts)

        except Exception as e:
            self.logger.error(f"Error exporting complete report: {e}")
            return ""

    def get_export_statistics(self) -> dict[str, Any]:
        """
        Get export usage statistics.

        Returns:
            Export statistics from data exporter
        """
        if hasattr(self.export_service, "get_export_statistics"):
            return self.export_service.get_export_statistics()
        return {}

    # Alert Management Methods

    def add_alert_rule(self, rule) -> None:
        """Add custom alert rule."""
        if hasattr(self.alert_service, "add_alert_rule"):
            self.alert_service.add_alert_rule(rule)

    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove alert rule."""
        if hasattr(self.alert_service, "remove_alert_rule"):
            self.alert_service.remove_alert_rule(rule_id)

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        return await self.alert_service.acknowledge_alert(alert_id, acknowledged_by)

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_note: str | None = None
    ) -> bool:
        """Resolve an alert."""
        if hasattr(self.alert_service, "resolve_alert"):
            return await self.alert_service.resolve_alert(alert_id, resolved_by, resolution_note)
        return False

    def get_alert_statistics(self, period_hours: int = 24) -> dict[str, Any]:
        """Get alert statistics."""
        if hasattr(self.alert_service, "get_alert_statistics"):
            return self.alert_service.get_alert_statistics(period_hours)
        return {}

    # Configuration and Management Methods

    def update_configuration(self, config: AnalyticsConfiguration) -> None:
        """
        Update analytics configuration.

        Args:
            config: New configuration
        """
        try:
            self.config = config

            # Update configurations in sub-engines
            self.realtime_analytics.config = config
            if hasattr(self.portfolio_service, "config"):
                self.portfolio_service.config = config
            if hasattr(self.reporting_service, "config"):
                self.reporting_service.config = config
            if hasattr(self.risk_service, "config"):
                self.risk_service.config = config
            if hasattr(self.operational_service, "config"):
                self.operational_service.config = config
            if hasattr(self.alert_service, "config"):
                self.alert_service.config = config

            self.logger.info("Analytics configuration updated")

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")

    def get_service_status(self) -> dict[str, Any]:
        """
        Get analytics service status.

        Returns:
            Service status information
        """
        return {
            "running": self._running,
            "engines": {
                "realtime_analytics": "running" if self._running else "stopped",
                "portfolio_analytics": "running" if self._running else "stopped",
                "performance_reporter": "running" if self._running else "stopped",
                "risk_monitor": "running" if self._running else "stopped",
                "operational_analytics": "running" if self._running else "stopped",
                "alert_manager": "running" if self._running else "stopped",
                "data_exporter": "ready",
            },
            "background_tasks": len(self._background_tasks),
            "cache_enabled": self._cache_enabled,
            "configuration": {
                "calculation_frequency": self.config.calculation_frequency.value,
                "reporting_frequency": self.config.reporting_frequency.value,
                "risk_free_rate": str(self.config.risk_free_rate),
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
            },
        }

    # Private Methods

    async def _periodic_reporting_loop(self) -> None:
        """Background loop for periodic report generation."""
        while self._running:
            try:
                # Generate daily report if it's a new day
                now = get_current_utc_timestamp()
                if now.hour == 0 and now.minute < 5:  # Run at start of day
                    await self.generate_performance_report(ReportType.DAILY_PERFORMANCE)

                # Wait before next check
                await asyncio.sleep(ANALYTICS_TIMING["periodic_report_check_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic reporting loop: {e}")
                await asyncio.sleep(ANALYTICS_TIMING["periodic_report_check_interval"])

    async def _cache_maintenance_loop(self) -> None:
        """Background loop for cache maintenance."""
        while self._running:
            try:
                # Clean expired cache entries
                now = get_current_utc_timestamp()
                expired_keys = []

                for key, data in self._cached_metrics.items():
                    if now - data["timestamp"] > self._cache_ttl:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self._cached_metrics[key]

                # Wait before next cleanup
                await asyncio.sleep(ANALYTICS_TIMING["cache_cleanup_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache maintenance loop: {e}")
                await asyncio.sleep(ANALYTICS_TIMING["cache_cleanup_interval"])

    def _get_cached_result(self, cache_key: str) -> Any | None:
        """Get cached result if valid."""
        if not self._cache_enabled or cache_key not in self._cached_metrics:
            return None

        cached_data = self._cached_metrics[cache_key]
        now = get_current_utc_timestamp()

        if now - cached_data["timestamp"] > self._cache_ttl:
            del self._cached_metrics[cache_key]
            return None

        return cached_data["result"]

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result with timestamp."""
        if not self._cache_enabled:
            return

        self._cached_metrics[cache_key] = {
            "timestamp": get_current_utc_timestamp(),
            "result": result,
        }

    # Event Recording Methods for Operational Analytics

    def record_strategy_event(
        self,
        strategy_name: str,
        event_type: str,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """
        Record strategy event for operational tracking.

        Args:
            strategy_name: Strategy name
            event_type: Event type
            success: Whether event was successful
            error_message: Error message if not successful

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If event recording fails
        """
        # Validate parameters

        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValidationError(
                "Invalid strategy_name parameter",
                field_name="strategy_name",
                field_value=strategy_name,
                expected_type="non-empty str",
            )

        if not isinstance(event_type, str) or not event_type:
            raise ValidationError(
                "Invalid event_type parameter",
                field_name="event_type",
                field_value=event_type,
                expected_type="non-empty str",
            )

        if not isinstance(success, bool):
            raise ValidationError(
                "Invalid success parameter",
                field_name="success",
                field_value=success,
                expected_type="bool",
            )

        try:
            self.operational_service.record_strategy_event(
                strategy_name, event_type, success, error_message=error_message
            )

        except ValidationError:
            raise
        except Exception as e:
            raise ComponentError(
                f"Failed to record strategy event: {e}",
                component="AnalyticsService",
                operation="record_strategy_event",
                context={
                    "strategy_name": strategy_name,
                    "event_type": event_type,
                    "success": success,
                    "error_message": error_message,
                },
            ) from e

    def record_market_data_event(
        self,
        exchange: str,
        symbol: str,
        event_type: str,
        latency_ms: float | None = None,
        success: bool = True,
    ) -> None:
        """
        Record market data event for operational tracking.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            event_type: Event type
            latency_ms: Data latency in milliseconds
            success: Whether event was successful
        """
        try:
            if hasattr(self.operational_service, "record_market_data_event"):
                self.operational_service.record_market_data_event(
                    exchange=exchange,
                    symbol=symbol,
                    event_type=event_type,
                    latency_ms=latency_ms,
                    success=success,
                )

        except Exception as e:
            self.logger.error(f"Error recording market data event: {e}")

    def record_system_error(
        self, component: str, error_type: str, error_message: str, severity: str = "low"
    ) -> None:
        """
        Record system error for tracking.

        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            severity: Error severity level

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If error recording fails
        """
        # Validate parameters

        if not isinstance(component, str) or not component:
            raise ValidationError(
                "Invalid component parameter",
                field_name="component",
                field_value=component,
                expected_type="non-empty str",
            )

        if not isinstance(error_type, str) or not error_type:
            raise ValidationError(
                "Invalid error_type parameter",
                field_name="error_type",
                field_value=error_type,
                expected_type="non-empty str",
            )

        valid_severities = ["low", "medium", "high", "critical", "info"]
        if severity not in valid_severities:
            raise ValidationError(
                "Invalid severity parameter",
                field_name="severity",
                field_value=severity,
                validation_rule=f"must be one of {valid_severities}",
            )

        try:
            self.operational_service.record_system_error(
                component, error_type, error_message, severity=severity
            )

        except ValidationError:
            raise
        except Exception as e:
            raise ComponentError(
                f"Failed to record system error: {e}",
                component="AnalyticsService",
                operation="record_system_error",
                context={
                    "component": component,
                    "error_type": error_type,
                    "severity": severity,
                    "error_message_length": len(error_message) if error_message else 0,
                },
            ) from e

    async def record_api_call(
        self,
        service: str,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        success: bool = True,
    ) -> None:
        """
        Record API call metrics.

        Args:
            service: Service name
            endpoint: API endpoint
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            success: Whether call was successful
        """
        try:
            if hasattr(self.operational_service, "record_api_call"):
                if self.operational_service:
                    await self.operational_service.record_api_call(
                        service=service,
                        endpoint=endpoint,
                        response_time_ms=response_time_ms,
                        status_code=status_code,
                        success=success,
                    )

        except Exception as e:
            self.logger.error(f"Error recording API call: {e}")

    # Enhanced Orchestration Methods

    async def generate_comprehensive_analytics_dashboard(self) -> dict[str, Any]:
        """
        Generate comprehensive real-time analytics dashboard combining all components.

        Returns:
            Complete dashboard data with all analytics components
        """
        try:
            # Get data from all enhanced analytics engines
            realtime_dashboard = await self.realtime_analytics.generate_real_time_dashboard_data()
            portfolio_report = (
                await self.portfolio_service.generate_institutional_analytics_report()
            )
            risk_dashboard = await self.risk_service.create_real_time_risk_dashboard()
            performance_report = (
                await self.reporting_service.generate_comprehensive_institutional_report()
            )
            operational_health = await self.operational_service.generate_system_health_dashboard()

            # Combine into unified dashboard
            comprehensive_dashboard = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "status": "active",
                "system_health": "operational",
                # Real-time trading analytics
                "realtime_analytics": {
                    "portfolio_metrics": realtime_dashboard.get("portfolio", {}),
                    "position_analytics": realtime_dashboard.get("positions", {}),
                    "execution_quality": realtime_dashboard.get("execution_quality", {}),
                    "stress_testing": realtime_dashboard.get("stress_testing", {}),
                    "advanced_var": realtime_dashboard.get("advanced_var", {}),
                },
                # Advanced portfolio analytics
                "portfolio_analytics": {
                    "composition": portfolio_report.get("portfolio_composition", {}),
                    "risk_decomposition": portfolio_report.get("risk_analysis", {}),
                    "factor_analysis": portfolio_report.get("factor_analysis", {}),
                    "optimization": portfolio_report.get("optimization_analysis", {}),
                    "regime_analysis": portfolio_report.get("regime_analysis", {}),
                },
                # Risk monitoring
                "risk_monitoring": {
                    "var_analysis": risk_dashboard.get("var_analysis", {}),
                    "stress_testing": risk_dashboard.get("stress_testing", {}),
                    "risk_limits": risk_dashboard.get("risk_limits", {}),
                    "market_context": risk_dashboard.get("market_context", {}),
                },
                # Performance reporting
                "performance_analytics": {
                    "executive_summary": performance_report.get("executive_summary", {}),
                    "attribution_analysis": performance_report.get("performance_analysis", {}).get(
                        "attribution", {}
                    ),
                    "benchmark_comparison": performance_report.get("performance_analysis", {}).get(
                        "benchmark_comparison", {}
                    ),
                    "performance_quality": performance_report.get("performance_quality", {}),
                },
                # Operational health
                "operational_health": operational_health,
                # Consolidated alerts
                "alerts": {
                    "active_alerts": realtime_dashboard.get("active_alerts", []),
                    "alert_summary": realtime_dashboard.get("alert_summary", {}),
                    "risk_alerts": risk_dashboard.get("active_alerts", []),
                    "operational_alerts": operational_health.get("active_alerts", []),
                },
                # Key performance indicators
                "kpis": {
                    "portfolio_value": realtime_dashboard.get("portfolio", {}).get(
                        "total_value", 0
                    ),
                    "daily_pnl": realtime_dashboard.get("performance_summary", {}).get(
                        "daily_pnl", 0
                    ),
                    "var_95": risk_dashboard.get("risk_summary", {}).get("var_95_1day", 0),
                    "sharpe_ratio": performance_report.get("executive_summary", {}).get(
                        "sharpe_ratio", 0
                    ),
                    "max_drawdown": performance_report.get("executive_summary", {}).get(
                        "max_drawdown", 0
                    ),
                    "tracking_error": performance_report.get("executive_summary", {}).get(
                        "tracking_error", 0
                    ),
                    "information_ratio": performance_report.get("executive_summary", {}).get(
                        "information_ratio", 0
                    ),
                },
            }

            return comprehensive_dashboard

        except Exception as e:
            self.logger.error(f"Error generating comprehensive dashboard: {e}")
            return {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "status": "error",
                "error": str(e),
            }

    async def export_analytics_to_external_systems(
        self, export_configs: list[dict]
    ) -> dict[str, bool]:
        """
        Export analytics data to multiple external systems simultaneously.

        Args:
            export_configs: List of export configurations

        Returns:
            Dictionary of export results by system
        """
        try:
            results = {}

            # Get latest analytics data
            dashboard_data = await self.generate_comprehensive_analytics_dashboard()

            # Execute exports concurrently
            export_tasks = []
            for config in export_configs:
                system_name = config.get("name", "unknown")
                export_type = config.get("type")

                if export_type == "prometheus":
                    task = asyncio.create_task(
                        self.export_service.export_to_prometheus(dashboard_data)
                    )
                elif export_type == "influxdb":
                    task = asyncio.create_task(
                        self.export_service.export_to_influxdb_line_protocol(dashboard_data)
                    )
                elif export_type == "kafka":
                    task = asyncio.create_task(
                        self.export_service.export_to_kafka(
                            config.get("topic"), dashboard_data, config.get("kafka_config")
                        )
                    )
                elif export_type == "api":
                    task = asyncio.create_task(
                        self.export_service.export_to_rest_api(
                            config.get("endpoint"),
                            dashboard_data,
                            config.get("headers"),
                            config.get("auth"),
                        )
                    )
                elif export_type == "regulatory":
                    task = asyncio.create_task(
                        self.export_service.export_regulatory_report(
                            config.get("report_type"), dashboard_data, config.get("template")
                        )
                    )
                else:
                    results[system_name] = False
                    continue

                export_tasks.append((system_name, task))

            # Wait for all exports to complete
            for system_name, task in export_tasks:
                try:
                    result = await task
                    results[system_name] = bool(result) if isinstance(result, bool) else True
                except Exception as e:
                    self.logger.error(f"Export to {system_name} failed: {e}")
                    results[system_name] = False

            return results

        except Exception as e:
            self.logger.error(f"Error in bulk export: {e}")
            raise

    async def run_comprehensive_analytics_cycle(self) -> dict[str, Any]:
        """
        Run a complete analytics cycle with all components.

        Returns:
            Summary of analytics cycle execution
        """
        try:
            cycle_start = get_current_utc_timestamp()

            # 1. Update all real-time analytics
            await self.realtime_analytics._portfolio_analytics_loop()
            await self.realtime_analytics._risk_monitoring_loop()

            # 2. Calculate portfolio optimization recommendations
            mvo_results = await self.portfolio_service.optimize_portfolio_mvo()
            black_litterman_results = await self.portfolio_service.optimize_black_litterman()
            risk_parity_results = await self.portfolio_service.optimize_risk_parity()

            # 3. Execute comprehensive risk analysis
            var_analysis = await self.risk_service.calculate_advanced_var_methodologies()
            stress_test_results = await self.risk_service.execute_comprehensive_stress_test()

            # 4. Generate performance attribution (for metrics collection)
            await self.reporting_service.generate_comprehensive_institutional_report()

            # 5. Check system health
            system_health = await self.operational_service.generate_system_health_dashboard()

            # 6. Update all metrics and alerts
            # Process any pending alerts through the alert service
            if hasattr(self.alert_service, "get_active_alerts"):
                active_alerts = self.alert_service.get_active_alerts()
                self.logger.debug(f"Active alerts: {len(active_alerts)}")

            cycle_end = get_current_utc_timestamp()
            execution_time = (cycle_end - cycle_start).total_seconds()

            # Generate cycle summary
            cycle_summary = {
                "cycle_timestamp": cycle_start.isoformat(),
                "execution_time_seconds": execution_time,
                "components_updated": [
                    "realtime_analytics",
                    "portfolio_analytics",
                    "risk_monitor",
                    "performance_reporter",
                    "operational_analytics",
                ],
                "optimization_results": {
                    "mvo_success": mvo_results.get("optimization_success", False),
                    "black_litterman_success": black_litterman_results.get(
                        "optimization_success", False
                    ),
                    "risk_parity_success": risk_parity_results.get("optimization_success", False),
                },
                "risk_analysis": {
                    "var_calculated": "methodologies" in var_analysis,
                    "stress_tests_completed": "scenarios" in stress_test_results,
                    "worst_case_loss": stress_test_results.get("summary", {}).get(
                        "worst_case_loss_percentage", 0
                    ),
                },
                "system_health": {
                    "overall_score": system_health.get("health_score", 0),
                    "component_status": system_health.get("component_status", {}),
                },
                "alerts_processed": len(self.alert_service.get_active_alerts())
                if self.alert_service and hasattr(self.alert_service, "get_active_alerts")
                else 0,
                "status": "completed",
            }

            self.logger.info(f"Analytics cycle completed in {execution_time:.2f}s")
            return cycle_summary

        except Exception as e:
            self.logger.error(f"Error in analytics cycle: {e}")
            return {
                "cycle_timestamp": get_current_utc_timestamp().isoformat(),
                "status": "error",
                "error": str(e),
            }

    async def start_continuous_analytics(self, cycle_interval_seconds: int = 30) -> None:
        """
        Start continuous analytics processing with scheduled cycles.

        Args:
            cycle_interval_seconds: Interval between analytics cycles
        """

        # Create a background task for continuous analytics instead of blocking
        async def _continuous_analytics_loop():
            try:
                self.logger.info("Starting continuous analytics processing")

                while self._running:
                    # Run analytics cycle
                    cycle_result = await self.run_comprehensive_analytics_cycle()

                    # Log cycle completion
                    if cycle_result.get("status") == "completed":
                        execution_time = cycle_result.get("execution_time_seconds", 0)
                        self.logger.info(f"Analytics cycle completed in {execution_time:.2f}s")
                    else:
                        self.logger.error(
                            f"Analytics cycle failed: {cycle_result.get('error', 'unknown error')}"
                        )

                    # Wait for next cycle
                    await asyncio.sleep(cycle_interval_seconds)

            except asyncio.CancelledError:
                self.logger.info("Continuous analytics processing cancelled")
            except Exception as e:
                self.logger.error(f"Error in continuous analytics: {e}")

        # Create task for the continuous loop
        if self._running:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(_continuous_analytics_loop())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                self.logger.info("Continuous analytics task created")
            except RuntimeError:
                self.logger.warning("No event loop running, cannot start continuous analytics")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for decoupled service communication."""
        from src.analytics.events import AnalyticsEventType

        # Portfolio event handler
        portfolio_handler = PortfolioEventHandler(self.portfolio_service)
        self.event_bus.register_handler(AnalyticsEventType.POSITION_UPDATED, portfolio_handler)
        self.event_bus.register_handler(AnalyticsEventType.PRICE_UPDATED, portfolio_handler)
        self.event_bus.register_handler(AnalyticsEventType.BENCHMARK_UPDATED, portfolio_handler)
        self._event_handlers.append(portfolio_handler)

        # Risk event handler
        risk_handler = RiskEventHandler(self.risk_service)
        self.event_bus.register_handler(AnalyticsEventType.POSITION_UPDATED, risk_handler)
        self.event_bus.register_handler(AnalyticsEventType.PRICE_UPDATED, risk_handler)
        self.event_bus.register_handler(AnalyticsEventType.TRADE_EXECUTED, risk_handler)
        self._event_handlers.append(risk_handler)

        # Alert event handler
        alert_handler = AlertEventHandler(self.alert_service)
        self.event_bus.register_handler(AnalyticsEventType.RISK_LIMIT_BREACHED, alert_handler)
        self.event_bus.register_handler(AnalyticsEventType.ERROR_OCCURRED, alert_handler)
        self._event_handlers.append(alert_handler)

        self.logger.info("Event handlers configured for decoupled service communication")

    async def generate_executive_summary(self) -> dict[str, Any]:
        """
        Generate executive summary for institutional reporting.

        Returns:
            Executive summary with key metrics and insights
        """
        try:
            # Get latest comprehensive dashboard
            dashboard = await self.generate_comprehensive_analytics_dashboard()

            # Extract key metrics
            kpis = dashboard.get("kpis", {})
            performance = dashboard.get("performance_analytics", {}).get("executive_summary", {})

            # Generate executive insights
            insights = []

            # Performance insights
            daily_pnl = kpis.get("daily_pnl", 0)
            if daily_pnl > 0:
                insights.append(f"Positive daily P&L of ${daily_pnl:,.0f}")
            elif daily_pnl < -ANALYTICS_PERFORMANCE_THRESHOLDS["significant_daily_loss"]:
                insights.append(f"Significant daily loss of ${abs(daily_pnl):,.0f}")

            # Risk insights
            var_95 = kpis.get("var_95", 0)
            if var_95 > ANALYTICS_PERFORMANCE_THRESHOLDS["elevated_var_threshold"]:
                insights.append(f"Elevated VaR 95% at ${var_95:,.0f}")

            # Performance quality insights
            sharpe_ratio = kpis.get("sharpe_ratio", 0)
            if sharpe_ratio > ANALYTICS_PERFORMANCE_THRESHOLDS["strong_sharpe_threshold"]:
                insights.append(
                    f"Strong risk-adjusted performance (Sharpe > {ANALYTICS_PERFORMANCE_THRESHOLDS['strong_sharpe_threshold']})"
                )
            elif sharpe_ratio < ANALYTICS_PERFORMANCE_THRESHOLDS["weak_sharpe_threshold"]:
                insights.append("Below-target risk-adjusted performance")

            executive_summary = {
                "summary_date": get_current_utc_timestamp().isoformat(),
                "portfolio_value": kpis.get("portfolio_value", 0),
                "daily_pnl": daily_pnl,
                "mtd_return": performance.get("period_return", 0),
                "ytd_return": performance.get("ytd_return", 0),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": kpis.get("max_drawdown", 0),
                "var_95_1day": var_95,
                "tracking_error": kpis.get("tracking_error", 0),
                "information_ratio": kpis.get("information_ratio", 0),
                "active_alerts": len(dashboard.get("alerts", {}).get("active_alerts", [])),
                "key_insights": insights,
                "overall_health": dashboard.get("system_health", "unknown"),
                "recommendations": [
                    "Monitor concentration risk exposure",
                    "Review factor exposures for regime changes",
                    "Maintain adequate liquidity buffers",
                    "Continue systematic rebalancing process",
                ],
            }

            return executive_summary

        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return {
                "summary_date": get_current_utc_timestamp().isoformat(),
                "status": "error",
                "error": str(e),
            }

    async def create_client_report_package(
        self, client_id: str, report_type: str = "monthly"
    ) -> dict[str, Any]:
        """
        Create comprehensive client report package with all analytics.

        Args:
            client_id: Client identifier
            report_type: Type of report (daily, weekly, monthly, quarterly)

        Returns:
            Complete client report package
        """
        try:
            # Generate all report components
            executive_summary = await self.generate_executive_summary()
            comprehensive_dashboard = await self.generate_comprehensive_analytics_dashboard()
            institutional_report = (
                await self.reporting_service.generate_comprehensive_institutional_report(
                    period=report_type, include_regulatory=True, include_esg=False
                )
            )

            # Create client report package
            report_package = {
                "report_metadata": {
                    "client_id": client_id,
                    "report_date": get_current_utc_timestamp().isoformat(),
                    "report_type": report_type,
                    "report_version": ANALYTICS_REPORT_VERSION,
                    "generated_by": "T-Bot Analytics Service",
                },
                "executive_summary": executive_summary,
                "detailed_analytics": comprehensive_dashboard,
                "institutional_report": institutional_report,
                "export_formats": {
                    "json": True,
                    "csv": True,
                    "excel": True,
                    "pdf": False,  # Would require PDF generation capability
                },
            }

            return report_package

        except Exception as e:
            self.logger.error(f"Error creating client report package: {e}")
            return {
                "client_id": client_id,
                "report_date": get_current_utc_timestamp().isoformat(),
                "status": "error",
                "error": str(e),
            }

    def _default_operational_metrics(self) -> OperationalMetrics:
        """Return default operational metrics when service is unavailable."""
        now = datetime.now()
        return OperationalMetrics(
            timestamp=now,
            system_uptime=Decimal("0"),
            strategies_active=0,
            strategies_total=0,
            exchanges_connected=0,
            exchanges_total=0,
            orders_placed_today=0,
            orders_filled_today=0,
            order_fill_rate=Decimal("0"),
            api_call_success_rate=Decimal("0"),
            websocket_uptime_percent=Decimal("0"),
            error_rate=Decimal("0"),
            critical_errors_today=0,
            memory_usage_percent=Decimal("0"),
            cpu_usage_percent=Decimal("0"),
            disk_usage_percent=Decimal("0"),
            database_connections_active=0,
            cache_hit_rate=Decimal("0"),
            backup_status="unknown",
            compliance_checks_passed=0,
            compliance_checks_failed=0,
            risk_limit_breaches=0,
            circuit_breaker_triggers=0,
            performance_degradation_events=0,
            data_quality_issues=0,
            exchange_outages=0,
        )
