"""
Analytics Service Integration Layer.

This module provides the main service interface for the comprehensive analytics system,
integrating real-time trading analytics, portfolio analytics, performance reporting,
risk monitoring, and operational analytics.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.analytics.alerts.alert_manager import AlertManager
from src.analytics.export.data_exporter import DataExporter
from src.analytics.operational.operational_analytics import OperationalAnalyticsEngine
from src.analytics.portfolio.portfolio_analytics import PortfolioAnalyticsEngine
from src.analytics.reporting.performance_reporter import PerformanceReporter
from src.analytics.risk.risk_monitor import RiskMonitor
from src.analytics.trading.realtime_analytics import RealtimeAnalyticsEngine
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
from src.base import BaseComponent
from src.core.types.trading import Order, Position, Trade
from src.monitoring.metrics import get_metrics_collector
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

    def __init__(self, config: AnalyticsConfiguration | None = None):
        """
        Initialize analytics service.

        Args:
            config: Analytics configuration (uses defaults if not provided)
        """
        super().__init__()
        self.config = config or AnalyticsConfiguration()
        self.metrics_collector = get_metrics_collector()

        # Initialize analytics engines
        self.realtime_analytics = RealtimeAnalyticsEngine(self.config)
        self.portfolio_analytics = PortfolioAnalyticsEngine(self.config)
        self.performance_reporter = PerformanceReporter(self.config)
        self.risk_monitor = RiskMonitor(self.config)
        self.operational_analytics = OperationalAnalyticsEngine(self.config)
        self.alert_manager = AlertManager(self.config)
        self.data_exporter = DataExporter()

        # Service state
        self._running = False
        self._background_tasks: set = set()

        # Caching for performance
        self._cache_enabled = True
        self._cache_ttl = timedelta(seconds=self.config.cache_ttl_seconds)
        self._cached_metrics: dict[str, dict[str, Any]] = {}

        self.logger.info("AnalyticsService initialized")

    async def start(self) -> None:
        """Start the analytics service and all sub-engines."""
        if self._running:
            self.logger.warning("Analytics service already running")
            return

        self._running = True

        try:
            # Start all analytics engines
            await self.realtime_analytics.start()
            await self.risk_monitor.start()
            await self.operational_analytics.start()
            await self.alert_manager.start()

            # Start background tasks
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
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Stop analytics engines
            await self.realtime_analytics.stop()
            await self.risk_monitor.stop()
            await self.operational_analytics.stop()
            await self.alert_manager.stop()

            self.logger.info("Analytics service stopped")

        except Exception as e:
            self.logger.error(f"Error stopping analytics service: {e}")

    # Trading Data Input Methods

    def update_position(self, position: Position) -> None:
        """
        Update position data across all analytics engines.

        Args:
            position: Position update
        """
        try:
            self.realtime_analytics.update_position(position)

            # Update portfolio analytics with all positions
            # (This is simplified - in practice would maintain position registry)
            positions = {f"{position.exchange}:{position.symbol}": position}
            self.portfolio_analytics.update_positions(positions)
            self.risk_monitor.update_positions(positions)

            self.logger.debug(f"Updated position: {position.symbol}")

        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

    def update_trade(self, trade: Trade) -> None:
        """
        Update trade data for analytics.

        Args:
            trade: Trade data
        """
        try:
            self.realtime_analytics.update_trade(trade)

            # Add transaction cost data to performance reporter
            if hasattr(trade, "fee") and trade.fee > 0:
                self.performance_reporter.add_transaction_cost(
                    timestamp=trade.timestamp,
                    symbol=trade.symbol,
                    cost_type="commission",
                    cost_amount=trade.fee,
                    trade_value=trade.quantity * trade.price,
                )

            self.logger.debug(f"Updated trade: {trade.trade_id}")

        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")

    def update_order(self, order: Order) -> None:
        """
        Update order data for analytics.

        Args:
            order: Order data
        """
        try:
            self.realtime_analytics.update_order(order)

            # Record order event for operational analytics
            self.operational_analytics.record_order_event(
                event_type="filled" if order.is_filled() else "updated",
                exchange=order.exchange,
                order_id=order.order_id,
                success=order.status.value not in ["rejected", "expired"],
            )

            self.logger.debug(f"Updated order: {order.order_id}")

        except Exception as e:
            self.logger.error(f"Error updating order: {e}")

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

            self.realtime_analytics.update_price(symbol, price)
            self.portfolio_analytics.update_price_data(symbol, price, timestamp)

            # Update risk monitor
            self.risk_monitor.update_prices({symbol: price})

            self.logger.debug(f"Updated price: {symbol} = {price}")

        except Exception as e:
            self.logger.error(f"Error updating price: {e}")

    def update_benchmark_data(self, benchmark_name: str, data: BenchmarkData) -> None:
        """
        Update benchmark data for performance comparison.

        Args:
            benchmark_name: Benchmark name
            data: Benchmark data
        """
        try:
            self.portfolio_analytics.update_benchmark_data(benchmark_name, data)
            self.logger.debug(f"Updated benchmark: {benchmark_name}")

        except Exception as e:
            self.logger.error(f"Error updating benchmark data: {e}")

    # Analytics Query Methods

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

            metrics = await self.realtime_analytics.get_portfolio_metrics()

            # Cache result
            if self._cache_enabled and metrics:
                self._cache_result(cache_key, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting portfolio metrics: {e}")
            return None

    async def get_position_metrics(self, symbol: str = None) -> list[PositionMetrics]:
        """
        Get position metrics.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position metrics
        """
        try:
            return await self.realtime_analytics.get_position_metrics(symbol)

        except Exception as e:
            self.logger.error(f"Error getting position metrics: {e}")
            return []

    async def get_strategy_metrics(self, strategy: str = None) -> list[StrategyMetrics]:
        """
        Get strategy performance metrics.

        Args:
            strategy: Optional strategy filter

        Returns:
            List of strategy metrics
        """
        try:
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

            metrics = await self.portfolio_analytics.calculate_risk_metrics()

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
            return await self.operational_analytics.calculate_operational_metrics()

        except Exception as e:
            self.logger.error(f"Error getting operational metrics: {e}")
            return OperationalMetrics(timestamp=get_current_utc_timestamp())

    # Risk Analysis Methods

    async def calculate_var(
        self, confidence_level: float = 0.95, time_horizon: int = 1, method: str = "historical"
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
            return await self.risk_monitor.calculate_var(confidence_level, time_horizon, method)

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {}

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
            return await self.risk_monitor.run_stress_test(scenario_name, scenario_params)

        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            return {}

    async def get_portfolio_composition(self) -> dict[str, Any]:
        """
        Get portfolio composition analysis.

        Returns:
            Portfolio composition metrics
        """
        try:
            return await self.portfolio_analytics.calculate_portfolio_composition()

        except Exception as e:
            self.logger.error(f"Error getting portfolio composition: {e}")
            return {}

    async def get_correlation_matrix(self) -> Any | None:
        """
        Get correlation matrix for portfolio positions.

        Returns:
            Correlation matrix DataFrame or None
        """
        try:
            return await self.portfolio_analytics.calculate_correlation_matrix()

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
            return await self.performance_reporter.generate_performance_report(
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
            return await self.risk_monitor.generate_risk_report()

        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {}

    async def generate_health_report(self) -> dict[str, Any]:
        """
        Generate system health report.

        Returns:
            System health report with status and recommendations
        """
        try:
            return await self.operational_analytics.generate_health_report()

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {}

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
            realtime_alerts = await self.realtime_analytics.get_active_alerts()
            alerts.extend(realtime_alerts)

            # Get alerts from alert manager
            alert_manager_alerts = self.alert_manager.get_active_alerts()
            alerts.extend(alert_manager_alerts)

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
                if hasattr(obj, "dict"):
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
            return {}

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
                return await self.data_exporter.export_portfolio_metrics(
                    portfolio_metrics, format, include_metadata
                )
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
            risk_metrics = await self.get_risk_metrics()
            return await self.data_exporter.export_risk_metrics(
                risk_metrics, format, include_metadata
            )

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
                return await self.data_exporter.export_position_metrics(
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
                return await self.data_exporter.export_strategy_metrics(
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
            return await self.data_exporter.export_operational_metrics(
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
            return await self.data_exporter.export_complete_report(report, format, include_charts)

        except Exception as e:
            self.logger.error(f"Error exporting complete report: {e}")
            return ""

    def get_export_statistics(self) -> dict[str, Any]:
        """
        Get export usage statistics.

        Returns:
            Export statistics from data exporter
        """
        return self.data_exporter.get_export_statistics()

    # Alert Management Methods

    def add_alert_rule(self, rule) -> None:
        """Add custom alert rule."""
        self.alert_manager.add_alert_rule(rule)

    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove alert rule."""
        self.alert_manager.remove_alert_rule(rule_id)

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        return await self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_note: str = None
    ) -> bool:
        """Resolve an alert."""
        return await self.alert_manager.resolve_alert(alert_id, resolved_by, resolution_note)

    def get_alert_statistics(self, period_hours: int = 24) -> dict[str, Any]:
        """Get alert statistics."""
        return self.alert_manager.get_alert_statistics(period_hours)

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
            self.portfolio_analytics.config = config
            self.performance_reporter.config = config
            self.risk_monitor.config = config
            self.operational_analytics.config = config
            self.alert_manager.config = config

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
                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic reporting loop: {e}")
                await asyncio.sleep(300)

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
                await asyncio.sleep(60)  # Clean every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache maintenance loop: {e}")
                await asyncio.sleep(60)

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
        """
        try:
            self.operational_analytics.record_strategy_event(
                strategy_name=strategy_name,
                event_type=event_type,
                success=success,
                error_message=error_message,
            )

        except Exception as e:
            self.logger.error(f"Error recording strategy event: {e}")

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
            self.operational_analytics.record_market_data_event(
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
        """
        try:
            from src.analytics.types import AlertSeverity

            severity_map = {
                "low": AlertSeverity.LOW,
                "medium": AlertSeverity.MEDIUM,
                "high": AlertSeverity.HIGH,
                "critical": AlertSeverity.CRITICAL,
            }

            self.operational_analytics.record_error(
                component=component,
                error_type=error_type,
                error_message=error_message,
                severity=severity_map.get(severity, AlertSeverity.LOW),
            )

        except Exception as e:
            self.logger.error(f"Error recording system error: {e}")

    def record_api_call(
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
            self.operational_analytics.record_api_call(
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
                await self.portfolio_analytics.generate_institutional_analytics_report()
            )
            risk_dashboard = await self.risk_monitor.create_real_time_risk_dashboard()
            performance_report = (
                await self.performance_reporter.generate_comprehensive_institutional_report()
            )
            operational_health = await self.operational_analytics.generate_system_health_dashboard()

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
                    task = self.data_exporter.export_to_prometheus(dashboard_data)
                elif export_type == "influxdb":
                    task = self.data_exporter.export_to_influxdb_line_protocol(dashboard_data)
                elif export_type == "kafka":
                    task = self.data_exporter.export_to_kafka(
                        config.get("topic"), dashboard_data, config.get("kafka_config")
                    )
                elif export_type == "api":
                    task = self.data_exporter.export_to_rest_api(
                        config.get("endpoint"),
                        dashboard_data,
                        config.get("headers"),
                        config.get("auth"),
                    )
                elif export_type == "regulatory":
                    task = self.data_exporter.export_regulatory_report(
                        config.get("report_type"), dashboard_data, config.get("template")
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
            return {}

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
            mvo_results = await self.portfolio_analytics.optimize_portfolio_mvo()
            black_litterman_results = await self.portfolio_analytics.optimize_black_litterman()
            risk_parity_results = await self.portfolio_analytics.optimize_risk_parity()

            # 3. Execute comprehensive risk analysis
            var_analysis = await self.risk_monitor.calculate_advanced_var_methodologies()
            stress_test_results = await self.risk_monitor.execute_comprehensive_stress_test()

            # 4. Generate performance attribution
            institutional_report = (
                await self.performance_reporter.generate_comprehensive_institutional_report()
            )

            # 5. Check system health
            system_health = await self.operational_analytics.generate_system_health_dashboard()

            # 6. Update all metrics and alerts
            await self.alert_manager.process_pending_alerts()

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
                "alerts_processed": len(await self.alert_manager.get_active_alerts()),
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
        try:
            self.logger.info("Starting continuous analytics processing")

            while True:
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
            risk = dashboard.get("risk_monitoring", {}).get("risk_summary", {})

            # Generate executive insights
            insights = []

            # Performance insights
            daily_pnl = kpis.get("daily_pnl", 0)
            if daily_pnl > 0:
                insights.append(f"Positive daily P&L of ${daily_pnl:,.0f}")
            elif daily_pnl < -10000:
                insights.append(f"Significant daily loss of ${abs(daily_pnl):,.0f}")

            # Risk insights
            var_95 = kpis.get("var_95", 0)
            if var_95 > 50000:
                insights.append(f"Elevated VaR 95% at ${var_95:,.0f}")

            # Performance quality insights
            sharpe_ratio = kpis.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.5:
                insights.append("Strong risk-adjusted performance (Sharpe > 1.5)")
            elif sharpe_ratio < 0.5:
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
                await self.performance_reporter.generate_comprehensive_institutional_report(
                    period=report_type, include_regulatory=True, include_esg=False
                )
            )

            # Create client report package
            report_package = {
                "report_metadata": {
                    "client_id": client_id,
                    "report_date": get_current_utc_timestamp().isoformat(),
                    "report_type": report_type,
                    "report_version": "2.0",
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
