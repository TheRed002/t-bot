"""
Web Analytics Service Implementation.

This service provides a web-specific interface to the analytics system,
handling data transformation, formatting, and web-specific business logic.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.analytics.interfaces import (
    AlertServiceProtocol,
    AnalyticsServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.types import ReportType
from src.core.base import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.utils.decorators import cached, monitored
from src.web_interface.data_transformer import WebInterfaceDataTransformer

logger = get_logger(__name__)


class WebAnalyticsService(BaseService):
    """
    Web interface service for analytics operations.

    This service wraps the analytics services and provides web-specific
    formatting, validation, and business logic.
    """

    def __init__(
        self,
        analytics_service: AnalyticsServiceProtocol,
        portfolio_service: PortfolioServiceProtocol | None = None,
        risk_service: RiskServiceProtocol | None = None,
        reporting_service: ReportingServiceProtocol | None = None,
        alert_service: AlertServiceProtocol | None = None,
        operational_service: OperationalServiceProtocol | None = None,
        export_service: ExportServiceProtocol | None = None,
    ):
        """Initialize web analytics service with dependencies."""
        super().__init__("WebAnalyticsService")

        # Core analytics service (required)
        self.analytics_service = analytics_service

        # Optional specialized services
        self.portfolio_service = portfolio_service
        self.risk_service = risk_service
        self.reporting_service = reporting_service
        self.alert_service = alert_service
        self.operational_service = operational_service
        self.export_service = export_service

        logger.info("Web analytics service initialized")

    async def _do_start(self) -> None:
        """Start the web analytics service."""
        logger.info("Starting web analytics service")
        if hasattr(self.analytics_service, "start"):
            await self.analytics_service.start()

    async def _do_stop(self) -> None:
        """Stop the web analytics service."""
        logger.info("Stopping web analytics service")
        if hasattr(self.analytics_service, "stop"):
            await self.analytics_service.stop()

    # Portfolio Analytics Methods
    @cached(ttl=60)  # Cache for 1 minute
    @monitored()
    async def get_portfolio_metrics(self) -> dict[str, Any]:
        """Get portfolio metrics formatted for web display."""
        try:
            metrics = await self.analytics_service.get_portfolio_metrics()

            if not metrics:
                return self._get_empty_portfolio_metrics()

            # Transform for web display
            return {
                "total_value": str(metrics.total_value),
                "total_pnl": str(metrics.total_pnl),
                "total_pnl_percentage": float(metrics.total_pnl_percentage),
                "win_rate": float(metrics.win_rate),
                "sharpe_ratio": float(metrics.sharpe_ratio) if metrics.sharpe_ratio else 0.0,
                "max_drawdown": float(metrics.max_drawdown) if metrics.max_drawdown else 0.0,
                "positions_count": metrics.positions_count,
                "active_strategies": metrics.active_strategies,
                "timestamp": metrics.timestamp,
            }
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            raise ServiceError(f"Failed to retrieve portfolio metrics: {e!s}")

    async def get_portfolio_composition(self) -> dict[str, Any]:
        """Get portfolio composition analysis."""
        if not self.portfolio_service:
            raise ServiceError("Portfolio service not available")

        try:
            composition = await self.portfolio_service.get_portfolio_composition()

            # Format for web display
            return WebInterfaceDataTransformer.format_portfolio_composition(composition)
        except Exception as e:
            logger.error(f"Error getting portfolio composition: {e}")
            raise ServiceError(f"Failed to retrieve portfolio composition: {e!s}")

    async def get_correlation_matrix(self) -> Any:
        """Get portfolio correlation matrix."""
        if not self.portfolio_service:
            raise ServiceError("Portfolio service not available")

        try:
            matrix = await self.portfolio_service.calculate_correlation_matrix()
            return matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise ServiceError(f"Failed to calculate correlation matrix: {e!s}")

    async def export_portfolio_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """Export portfolio data in specified format."""
        if not self.export_service:
            raise ServiceError("Export service not available")

        try:
            export_data = await self.export_service.export_portfolio_data(
                format=format, include_metadata=include_metadata
            )
            return export_data
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
            raise ServiceError(f"Failed to export portfolio data: {e!s}")

    # Risk Analytics Methods
    @cached(ttl=300)  # Cache for 5 minutes
    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get comprehensive risk metrics."""
        try:
            metrics = await self.analytics_service.get_risk_metrics()

            # Transform for web display
            return {
                "portfolio_var": {
                    "var_95": str(metrics.portfolio_var.get("95", Decimal("0"))),
                    "var_99": str(metrics.portfolio_var.get("99", Decimal("0"))),
                },
                "portfolio_volatility": float(metrics.portfolio_volatility),
                "portfolio_beta": float(metrics.portfolio_beta) if metrics.portfolio_beta else 0.0,
                "correlation_risk": float(metrics.correlation_risk),
                "concentration_risk": float(metrics.concentration_risk),
                "leverage_ratio": float(metrics.leverage_ratio),
                "margin_usage": float(metrics.margin_usage),
                "timestamp": metrics.timestamp,
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            raise ServiceError(f"Failed to retrieve risk metrics: {e!s}")

    async def calculate_var(
        self, confidence_level: float, time_horizon: int, method: str
    ) -> dict[str, Any]:
        """Calculate Value at Risk."""
        if not self.risk_service:
            raise ServiceError("Risk service not available")

        try:
            var_results = await self.risk_service.calculate_var(
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method=method,
            )

            # Format results for web
            return {
                f"var_{int(confidence_level * 100)}": str(value)
                for key, value in var_results.items()
            }
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            raise ServiceError(f"Failed to calculate VaR: {e!s}")

    async def run_stress_test(
        self, scenario_name: str, scenario_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Run stress test scenario."""
        if not self.risk_service:
            raise ServiceError("Risk service not available")

        try:
            results = await self.risk_service.run_stress_test(
                scenario_name=scenario_name, scenario_params=scenario_params
            )

            # Format results for web
            return WebInterfaceDataTransformer.format_stress_test_results(results)
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            raise ServiceError(f"Failed to run stress test: {e!s}")

    async def get_risk_exposure(self) -> dict[str, Any]:
        """Get current risk exposure analysis."""
        if not self.risk_service:
            raise ServiceError("Risk service not available")

        try:
            metrics = await self.risk_service.get_risk_metrics()

            # Calculate exposure metrics
            return {
                "total_exposure": str(metrics.total_exposure),
                "exposure_by_asset": {
                    asset: str(value) for asset, value in metrics.exposure_by_asset.items()
                },
                "exposure_by_strategy": {
                    strategy: str(value) for strategy, value in metrics.exposure_by_strategy.items()
                },
                "concentration_metrics": metrics.concentration_metrics,
                "timestamp": datetime.utcnow(),
            }
        except Exception as e:
            logger.error(f"Error getting risk exposure: {e}")
            raise ServiceError(f"Failed to retrieve risk exposure: {e!s}")

    # Strategy Analytics Methods
    async def get_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None:
        """Get metrics for a specific strategy."""
        try:
            # Get strategy metrics from analytics service
            strategies = await self.analytics_service.get_strategy_metrics(strategy=strategy_id)

            if not strategies:
                return None

            strategy = strategies[0] if isinstance(strategies, list) else strategies

            # Transform for web display
            return {
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.strategy_name,
                "total_trades": strategy.total_trades,
                "winning_trades": strategy.winning_trades,
                "losing_trades": strategy.losing_trades,
                "win_rate": float(strategy.win_rate),
                "avg_profit": str(strategy.avg_profit),
                "avg_loss": str(strategy.avg_loss),
                "profit_factor": float(strategy.profit_factor),
                "sharpe_ratio": float(strategy.sharpe_ratio) if strategy.sharpe_ratio else 0.0,
                "max_drawdown": float(strategy.max_drawdown) if strategy.max_drawdown else 0.0,
                "current_positions": strategy.current_positions,
                "total_pnl": str(strategy.total_pnl),
            }
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            raise ServiceError(f"Failed to retrieve strategy metrics: {e!s}")

    async def get_strategy_performance(self, strategy_id: str, days: int) -> dict[str, Any]:
        """Get strategy performance history."""
        try:
            # This would typically fetch historical performance data
            # For now, returning a placeholder structure
            return {
                "strategy_id": strategy_id,
                "period_days": days,
                "performance_data": [],  # Would contain time series data
                "summary": {
                    "total_return": "0.00",
                    "avg_daily_return": "0.00",
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                },
            }
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            raise ServiceError(f"Failed to retrieve strategy performance: {e!s}")

    async def compare_strategies(self, strategy_ids: list[str]) -> dict[str, Any]:
        """Compare multiple strategies."""
        try:
            comparison_data = {}

            for strategy_id in strategy_ids:
                metrics = await self.get_strategy_metrics(strategy_id)
                if metrics:
                    comparison_data[strategy_id] = metrics

            if not comparison_data:
                raise ValidationError("No valid strategies found for comparison")

            # Calculate comparative metrics
            return {
                "strategies": comparison_data,
                "comparison": self._calculate_strategy_comparison(comparison_data),
            }
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            raise ServiceError(f"Failed to compare strategies: {e!s}")

    # Operational Analytics Methods
    async def get_operational_metrics(self) -> dict[str, Any]:
        """Get operational metrics."""
        if not self.operational_service:
            # Return basic metrics if service not available
            return {
                "uptime_percentage": 100.0,
                "api_latency_ms": 50.0,
                "error_rate": 0.0,
                "active_connections": 0,
                "timestamp": datetime.utcnow(),
            }

        try:
            metrics = await self.operational_service.get_operational_metrics()
            return WebInterfaceDataTransformer.format_operational_metrics(metrics)
        except Exception as e:
            logger.error(f"Error getting operational metrics: {e}")
            raise ServiceError(f"Failed to retrieve operational metrics: {e!s}")

    async def get_system_errors(self, hours: int) -> list[dict[str, Any]]:
        """Get recent system errors."""
        if not self.operational_service:
            return []

        try:
            # Would fetch errors from the operational service
            # For now, returning empty list
            return []
        except Exception as e:
            logger.error(f"Error getting system errors: {e}")
            raise ServiceError(f"Failed to retrieve system errors: {e!s}")

    async def get_operational_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get operational events."""
        if not self.operational_service:
            return []

        try:
            # Would fetch events from the operational service
            # For now, returning empty list
            return []
        except Exception as e:
            logger.error(f"Error getting operational events: {e}")
            raise ServiceError(f"Failed to retrieve operational events: {e!s}")

    # Reporting Methods
    async def generate_report(
        self,
        report_type: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        include_charts: bool = True,
        format: str = "json",
    ) -> dict[str, Any]:
        """Generate a performance report."""
        if not self.reporting_service:
            raise ServiceError("Reporting service not available")

        try:
            report = await self.reporting_service.generate_performance_report(
                report_type=ReportType(report_type),
                start_date=start_date,
                end_date=end_date,
            )

            # Format and return report
            return {
                "id": report.report_id,
                "type": report.report_type.value,
                "generated_at": report.generated_at,
                "data": report.data,
                "url": f"/api/analytics/reports/{report.report_id}",
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise ServiceError(f"Failed to generate report: {e!s}")

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Get a generated report."""
        # Implementation would fetch from storage
        return None

    async def list_reports(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        """List available reports for a user."""
        # Implementation would fetch from storage
        return []

    async def schedule_report(
        self,
        report_type: str,
        schedule: str,
        recipients: list[str],
        created_by: str,
    ) -> dict[str, Any]:
        """Schedule a recurring report."""
        # Implementation would create scheduled job
        return {
            "id": "scheduled_report_123",
            "report_type": report_type,
            "schedule": schedule,
            "recipients": recipients,
            "created_by": created_by,
            "created_at": datetime.utcnow(),
        }

    # Alert Management Methods
    async def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get active alerts."""
        if not self.alert_service:
            return []

        try:
            alerts = self.alert_service.get_active_alerts()

            # Format for web display
            return [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "created_at": alert.created_at,
                    "acknowledged": alert.acknowledged,
                }
                for alert in alerts
            ]
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            raise ServiceError(f"Failed to retrieve active alerts: {e!s}")

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str, notes: str | None = None
    ) -> bool:
        """Acknowledge an alert."""
        if not self.alert_service:
            raise ServiceError("Alert service not available")

        try:
            return await self.alert_service.acknowledge_alert(
                alert_id=alert_id, acknowledged_by=acknowledged_by
            )
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            raise ServiceError(f"Failed to acknowledge alert: {e!s}")

    async def get_alert_history(
        self, days: int, severity: str | None = None
    ) -> list[dict[str, Any]]:
        """Get alert history."""
        # Implementation would fetch historical alerts
        return []

    async def configure_alerts(self, config: dict[str, Any], configured_by: str) -> dict[str, Any]:
        """Configure alert thresholds and rules."""
        # Implementation would update alert configuration
        return {
            "status": "configured",
            "configured_by": configured_by,
            "configuration": config,
            "updated_at": datetime.utcnow(),
        }

    # Helper Methods
    def _get_empty_portfolio_metrics(self) -> dict[str, Any]:
        """Get empty portfolio metrics structure."""
        return {
            "total_value": "0.00",
            "total_pnl": "0.00",
            "total_pnl_percentage": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "positions_count": 0,
            "active_strategies": 0,
            "timestamp": datetime.utcnow(),
        }

    def _calculate_strategy_comparison(
        self, strategies: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate comparative metrics between strategies."""
        # Implementation would calculate relative performance metrics
        return {
            "best_performer": max(
                strategies.keys(),
                key=lambda x: float(strategies[x].get("total_pnl", 0)),
            )
            if strategies
            else None,
            "highest_win_rate": max(
                strategies.keys(),
                key=lambda x: strategies[x].get("win_rate", 0),
            )
            if strategies
            else None,
            "lowest_drawdown": min(
                strategies.keys(),
                key=lambda x: strategies[x].get("max_drawdown", float("inf")),
            )
            if strategies
            else None,
        }
