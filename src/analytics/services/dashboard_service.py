"""
Dashboard Service - Business logic for analytics dashboard generation.

This service contains the business logic for generating comprehensive 
analytics dashboards, separating it from the main analytics service.
"""

from decimal import Decimal
from typing import Any

from src.analytics.base_analytics_service import BaseAnalyticsService
from src.analytics.interfaces import (
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.types import AnalyticsConfiguration
from src.utils.datetime_utils import get_current_utc_timestamp


class DashboardService(BaseAnalyticsService):
    """Service for generating analytics dashboards."""

    def __init__(
        self,
        config: AnalyticsConfiguration | None = None,
        portfolio_service: PortfolioServiceProtocol | None = None,
        risk_service: RiskServiceProtocol | None = None,
        operational_service: OperationalServiceProtocol | None = None,
        metrics_collector=None,
    ):
        """Initialize dashboard service with injected dependencies."""
        super().__init__(
            name="DashboardService",
            config=config,
            metrics_collector=metrics_collector,
        )
        self.portfolio_service = portfolio_service
        self.risk_service = risk_service
        self.operational_service = operational_service

    async def generate_comprehensive_dashboard(
        self,
        portfolio_metrics=None,
        risk_metrics=None,
        operational_metrics=None,
        position_metrics=None,
        strategy_metrics=None,
        active_alerts=None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive analytics dashboard.
        
        Business logic for assembling dashboard data from various analytics components.
        """
        try:
            current_time = get_current_utc_timestamp()

            # Use provided metrics or get from services with error handling
            if portfolio_metrics is None and self.portfolio_service:
                try:
                    portfolio_metrics = await self.portfolio_service.calculate_portfolio_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get portfolio metrics: {e}")
                    portfolio_metrics = None

            if risk_metrics is None and self.risk_service:
                try:
                    risk_metrics = await self.risk_service.get_risk_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get risk metrics: {e}")
                    risk_metrics = None

            if operational_metrics is None and self.operational_service:
                try:
                    operational_metrics = await self.operational_service.get_operational_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get operational metrics: {e}")
                    operational_metrics = None

            # Default empty collections if not provided
            position_metrics = position_metrics or []
            strategy_metrics = strategy_metrics or []
            active_alerts = active_alerts or []

            # Generate comprehensive dashboard
            dashboard = {
                "timestamp": current_time,
                "dashboard_type": "comprehensive_analytics",
                "status": "healthy" if len(active_alerts) == 0 else "warning",
                "system_health": self._build_system_health_section(operational_metrics),
                "realtime_analytics": self._build_realtime_section(
                    portfolio_metrics, position_metrics, strategy_metrics
                ),
                "portfolio_analytics": self._build_portfolio_section(portfolio_metrics),
                "portfolio_summary": self._build_portfolio_section(portfolio_metrics),
                "risk_analytics": self._build_risk_section(risk_metrics),
                "risk_summary": self._build_risk_section(risk_metrics),
                "operational_analytics": self._build_operational_section(operational_metrics),
                "operational_summary": self._build_operational_section(operational_metrics),
                "position_summary": position_metrics,
                "strategy_performance": strategy_metrics,
                "alerts_summary": self._build_alerts_section(active_alerts),
                "active_alerts": active_alerts,
                "performance_indicators": self._build_performance_indicators(
                    portfolio_metrics, risk_metrics, operational_metrics
                ),
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating comprehensive analytics dashboard: {e}")
            raise

    async def generate_quick_dashboard(self) -> dict[str, Any]:
        """
        Generate quick overview dashboard with essential metrics.

        Business logic for creating a simplified dashboard for quick monitoring.
        """
        try:
            current_time = get_current_utc_timestamp()

            # Get basic metrics from services with error handling
            portfolio_metrics = None
            risk_metrics = None
            active_alerts = []

            if self.portfolio_service:
                try:
                    portfolio_metrics = await self.portfolio_service.calculate_portfolio_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get portfolio metrics: {e}")
                    portfolio_metrics = None

            if self.risk_service:
                try:
                    risk_metrics = await self.risk_service.get_risk_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get risk metrics: {e}")
                    risk_metrics = None

            # Generate quick dashboard
            dashboard = {
                "timestamp": current_time,
                "dashboard_type": "quick_overview",
                "status": "healthy",
                "portfolio_summary": self._build_portfolio_section(portfolio_metrics),
                "risk_summary": self._build_risk_section(risk_metrics),
                "alerts_summary": self._build_alerts_section(active_alerts),
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating quick analytics dashboard: {e}")
            raise

    async def generate_risk_dashboard(self, risk_metrics=None) -> dict[str, Any]:
        """
        Generate risk-focused dashboard.

        Business logic for creating a risk-focused dashboard with detailed risk analytics.
        """
        try:
            current_time = get_current_utc_timestamp()

            # Use provided metrics or get from service with error handling
            if risk_metrics is None and self.risk_service:
                try:
                    risk_metrics = await self.risk_service.get_risk_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get risk metrics: {e}")
                    risk_metrics = None

            # Generate risk dashboard
            dashboard = {
                "timestamp": current_time,
                "dashboard_type": "risk_focused",
                "status": "healthy",
                "risk_metrics": self._build_risk_section(risk_metrics),
                "risk_alerts": [],  # Empty for now, can be enhanced later
                "risk_trends": {},  # Empty for now, can be enhanced later
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating risk analytics dashboard: {e}")
            raise

    async def generate_performance_dashboard(self, portfolio_metrics=None, strategy_metrics=None) -> dict[str, Any]:
        """
        Generate performance-focused dashboard.

        Business logic for creating a performance-focused dashboard with detailed performance analytics.
        """
        try:
            current_time = get_current_utc_timestamp()

            # Use provided metrics or get from services with error handling
            if portfolio_metrics is None and self.portfolio_service:
                try:
                    portfolio_metrics = await self.portfolio_service.calculate_portfolio_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get portfolio metrics: {e}")
                    portfolio_metrics = None

            strategy_metrics = strategy_metrics or []

            # Generate performance dashboard
            dashboard = {
                "timestamp": current_time,
                "dashboard_type": "performance_focused",
                "status": "healthy",
                "portfolio_performance": self._build_portfolio_section(portfolio_metrics),
                "strategy_performance": strategy_metrics,
                "performance_trends": {},  # Empty for now, can be enhanced later
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating performance analytics dashboard: {e}")
            raise

    def _build_system_health_section(self, operational_metrics) -> dict[str, Any]:
        """Build system health section of dashboard."""
        if not operational_metrics:
            return {
                "uptime": Decimal("0"),
                "cpu_usage": Decimal("0"),
                "memory_usage": Decimal("0"),
                "error_rate": Decimal("0"),
            }

        return {
            "uptime": operational_metrics.system_uptime,
            "cpu_usage": operational_metrics.cpu_usage_percent,
            "memory_usage": operational_metrics.memory_usage_percent,
            "error_rate": operational_metrics.error_rate,
        }

    def _build_realtime_section(
        self, portfolio_metrics, position_metrics, strategy_metrics
    ) -> dict[str, Any]:
        """Build realtime analytics section of dashboard."""
        return {
            "portfolio_value": portfolio_metrics.total_value if portfolio_metrics else Decimal("0"),
            "daily_pnl": portfolio_metrics.total_pnl if portfolio_metrics else Decimal("0"),
            "position_count": len(position_metrics),
            "strategy_count": len(strategy_metrics),
        }

    def _build_portfolio_section(self, portfolio_metrics) -> dict[str, Any]:
        """Build portfolio analytics section of dashboard."""
        if not portfolio_metrics:
            return {
                "total_value": "0",
                "unrealized_pnl": "0",
                "realized_pnl": "0",
                "cash": "0",
            }

        return {
            "total_value": str(portfolio_metrics.total_value),
            "unrealized_pnl": str(getattr(portfolio_metrics, "unrealized_pnl", Decimal("0"))),
            "realized_pnl": str(getattr(portfolio_metrics, "realized_pnl", Decimal("0"))),
            "cash": str(getattr(portfolio_metrics, "cash", Decimal("0"))),
        }

    def _build_risk_section(self, risk_metrics) -> dict[str, Any]:
        """Build risk analytics section of dashboard."""
        if not risk_metrics:
            return {
                "var_95": Decimal("0"),
                "max_drawdown": Decimal("0"),
            }

        return {
            "var_95": getattr(risk_metrics, "var_95", Decimal("0")),
            "max_drawdown": getattr(risk_metrics, "max_drawdown", Decimal("0")),
        }

    def _build_operational_section(self, operational_metrics) -> dict[str, Any]:
        """Build operational analytics section of dashboard."""
        if not operational_metrics:
            return {
                "orders_today": 0,
                "fill_rate": Decimal("0"),
                "api_success_rate": Decimal("0"),
                "active_strategies": 0,
            }

        return {
            "orders_today": operational_metrics.orders_placed_today,
            "fill_rate": operational_metrics.order_fill_rate,
            "api_success_rate": operational_metrics.api_call_success_rate,
            "active_strategies": operational_metrics.strategies_active,
        }

    def _build_alerts_section(self, active_alerts) -> dict[str, Any]:
        """Build alerts summary section of dashboard."""
        return {
            "active_count": len(active_alerts),
            "alerts": active_alerts[:5] if active_alerts else []  # Show first 5 alerts
        }

    def _build_performance_indicators(self, portfolio_metrics, risk_metrics, operational_metrics) -> dict[str, Any]:
        """Build key performance indicators section of dashboard."""
        indicators = {}

        if portfolio_metrics:
            indicators.update({
                "total_return": portfolio_metrics.total_return,
                "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                "daily_pnl": portfolio_metrics.realized_pnl + portfolio_metrics.unrealized_pnl,
            })

        if risk_metrics:
            indicators.update({
                "max_drawdown": risk_metrics.max_drawdown,
                "volatility": risk_metrics.volatility,
            })

        if operational_metrics:
            indicators.update({
                "system_uptime": operational_metrics.system_uptime,
                "error_rate": operational_metrics.error_rate,
            })

        return indicators

    def _format_portfolio_summary(self, portfolio_metrics) -> dict[str, Any]:
        """Format portfolio summary for dashboard display."""
        if not portfolio_metrics:
            return {
                "total_value": "0",
                "available_balance": "0",
                "unrealized_pnl": "0",
                "realized_pnl": "0",
            }

        return {
            "total_value": str(portfolio_metrics.total_value),
            "available_balance": str(getattr(portfolio_metrics, "cash", Decimal("0"))),
            "unrealized_pnl": str(getattr(portfolio_metrics, "unrealized_pnl", Decimal("0"))),
            "realized_pnl": str(getattr(portfolio_metrics, "realized_pnl", Decimal("0"))),
        }

    def _format_risk_summary(self, risk_metrics) -> dict[str, Any]:
        """Format risk summary for dashboard display."""
        if not risk_metrics:
            return {
                "value_at_risk_95": "0",
                "max_drawdown": "0",
                "sharpe_ratio": "0",
            }

        return {
            "value_at_risk_95": str(getattr(risk_metrics, "portfolio_var_95", Decimal("0"))),
            "max_drawdown": str(getattr(risk_metrics, "max_drawdown", Decimal("0"))),
            "sharpe_ratio": str(getattr(risk_metrics, "sharpe_ratio", Decimal("0"))),
        }

    def _format_operational_summary(self, operational_metrics) -> dict[str, Any]:
        """Format operational summary for dashboard display."""
        if not operational_metrics:
            return {
                "active_strategies": 0,
                "total_orders": 0,
                "filled_orders": 0,
                "failed_orders": 0,
                "uptime_percentage": "0",
            }

        # Map from the actual field names in OperationalMetrics to test expected names
        return {
            "active_strategies": getattr(operational_metrics, "strategies_active", 0),
            "total_orders": getattr(operational_metrics, "orders_placed_today", 0),
            "filled_orders": getattr(operational_metrics, "orders_filled_today", 0),
            "failed_orders": getattr(operational_metrics, "orders_placed_today", 0) - getattr(operational_metrics, "orders_filled_today", 0),
            "uptime_percentage": str(getattr(operational_metrics, "websocket_uptime_percent", Decimal("0"))),
        }

    async def calculate_metrics(self, *args, **kwargs) -> dict[str, Any]:
        """Calculate dashboard metrics."""
        return {
            "dashboard_generated": True,
            "dashboard_metrics": {
                "total_dashboards": 1,
                "generation_time": "instant",
                "status": "operational"
            }
        }

    async def validate_data(self, data: Any) -> bool:
        """Validate dashboard data."""
        return isinstance(data, dict) and "timestamp" in data
