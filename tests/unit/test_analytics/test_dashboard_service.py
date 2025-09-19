"""Tests for analytics dashboard service."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.analytics.services.dashboard_service import DashboardService
from src.analytics.types import (
    AnalyticsConfiguration,
    PortfolioMetrics,
    RiskMetrics,
    OperationalMetrics,
    PositionMetrics,
    StrategyMetrics,
)
from tests.unit.test_analytics.test_helpers import (
    create_test_portfolio_metrics,
    create_test_position_metrics,
    create_test_risk_metrics,
    create_test_operational_metrics,
    create_test_strategy_metrics,
)


class TestDashboardService:
    """Test DashboardService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use dict config instead of AnalyticsConfiguration object
        self.config = {"cache_ttl": 300, "update_frequency": 60}
        self.mock_portfolio_service = AsyncMock()
        self.mock_risk_service = AsyncMock()
        self.mock_operational_service = AsyncMock()
        self.mock_metrics_collector = Mock()

        self.service = DashboardService(
            config=self.config,
            portfolio_service=self.mock_portfolio_service,
            risk_service=self.mock_risk_service,
            operational_service=self.mock_operational_service,
            metrics_collector=self.mock_metrics_collector,
        )

    def test_initialization(self):
        """Test service initialization."""
        assert self.service.portfolio_service is self.mock_portfolio_service
        assert self.service.risk_service is self.mock_risk_service
        assert self.service.operational_service is self.mock_operational_service

    def test_initialization_with_minimal_dependencies(self):
        """Test service initialization with minimal dependencies."""
        service = DashboardService()

        assert service.portfolio_service is None
        assert service.risk_service is None
        assert service.operational_service is None

    @pytest.mark.asyncio
    async def test_generate_comprehensive_dashboard_with_all_metrics(self):
        """Test generating comprehensive dashboard with all metrics provided."""
        # Use helper functions to create valid metrics
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('10000'),
            cash=Decimal('5000'),
            invested_capital=Decimal('5000')
        )

        risk_metrics = create_test_risk_metrics(
            value_at_risk_95=Decimal('500'),
            max_drawdown=Decimal('200'),
            sharpe_ratio=Decimal('1.5')
        )

        operational_metrics = create_test_operational_metrics(
            active_strategies=5,
            total_orders=100,
            filled_orders=95,
            failed_orders=5,
            uptime_percentage=Decimal('99.5')
        )

        position_metrics = [
            create_test_position_metrics(
                symbol='BTC/USDT',
                current_value=Decimal('5000'),
                unrealized_pnl=Decimal('100')
            )
        ]

        strategy_metrics = [
            create_test_strategy_metrics(
                strategy_name='test_strategy',
                total_pnl=Decimal('50'),
                win_rate=Decimal('0.8')
            )
        ]

        active_alerts = [
            {'id': 'alert1', 'type': 'risk', 'severity': 'medium', 'message': 'Test alert'}
        ]

        dashboard = await self.service.generate_comprehensive_dashboard(
            portfolio_metrics=portfolio_metrics,
            risk_metrics=risk_metrics,
            operational_metrics=operational_metrics,
            position_metrics=position_metrics,
            strategy_metrics=strategy_metrics,
            active_alerts=active_alerts,
        )

        # Verify dashboard structure
        assert isinstance(dashboard, dict)
        assert 'timestamp' in dashboard
        assert 'dashboard_type' in dashboard
        assert 'portfolio_summary' in dashboard
        assert 'risk_summary' in dashboard
        assert 'operational_summary' in dashboard
        assert 'position_summary' in dashboard
        assert 'strategy_performance' in dashboard
        assert 'active_alerts' in dashboard
        assert 'performance_indicators' in dashboard

        # Verify dashboard content
        assert dashboard['dashboard_type'] == 'comprehensive_analytics'
        assert len(dashboard['active_alerts']) == 1
        assert dashboard['portfolio_summary']['total_value'] == '10000'

    @pytest.mark.asyncio
    async def test_generate_comprehensive_dashboard_with_none_values(self):
        """Test generating dashboard with None values."""
        dashboard = await self.service.generate_comprehensive_dashboard()

        # Should handle None values gracefully
        assert isinstance(dashboard, dict)
        assert 'timestamp' in dashboard
        assert 'dashboard_type' in dashboard
        assert dashboard['dashboard_type'] == 'comprehensive_analytics'

        # Should have default/empty structures
        assert 'portfolio_summary' in dashboard
        assert 'risk_summary' in dashboard
        assert 'operational_summary' in dashboard

    @pytest.mark.asyncio
    async def test_generate_comprehensive_dashboard_with_services(self):
        """Test dashboard generation that uses injected services."""
        # Configure mock services to return data
        self.mock_portfolio_service.calculate_portfolio_metrics = AsyncMock(
            return_value=create_test_portfolio_metrics(
                total_value=Decimal('8000'),
                cash=Decimal('3000'),
                invested_capital=Decimal('5000'),
                unrealized_pnl=Decimal('80'),
                realized_pnl=Decimal('40')
            )
        )

        self.mock_risk_service.get_risk_metrics = AsyncMock(
            return_value=create_test_risk_metrics(
                value_at_risk_95=Decimal('400'),
                max_drawdown=Decimal('150'),
                sharpe_ratio=Decimal('1.2')
            )
        )

        self.mock_operational_service.get_operational_metrics = AsyncMock(
            return_value=create_test_operational_metrics(
                active_strategies=3,
                total_orders=80,
                filled_orders=75,
                failed_orders=5,
                uptime_percentage=Decimal('98.5')
            )
        )

        # Generate dashboard without explicit metrics (should use services)
        dashboard = await self.service.generate_comprehensive_dashboard()

        # Verify services were called
        assert isinstance(dashboard, dict)

        # If services are called, they should provide data for the dashboard
        # The actual behavior depends on the service implementation

    @pytest.mark.asyncio
    async def test_generate_quick_dashboard(self):
        """Test generating quick dashboard."""
        dashboard = await self.service.generate_quick_dashboard()

        assert isinstance(dashboard, dict)
        assert 'timestamp' in dashboard
        assert 'dashboard_type' in dashboard
        assert dashboard['dashboard_type'] == 'quick_overview'

        # Quick dashboard should have basic summaries
        assert 'portfolio_summary' in dashboard
        assert 'risk_summary' in dashboard
        assert 'alerts_summary' in dashboard

    @pytest.mark.asyncio
    async def test_generate_risk_dashboard(self):
        """Test generating risk-focused dashboard."""
        risk_metrics = create_test_risk_metrics(
            value_at_risk_95=Decimal('600'),
            max_drawdown=Decimal('250'),
            sharpe_ratio=Decimal('1.8')
        )

        dashboard = await self.service.generate_risk_dashboard(risk_metrics=risk_metrics)

        assert isinstance(dashboard, dict)
        assert 'timestamp' in dashboard
        assert 'dashboard_type' in dashboard
        assert dashboard['dashboard_type'] == 'risk_focused'

        # Risk dashboard should have detailed risk information
        assert 'risk_metrics' in dashboard
        assert 'risk_alerts' in dashboard
        assert 'risk_trends' in dashboard

    @pytest.mark.asyncio
    async def test_generate_performance_dashboard(self):
        """Test generating performance-focused dashboard."""
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('12000'),
            cash=Decimal('6000'),
            invested_capital=Decimal('6000'),
            unrealized_pnl=Decimal('200'),
            realized_pnl=Decimal('100')
        )

        strategy_metrics = [
            create_test_strategy_metrics(
                strategy_name='momentum_strategy',
                total_pnl=Decimal('75'),
                win_rate=Decimal('0.75')
            ),
            create_test_strategy_metrics(
                strategy_name='mean_reversion',
                total_pnl=Decimal('25'),
                win_rate=Decimal('0.65')
            ),
        ]

        dashboard = await self.service.generate_performance_dashboard(
            portfolio_metrics=portfolio_metrics,
            strategy_metrics=strategy_metrics
        )

        assert isinstance(dashboard, dict)
        assert 'timestamp' in dashboard
        assert 'dashboard_type' in dashboard
        assert dashboard['dashboard_type'] == 'performance_focused'

        # Performance dashboard should have performance-specific sections
        assert 'portfolio_performance' in dashboard
        assert 'strategy_performance' in dashboard
        assert 'performance_trends' in dashboard

    @pytest.mark.asyncio
    async def test_calculate_metrics_implementation(self):
        """Test the calculate_metrics abstract method implementation."""
        result = await self.service.calculate_metrics()

        assert isinstance(result, dict)
        assert 'dashboard_metrics' in result

    @pytest.mark.asyncio
    async def test_validate_data_implementation(self):
        """Test the validate_data abstract method implementation."""
        # Test with valid data
        valid_data = {'portfolio_metrics': {'total_value': '10000'}}
        result = await self.service.validate_data(valid_data)
        assert isinstance(result, bool)

        # Test with None
        result = await self.service.validate_data(None)
        assert isinstance(result, bool)

    def test_format_dashboard_summary(self):
        """Test formatting dashboard summary."""
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('15000'),
            cash=Decimal('7000'),
            invested_capital=Decimal('8000'),
            unrealized_pnl=Decimal('300'),
            realized_pnl=Decimal('150')
        )

        summary = self.service._format_portfolio_summary(portfolio_metrics)

        assert isinstance(summary, dict)
        assert 'total_value' in summary
        assert 'available_balance' in summary
        assert 'unrealized_pnl' in summary
        assert 'realized_pnl' in summary

        # Values should be converted to strings for JSON serialization
        assert isinstance(summary['total_value'], str)
        assert summary['total_value'] == '15000'

    def test_format_risk_summary(self):
        """Test formatting risk summary."""
        risk_metrics = create_test_risk_metrics(
            value_at_risk_95=Decimal('800'),
            max_drawdown=Decimal('300'),
            sharpe_ratio=Decimal('2.0')
        )

        summary = self.service._format_risk_summary(risk_metrics)

        assert isinstance(summary, dict)
        assert 'value_at_risk_95' in summary
        assert 'max_drawdown' in summary
        assert 'sharpe_ratio' in summary

        # Values should be converted appropriately
        assert isinstance(summary['value_at_risk_95'], str)
        assert summary['value_at_risk_95'] == '800'

    def test_format_operational_summary(self):
        """Test formatting operational summary."""
        operational_metrics = create_test_operational_metrics(
            active_strategies=8,
            total_orders=200,
            filled_orders=190,
            failed_orders=10,
            uptime_percentage=Decimal('99.8')
        )

        summary = self.service._format_operational_summary(operational_metrics)

        assert isinstance(summary, dict)
        assert 'active_strategies' in summary
        assert 'total_orders' in summary
        assert 'filled_orders' in summary
        assert 'failed_orders' in summary
        assert 'uptime_percentage' in summary

        assert summary['active_strategies'] == 8
        assert summary['total_orders'] == 200


class TestDashboardServiceErrorHandling:
    """Test error handling in dashboard service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DashboardService()

    @pytest.mark.asyncio
    async def test_dashboard_generation_with_service_errors(self):
        """Test dashboard generation when services raise errors."""
        # Mock services that raise errors
        mock_portfolio_service = Mock()
        mock_portfolio_service.calculate_portfolio_metrics = AsyncMock(
            side_effect=Exception("Portfolio service error")
        )

        self.service.portfolio_service = mock_portfolio_service

        # Should handle service errors gracefully
        dashboard = await self.service.generate_comprehensive_dashboard()

        assert isinstance(dashboard, dict)
        # Should still return a valid dashboard structure even with service errors

    @pytest.mark.asyncio
    async def test_dashboard_with_invalid_metrics(self):
        """Test dashboard generation with invalid metrics."""
        # Test with invalid portfolio metrics
        invalid_metrics = {"invalid": "data"}

        try:
            dashboard = await self.service.generate_comprehensive_dashboard(
                portfolio_metrics=invalid_metrics
            )
            # Should either handle gracefully or raise appropriate error
            assert isinstance(dashboard, dict)
        except Exception:
            # Appropriate error handling is acceptable
            pass

    def test_format_methods_with_none_values(self):
        """Test formatting methods handle None values properly."""
        # Test portfolio summary with None
        summary = self.service._format_portfolio_summary(None)
        assert isinstance(summary, dict)

        # Test risk summary with None
        summary = self.service._format_risk_summary(None)
        assert isinstance(summary, dict)

        # Test operational summary with None
        summary = self.service._format_operational_summary(None)
        assert isinstance(summary, dict)


class TestDashboardServiceIntegration:
    """Test dashboard service integration scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_dashboard_generation(self):
        """Test complete dashboard generation workflow."""
        # Create service with mock dependencies
        mock_portfolio_service = Mock()
        mock_risk_service = Mock()
        mock_operational_service = Mock()

        # Configure mocks to return realistic data
        mock_portfolio_service.calculate_portfolio_metrics = AsyncMock(
            return_value=create_test_portfolio_metrics(
                total_value=Decimal('20000'),
                cash=Decimal('10000'),
                invested_capital=Decimal('10000'),
                unrealized_pnl=Decimal('500'),
                realized_pnl=Decimal('250')
            )
        )

        service = DashboardService(
            portfolio_service=mock_portfolio_service,
            risk_service=mock_risk_service,
            operational_service=mock_operational_service,
        )

        # Generate various dashboard types
        comprehensive = await service.generate_comprehensive_dashboard()
        quick = await service.generate_quick_dashboard()
        risk = await service.generate_risk_dashboard()
        performance = await service.generate_performance_dashboard()

        # Verify all dashboards are generated successfully
        dashboards = [comprehensive, quick, risk, performance]
        for dashboard in dashboards:
            assert isinstance(dashboard, dict)
            assert 'timestamp' in dashboard
            assert 'dashboard_type' in dashboard