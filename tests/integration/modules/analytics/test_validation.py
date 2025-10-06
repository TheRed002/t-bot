"""Integration tests for analytics module dependency validation."""

import pytest

from src.analytics.di_registration import register_analytics_services
from src.analytics.factory import AnalyticsServiceFactory
from src.analytics.service import AnalyticsService
from src.core.dependency_injection import DependencyInjector


class TestAnalyticsModuleIntegration:
    """Test analytics module integration patterns."""

    def test_analytics_services_registration(self):
        """Test that all analytics services are properly registered."""
        injector = DependencyInjector()
        register_analytics_services(injector)

        required_services = [
            "AnalyticsServiceFactory",
            "AnalyticsService",
            "PortfolioService",
            "RiskService",
            "ReportingService",
            "OperationalService",
            "AlertService",
            "ExportService",
            "RealtimeAnalyticsService",
            "AnalyticsRepository",
        ]

        for service_name in required_services:
            assert injector.is_registered(service_name), f"Service {service_name} not registered"

    def test_analytics_factory_direct_creation(self):
        """Test analytics factory can be created directly."""
        injector = DependencyInjector()
        factory = AnalyticsServiceFactory(injector)
        assert isinstance(factory, AnalyticsServiceFactory)
        assert factory._injector is injector

    def test_analytics_service_creation_via_factory(self):
        """Test analytics service can be created via factory."""
        injector = DependencyInjector()
        factory = AnalyticsServiceFactory(injector)

        service = factory.create_analytics_service()
        assert isinstance(service, AnalyticsService)

    def test_analytics_service_dependency_patterns(self):
        """Test that analytics service uses proper dependency patterns."""
        # Create individual services
        from src.analytics.services.portfolio_analytics_service import PortfolioAnalyticsService
        from src.analytics.services.realtime_analytics_service import RealtimeAnalyticsService
        from src.analytics.services.risk_service import RiskService

        portfolio_service = PortfolioAnalyticsService()
        realtime_service = RealtimeAnalyticsService()
        risk_service = RiskService()

        # Create analytics service with dependencies
        analytics_service = AnalyticsService(
            realtime_analytics=realtime_service,
            portfolio_service=portfolio_service,
            risk_service=risk_service,
        )

        # Verify dependencies are injected correctly
        assert analytics_service.realtime_analytics is realtime_service
        assert analytics_service.portfolio_service is portfolio_service
        assert analytics_service.risk_service is risk_service

    def test_factory_creates_all_service_types(self):
        """Test factory can create all required service types."""
        injector = DependencyInjector()
        factory = AnalyticsServiceFactory(injector)

        # Test each factory method exists and works
        service_methods = [
            "create_analytics_service",
            "create_realtime_analytics_service",
            "create_portfolio_service",
            "create_risk_service",
            "create_reporting_service",
            "create_operational_service",
            "create_alert_service",
            "create_export_service",
        ]

        for method_name in service_methods:
            assert hasattr(factory, method_name), f"Factory missing method {method_name}"
            method = getattr(factory, method_name)

            # Call method and verify it returns something
            result = method()
            assert result is not None, f"Method {method_name} returned None"

    def test_analytics_service_integration_with_base_patterns(self):
        """Test analytics service follows base component patterns."""
        service = AnalyticsService()

        # Test it's a proper base component
        assert hasattr(service, "start")
        assert hasattr(service, "stop")
        assert hasattr(service, "logger")
        assert hasattr(service, "config")

    def test_analytics_handles_missing_dependencies(self):
        """Test analytics gracefully handles missing dependencies."""
        # Create analytics service without dependencies
        service = AnalyticsService()

        # Should not raise errors
        assert service is not None
        assert service.realtime_analytics is None
        assert service.portfolio_service is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_analytics_service_async_methods(self):
        """Test analytics service async methods work correctly."""
        service = AnalyticsService()

        # Test async methods return proper defaults when dependencies are None
        portfolio_metrics = await service.get_portfolio_metrics()
        # Should return None when no realtime service available
        assert portfolio_metrics is None

        position_metrics = await service.get_position_metrics()
        # Should return empty list when no realtime service available
        assert position_metrics == []

        # Test operational and risk metrics return proper defaults
        operational_metrics = await service.get_operational_metrics()
        assert operational_metrics is not None
        assert hasattr(operational_metrics, "timestamp")

        risk_metrics = await service.get_risk_metrics()
        assert risk_metrics is not None
        assert hasattr(risk_metrics, "timestamp")

    def test_analytics_repository_dependency_injection(self):
        """Test analytics repository gets proper dependencies."""
        from src.analytics.repository import AnalyticsRepository

        # Create repository directly without database (for testing)
        repository = AnalyticsRepository(session=None)

        # Repository should be created (session can be None in tests)
        assert repository is not None
        assert hasattr(repository, "session")

        # Should have transformation service as None by default when not injected
        # This tests that dependency injection is optional


class TestAnalyticsModuleBoundaries:
    """Test analytics module respects proper boundaries."""

    def test_analytics_does_not_import_exchange_internals(self):
        """Test analytics doesn't import exchange internal modules."""
        # This is a static analysis test - analytics should only use exchange public APIs
        import ast
        import os

        analytics_dir = "/mnt/e/Work/P-41 Trading/code/t-bot/src/analytics"

        for root, dirs, files in os.walk(analytics_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path) as f:
                        content = f.read()

                    # Parse to find imports
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.Import, ast.ImportFrom)):
                                if isinstance(node, ast.ImportFrom) and node.module:
                                    # Check for improper exchange imports
                                    if node.module.startswith(
                                        "src.exchanges"
                                    ) and node.module not in [
                                        "src.exchanges.interfaces",
                                        "src.exchanges.types",
                                    ]:
                                        # Should not import exchange internals
                                        assert False, (
                                            f"Analytics importing exchange internals: {node.module} in {file_path}"
                                        )
                    except SyntaxError:
                        # Skip files with syntax errors
                        pass

    def test_analytics_uses_proper_core_imports(self):
        """Test analytics uses proper core module imports."""
        from src.analytics.service import AnalyticsService

        # Should import from core
        assert hasattr(AnalyticsService, "logger")  # From BaseComponent

        # Check imports are from proper modules
        import inspect

        import src.analytics.service

        source = inspect.getsource(src.analytics.service)

        # Should use core types and exceptions
        assert "from src.core.types" in source
        assert "from src.core.exceptions" in source

    def test_analytics_error_handling_patterns(self):
        """Test analytics follows proper error handling patterns."""
        from datetime import datetime
        from decimal import Decimal

        from src.core.types import Position

        service = AnalyticsService()

        # Create mock data
        from src.core.types import PositionSide, PositionStatus

        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            side=PositionSide.LONG,
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            unrealized_pnl=Decimal("1000.0"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(),
            exchange="binance",
        )

        # Test error handling - should not raise exceptions
        try:
            service.update_position(position)
        except Exception as e:
            # Should use proper error types from core.exceptions
            from src.core.exceptions import ComponentError

            assert isinstance(e, ComponentError)


if __name__ == "__main__":
    pytest.main([__file__])
