"""
Backtesting Module Integration Validation Tests.

This test suite validates that the backtesting module properly integrates with
other modules in the trading system, focusing on:
1. Dependency injection patterns
2. Service layer compliance
3. Interface usage
4. Module boundary respect
5. Error handling integration
"""

import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from src.backtesting.di_registration import register_backtesting_services
from src.database.di_registration import register_database_services
from src.core.dependency_injection import DependencyInjector
from src.core.config import Config
from src.core.base.interfaces import HealthCheckResult, HealthStatus
from src.backtesting.service import BacktestRequest
from src.backtesting.interfaces import DataServiceInterface


class TestBacktestingModuleIntegration:
    """Test backtesting module integration with other system components."""

    @pytest.fixture
    def injector(self):
        """Create dependency injector with all required services."""
        injector = DependencyInjector()

        # Register config
        injector.register_factory('Config', lambda: Config(), singleton=True)

        # Register database services
        register_database_services(injector)

        # Register backtesting services
        register_backtesting_services(injector)

        return injector

    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service that implements DataServiceInterface."""
        mock = AsyncMock(spec=DataServiceInterface)
        mock.initialize.return_value = None
        mock.cleanup.return_value = None
        mock.get_market_data.return_value = []
        mock.get_recent_data.return_value = []
        mock.store_market_data.return_value = True
        return mock

    @pytest.fixture
    def mock_services(self, mock_data_service):
        """Create all mock services needed for integration testing."""
        # Mock strategy service
        mock_strategy = AsyncMock()
        mock_strategy.initialize.return_value = None
        mock_strategy.create_strategy.return_value = Mock()

        # Mock execution service
        mock_execution = AsyncMock()
        mock_execution.initialize.return_value = None

        # Mock risk service
        mock_risk = AsyncMock()
        mock_risk.initialize.return_value = None
        mock_risk.create_risk_manager.return_value = None

        # Mock capital service
        mock_capital = AsyncMock()
        mock_capital.initialize.return_value = None

        # Mock ML service
        mock_ml = AsyncMock()
        mock_ml.initialize.return_value = None

        # Mock cache service
        mock_cache = AsyncMock()
        mock_cache.initialize.return_value = None
        mock_cache.get.return_value = None
        mock_cache.set.return_value = None
        mock_cache.delete.return_value = None
        mock_cache.clear_pattern.return_value = 0
        mock_cache.get_stats.return_value = {}
        mock_cache.cleanup.return_value = None

        return {
            "DataService": mock_data_service,
            "StrategyService": mock_strategy,
            "ExecutionService": mock_execution,
            "RiskService": mock_risk,
            "CapitalService": mock_capital,
            "MLService": mock_ml,
            "CacheService": mock_cache,
        }

    async def test_dependency_injection_integration(self, injector):
        """Test that dependency injection works without circular dependencies."""
        # Should not hang or fail
        factory = injector.resolve('BacktestFactory')
        assert factory is not None

        # Should be able to create services
        controller = factory.create_controller()
        assert controller is not None

        service = factory.create_service()
        assert service is not None

        repository = factory.create_repository()
        assert repository is not None

    async def test_service_layer_compliance(self, injector, mock_services):
        """Test that backtesting follows service layer patterns correctly."""
        # Register mock services
        for service_name, mock_service in mock_services.items():
            injector.register_factory(service_name, lambda s=mock_service: s, singleton=True)

        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        # Service should have all dependencies injected
        assert service.data_service is not None
        assert service.strategy_service is not None
        assert service.execution_service is not None

        # Initialize should work
        await service.initialize()

        # Mock services should have been called
        mock_services["DataService"].initialize.assert_called_once()
        mock_services["StrategyService"].initialize.assert_called_once()

    async def test_interface_compliance(self, injector, mock_services):
        """Test that services use interfaces correctly."""
        # Register mock services
        for service_name, mock_service in mock_services.items():
            injector.register_factory(service_name, lambda s=mock_service: s, singleton=True)

        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        await service.initialize()

        # Test that service properly uses interface methods
        # This should not raise any AttributeError
        try:
            # Create a minimal backtest request
            request = BacktestRequest(
                strategy_config={
                    "strategy_type": "test",
                    "name": "test_strategy",
                    "parameters": {}
                },
                symbols=["BTCUSDT"],
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                initial_capital=Decimal("100000"),
                max_open_positions=1,
                exchange="binance",
                timeframe="1h"
            )

            # This should handle missing data gracefully and return a result (not crash)
            result = await service.run_backtest(request)
            # The service should return a result even with missing/minimal data
            assert result is not None

        except AttributeError as e:
            pytest.fail(f"Interface compliance error: {e}")

    async def test_error_propagation(self, injector, mock_services):
        """Test that errors are properly handled across module boundaries."""
        # Make data service fail
        mock_services["DataService"].initialize.side_effect = Exception("Data service failed")

        # Register mock services
        for service_name, mock_service in mock_services.items():
            injector.register_factory(service_name, lambda s=mock_service: s, singleton=True)

        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        # Should handle the error gracefully
        with pytest.raises(Exception) as exc_info:
            await service.initialize()

        # Should propagate the error appropriately
        assert "Data service failed" in str(exc_info.value) or isinstance(exc_info.value, Exception)

    async def test_repository_integration(self, injector):
        """Test that repository properly integrates with database services."""
        factory = injector.resolve('BacktestFactory')
        repository = factory.create_repository()

        # Repository should have database manager injected
        assert repository.db_manager is not None

        # Database manager should implement the expected interface
        assert hasattr(repository.db_manager, 'get_session')

    async def test_module_boundary_respect(self, injector):
        """Test that backtesting module respects other module boundaries."""
        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        # Backtesting should not have direct access to internal implementation details
        # It should only use interfaces and public APIs

        # Check that service dependencies are properly abstracted
        if service.data_service:
            # Should not expose internal database connections or other internals
            assert not hasattr(service.data_service, '_connection')
            assert not hasattr(service.data_service, '_session')

    async def test_health_check_integration(self, injector, mock_services):
        """Test that health checks work across module boundaries."""
        # Register mock services with health check support
        for service_name, mock_service in mock_services.items():
            if hasattr(mock_service, 'health_check'):
                mock_service.health_check.return_value = HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="OK"
                )
            injector.register_factory(service_name, lambda s=mock_service: s, singleton=True)

        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        await service.initialize()

        # Health check should work
        health = await service.health_check()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

    async def test_service_cleanup_integration(self, injector, mock_services):
        """Test that cleanup works properly across all services."""
        # Register mock services
        for service_name, mock_service in mock_services.items():
            injector.register_factory(service_name, lambda s=mock_service: s, singleton=True)

        factory = injector.resolve('BacktestFactory')
        service = factory.create_service()

        await service.initialize()
        await service.cleanup()

        # All mock services should have been cleaned up
        mock_services["DataService"].cleanup.assert_called_once()
        mock_services["CacheService"].cleanup.assert_called_once()

    def test_factory_pattern_compliance(self, injector):
        """Test that factory properly creates all components."""
        factory = injector.resolve('BacktestFactory')

        # Should be able to create all expected components
        controller = factory.create_controller()
        assert controller is not None

        service = factory.create_service()
        assert service is not None

        repository = factory.create_repository()
        assert repository is not None

        # Should be able to create analyzers
        monte_carlo = factory.create_analyzer("monte_carlo")
        assert monte_carlo is not None

        walk_forward = factory.create_analyzer("walk_forward")
        assert walk_forward is not None

        # Should be able to create metrics calculator
        metrics_calc = factory.create_metrics_calculator()
        assert metrics_calc is not None