"""
Integration validation tests for backtesting module.

This test suite validates that the backtesting module properly integrates
with other modules through correct dependency injection patterns and
service layer architecture.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pytest_asyncio
from src.backtesting.di_registration import register_backtesting_services
from src.backtesting.interfaces import BacktestServiceInterface
from src.backtesting.service import BacktestRequest
from src.core.dependency_injection import DependencyInjector
from src.core.config import Config
from src.core.exceptions import ServiceError
from src.utils.decimal_utils import to_decimal


class TestBacktestingIntegrationValidation:
    """Validate backtesting module integration patterns."""

    @pytest_asyncio.fixture
    async def mock_config(self):
        """Create mock config."""
        config = MagicMock(spec=Config)
        config.get = MagicMock(return_value={})
        config.backtest_cache = {}
        return config

    @pytest_asyncio.fixture
    async def mock_services(self):
        """Create properly mocked services with correct interface contracts."""
        # Mock DataService with proper interface
        data_service = AsyncMock()
        data_service.initialize = AsyncMock()
        data_service.get_market_data = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(timezone.utc),
                'symbol': 'BTCUSDT',
                'open': to_decimal('50000'),
                'high': to_decimal('50100'),
                'low': to_decimal('49900'),
                'close': to_decimal('50050'),
                'volume': to_decimal('100'),
            }
        ])
        data_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        # Mock ExecutionService
        execution_service = AsyncMock()
        execution_service.initialize = AsyncMock()
        execution_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        # Mock RiskService
        risk_service = AsyncMock()
        risk_service.initialize = AsyncMock()
        risk_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        # Mock StrategyService
        strategy_service = AsyncMock()
        strategy_service.initialize = AsyncMock()
        strategy_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        # Mock CacheService
        cache_service = AsyncMock()
        cache_service.initialize = AsyncMock()
        cache_service.get = AsyncMock(return_value=None)
        cache_service.set = AsyncMock()

        # Mock Repository
        repository = AsyncMock()
        repository.save_backtest_result = AsyncMock(return_value="test_id_123")

        return {
            'DataService': data_service,
            'ExecutionService': execution_service,
            'RiskService': risk_service,
            'StrategyService': strategy_service,
            'CacheService': cache_service,
            'BacktestRepositoryInterface': repository,
        }

    @pytest_asyncio.fixture
    async def injector(self, mock_config, mock_services):
        """Create dependency injector with backtesting services."""
        injector = DependencyInjector()

        # Register config
        injector.register_service("Config", lambda: mock_config, singleton=True)

        # Register mock services
        for service_name, service in mock_services.items():
            injector.register_service(service_name, lambda s=service: s, singleton=True)

        # Register backtesting services
        register_backtesting_services(injector)

        return injector

    async def test_service_dependency_resolution(self, injector):
        """Test that BacktestService correctly resolves dependencies via DI."""
        # Resolve service through interface
        service = injector.resolve("BacktestServiceInterface")
        assert service is not None
        assert isinstance(service, BacktestServiceInterface)

        # Verify service has correct dependencies
        await service.initialize()

        # Check that services were properly injected
        assert service.data_service is not None
        assert service.execution_service is not None
        assert service.risk_service is not None
        assert service.strategy_service is not None
        assert service._cache_service is not None
        assert service.repository is not None

    async def test_service_layer_architecture(self, injector):
        """Test that service follows proper layer architecture."""
        service = injector.resolve("BacktestServiceInterface")

        # Initialize service
        await service.initialize()

        # Verify service uses dependencies through interfaces, not direct access
        assert hasattr(service, 'data_service')
        assert hasattr(service, 'execution_service')
        assert hasattr(service, 'risk_service')
        assert hasattr(service, 'strategy_service')

        # Test service method delegation
        health_result = await service.health_check()
        assert health_result.status in ['healthy', 'degraded', 'unhealthy']

    async def test_controller_uses_service_interface(self, injector):
        """Test that BacktestController uses service through interface."""
        controller = injector.resolve("BacktestController")
        assert controller is not None

        # Verify controller has service interface
        assert hasattr(controller, '_backtest_service')
        service = controller._backtest_service
        assert isinstance(service, BacktestServiceInterface)

    async def test_factory_creates_proper_components(self, injector):
        """Test that factory creates components with proper dependencies."""
        factory = injector.resolve("BacktestFactory")
        assert factory is not None

        # Test factory can create service with dependencies
        service = factory.create_service()
        assert service is not None

        # Test factory can create controller
        controller = factory.create_controller()
        assert controller is not None

        # Test factory can create repository
        repository = factory.create_repository()
        assert repository is not None

    async def test_no_circular_dependencies(self, injector):
        """Test that there are no circular dependencies."""
        # This should not hang or throw circular dependency errors
        service = injector.resolve("BacktestServiceInterface")
        controller = injector.resolve("BacktestController")
        factory = injector.resolve("BacktestFactory")

        assert service is not None
        assert controller is not None
        assert factory is not None

    async def test_service_error_propagation(self, injector, mock_services):
        """Test that service errors are properly propagated."""
        # Mock service to throw error
        mock_services['DataService'].get_market_data = AsyncMock(
            side_effect=Exception("Data service error")
        )

        service = injector.resolve("BacktestServiceInterface")
        await service.initialize()

        # Create invalid request
        request = BacktestRequest(
            strategy_config={'name': 'test'},
            symbols=['INVALID'],
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
            initial_capital=to_decimal('1000'),
        )

        # Should handle error gracefully without exposing internal details
        with pytest.raises(ServiceError):
            await service.run_backtest(request)

    async def test_module_boundary_respect(self, injector):
        """Test that backtesting respects other module boundaries."""
        service = injector.resolve("BacktestServiceInterface")

        # Service should not directly access other module internals
        # Only use provided interfaces
        assert not hasattr(service, '_database_connection')
        assert not hasattr(service, '_redis_client')

        # Should use abstracted services
        assert hasattr(service, 'data_service')
        assert hasattr(service, 'execution_service')
        assert hasattr(service, '_cache_service')

    @pytest.mark.asyncio
    async def test_backtest_service_initialization_robustness(self, injector):
        """Test that service initializes robustly even with missing dependencies."""
        # Create injector with minimal dependencies
        minimal_injector = DependencyInjector()
        config = MagicMock(spec=Config)
        config.get = MagicMock(return_value={})

        minimal_injector.register_service("Config", lambda: config, singleton=True)

        # Register backtesting services without all dependencies
        register_backtesting_services(minimal_injector)

        # Should still create service, just with limited functionality
        service = minimal_injector.resolve("BacktestServiceInterface")
        assert service is not None

        # Should initialize without throwing errors
        await service.initialize()

    async def test_interface_compliance(self, injector):
        """Test that concrete implementations comply with interfaces."""
        service = injector.resolve("BacktestServiceInterface")

        # Check all interface methods are implemented
        assert hasattr(service, 'initialize')
        assert hasattr(service, 'run_backtest')
        assert hasattr(service, 'run_backtest_from_dict')
        assert hasattr(service, 'serialize_result')
        assert hasattr(service, 'get_active_backtests')
        assert hasattr(service, 'cancel_backtest')
        assert hasattr(service, 'clear_cache')
        assert hasattr(service, 'get_cache_stats')
        assert hasattr(service, 'health_check')
        assert hasattr(service, 'get_backtest_result')
        assert hasattr(service, 'list_backtest_results')
        assert hasattr(service, 'delete_backtest_result')
        assert hasattr(service, 'cleanup')

    async def test_dependency_injection_patterns(self, injector):
        """Test that proper DI patterns are followed."""
        # Constructor injection
        service = injector.resolve("BacktestServiceInterface")
        assert service is not None

        # Service locator pattern through factory
        factory = injector.resolve("BacktestFactory")
        created_service = factory.create_service()
        assert created_service is not None

        # Interface registration
        service_via_interface = injector.resolve("BacktestServiceInterface")
        assert service_via_interface is not None

    async def test_service_lifecycle_management(self, injector):
        """Test proper service lifecycle management."""
        service = injector.resolve("BacktestServiceInterface")

        # Service should start uninitialized
        assert service._initialized is False

        # Initialize service
        await service.initialize()
        assert service._initialized is True

        # Cleanup service
        await service.cleanup()

        # Should handle multiple cleanup calls
        await service.cleanup()


@pytest.mark.integration
class TestOptimizationBacktestingIntegration:
    """Test optimization module's integration with backtesting."""

    async def test_optimization_uses_interface(self):
        """Test that optimization module uses BacktestService through interface."""
        with patch('src.optimization.backtesting_integration.BacktestServiceInterface') as mock_interface:
            from src.optimization.backtesting_integration import BacktestIntegrationService

            mock_service = AsyncMock()
            service = BacktestIntegrationService(backtest_service=mock_service)
            assert service._backtest_service is mock_service

    async def test_strategy_evaluation_error_handling(self):
        """Test that strategy evaluation handles errors properly."""
        with patch('src.optimization.backtesting_integration.BacktestServiceInterface'):
            from src.optimization.backtesting_integration import BacktestIntegrationService
            from src.core.types import StrategyConfig, StrategyType

            # Mock backtest service that throws error
            mock_service = AsyncMock()
            mock_service.run_backtest = AsyncMock(side_effect=Exception("Backtest failed"))

            service = BacktestIntegrationService(backtest_service=mock_service)

            config = StrategyConfig(
                strategy_id="test",
                strategy_type=StrategyType.CUSTOM,
                name="test_strategy",
                symbol="BTCUSDT",
                timeframe="1h",
                enabled=True,
                parameters={}
            )

            # Should return poor performance instead of throwing error
            result = await service.evaluate_strategy(config)
            assert result['total_return'] < 0
            assert result['sharpe_ratio'] < 0


@pytest.mark.integration
class TestWebInterfaceBacktestingIntegration:
    """Test web interface integration with backtesting."""

    async def test_playground_uses_dependency_injection(self):
        """Test that playground API uses DI instead of globals."""
        from src.web_interface.api.playground import get_dependencies

        # Mock global injector
        with patch('src.web_interface.api.playground.get_global_injector') as mock_get_injector:
            mock_injector = MagicMock()
            mock_injector.resolve = MagicMock(side_effect=Exception("No service"))
            mock_get_injector.return_value = mock_injector

            deps = get_dependencies()

            # Should handle missing services gracefully
            assert deps['backtesting_service'] is None
            assert deps['strategy_factory'] is None
            assert deps['bot_orchestrator'] is None

    async def test_playground_handles_missing_services(self):
        """Test that playground handles missing services gracefully."""
        from src.web_interface.api.playground import get_dependencies

        # Mock failed injector resolution
        with patch('src.web_interface.api.playground.get_global_injector', side_effect=Exception()):
            deps = get_dependencies()

            # Should return None services without throwing error
            assert all(service is None for service in deps.values())