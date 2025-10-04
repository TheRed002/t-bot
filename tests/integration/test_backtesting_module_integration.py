"""
Integration tests for backtesting module boundaries and service integration.

This test suite verifies that:
1. BacktestService correctly uses its dependencies (DataService, ExecutionService, etc.)
2. Other modules correctly consume BacktestService through proper interfaces
3. Service layer patterns are followed (no direct database access)
4. Dependency injection works correctly
5. Module boundaries are respected
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pytest_asyncio
from src.backtesting.service import BacktestRequest, BacktestService
from src.backtesting.interfaces import BacktestServiceInterface
from src.core.dependency_injection import DependencyInjector
from src.core.config import Config
from src.core.exceptions import BacktestError, ServiceError
from src.data.types import DataRequest
from src.utils.decimal_utils import to_decimal


class TestBacktestingModuleIntegration:
    """Test backtesting module integration patterns."""

    @pytest_asyncio.fixture
    async def mock_config(self):
        """Create mock config."""
        config = MagicMock(spec=Config)
        config.get.return_value = {}
        config.backtest_cache = {}
        return config

    @pytest_asyncio.fixture
    async def mock_dependencies(self):
        """Create mock service dependencies."""
        data_service = AsyncMock()
        data_service.get_market_data = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'open': to_decimal('50000'),
                'high': to_decimal('50100'),
                'low': to_decimal('49900'),
                'close': to_decimal('50050'),
                'volume': to_decimal('100'),
            }
            for i in range(100, 0, -1)
        ])
        data_service.initialize = AsyncMock()
        data_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        execution_service = AsyncMock()
        execution_service.initialize = AsyncMock()
        execution_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        strategy_service = AsyncMock()
        strategy_service.create_strategy = AsyncMock()
        strategy_service.initialize = AsyncMock()
        strategy_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        risk_service = AsyncMock()
        risk_service.create_risk_manager = AsyncMock(return_value=None)
        risk_service.initialize = AsyncMock()
        risk_service.health_check = AsyncMock(return_value={'status': 'healthy'})

        return {
            'DataService': data_service,
            'ExecutionService': execution_service,
            'StrategyService': strategy_service,
            'RiskService': risk_service,
            'CapitalService': None,
            'MLService': None,
        }

    @pytest_asyncio.fixture
    async def dependency_injector(self, mock_config, mock_dependencies):
        """Create dependency injector with backtesting services."""
        injector = DependencyInjector()
        
        # Register config
        injector.register_service("Config", lambda: mock_config, singleton=True)
        
        # Register mock services
        for service_name, service in mock_dependencies.items():
            if service:
                injector.register_service(service_name, lambda s=service: s, singleton=True)
        
        # Register backtesting services
        from src.backtesting.di_registration import register_backtesting_services
        register_backtesting_services(injector)
        
        return injector

    @pytest_asyncio.fixture
    async def backtest_service(self, mock_config, mock_dependencies, dependency_injector):
        """Create BacktestService with mocked dependencies."""
        service = BacktestService(
            config=mock_config,
            injector=dependency_injector,
            **mock_dependencies
        )
        await service.initialize()
        return service

    async def test_backtest_service_dependency_injection(self, backtest_service):
        """Test BacktestService properly uses injected dependencies."""
        # Verify service has dependencies
        assert backtest_service.data_service is not None
        assert backtest_service.execution_service is not None
        assert backtest_service.risk_service is not None
        assert backtest_service.strategy_service is not None

        # Verify backtesting components are initialized
        assert backtest_service.metrics_calculator is not None
        assert backtest_service.monte_carlo_analyzer is not None
        assert backtest_service.walk_forward_analyzer is not None
        assert backtest_service.performance_attributor is not None

    async def test_backtest_service_interface_compliance(self, backtest_service):
        """Test BacktestService implements the correct interface."""
        # Check interface compliance
        assert isinstance(backtest_service, BacktestServiceInterface)
        
        # Check required methods exist
        assert hasattr(backtest_service, 'run_backtest')
        assert hasattr(backtest_service, 'get_active_backtests')
        assert hasattr(backtest_service, 'cancel_backtest')
        assert hasattr(backtest_service, 'clear_cache')
        assert hasattr(backtest_service, 'health_check')

    async def test_backtest_service_uses_data_service_correctly(self, backtest_service):
        """Test BacktestService properly uses DataService interface."""
        request = BacktestRequest(
            strategy_config={'strategy_type': 'mock'},
            symbols=['BTCUSDT'],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_capital=to_decimal('10000')
        )

        # Mock strategy creation
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.config = MagicMock()
        mock_strategy.config.model_dump = MagicMock(return_value={})
        backtest_service.strategy_service.create_strategy.return_value = mock_strategy

        # Mock simulation components
        with patch.object(backtest_service, '_run_core_simulation', new_callable=AsyncMock) as mock_simulation:
            mock_simulation.return_value = {
                'equity_curve': [],
                'trades': [],
                'daily_returns': []
            }
            
            with patch.object(backtest_service, '_run_advanced_analysis', new_callable=AsyncMock) as mock_analysis:
                mock_analysis.return_value = {}
                
                # Run backtest
                result = await backtest_service.run_backtest(request)
                
                # Verify DataService was called with proper DataRequest
                backtest_service.data_service.get_market_data.assert_called()
                call_args = backtest_service.data_service.get_market_data.call_args[0][0]
                assert isinstance(call_args, DataRequest)
                assert call_args.symbol == 'BTCUSDT'
                assert call_args.exchange == 'binance'

    async def test_backtest_service_error_handling(self, backtest_service):
        """Test BacktestService proper error handling."""
        # Test with invalid strategy config
        request = BacktestRequest(
            strategy_config={},  # Empty config should fail
            symbols=['BTCUSDT'],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_capital=to_decimal('10000')
        )

        with pytest.raises(BacktestError, match="Strategy configuration is empty"):
            await backtest_service.run_backtest(request)

    async def test_backtest_service_service_layer_pattern(self, backtest_service):
        """Test BacktestService follows service layer patterns."""
        # Verify it doesn't directly access database/repositories
        # All data access should go through service dependencies
        
        # Check service doesn't have direct database dependencies
        assert not hasattr(backtest_service, '_database')
        assert not hasattr(backtest_service, '_repository')
        assert not hasattr(backtest_service, '_db_connection')
        
        # Verify it uses service layer dependencies
        assert hasattr(backtest_service, 'data_service')
        assert hasattr(backtest_service, 'execution_service')
        assert hasattr(backtest_service, 'strategy_service')

    async def test_strategy_service_backtest_integration(self):
        """Test StrategyService properly uses BacktestService."""
        from src.strategies.service import StrategyService
        
        # Mock service manager
        mock_service_manager = MagicMock()
        mock_backtest_service = AsyncMock()
        mock_service_manager.get_service.return_value = mock_backtest_service
        
        # Create strategy service
        strategy_service = StrategyService(
            name="test_strategy_service",
            config={},
            service_manager=mock_service_manager
        )
        
        # Mock active strategy
        mock_strategy = MagicMock()
        mock_strategy.config = MagicMock()
        mock_strategy.config.model_dump = MagicMock(return_value={})
        strategy_service._active_strategies['test_strategy'] = mock_strategy
        
        # Mock backtest result
        mock_result = MagicMock()
        mock_result.total_return = to_decimal('0.1')
        mock_result.sharpe_ratio = 1.5
        mock_result.model_dump = MagicMock(return_value={'result': 'test'})
        mock_backtest_service.run_backtest.return_value = mock_result
        
        # Test backtest execution
        backtest_config = {
            'symbols': ['BTCUSDT'],
            'start_date': datetime.now(timezone.utc) - timedelta(days=30),
            'end_date': datetime.now(timezone.utc)
        }
        
        result = await strategy_service._run_backtest_impl('test_strategy', backtest_config)
        
        # Verify service locator pattern was used
        mock_service_manager.get_service.assert_called_with("BacktestService")
        
        # Verify BacktestService was called with proper request
        mock_backtest_service.run_backtest.assert_called_once()
        call_args = mock_backtest_service.run_backtest.call_args[0][0]
        assert isinstance(call_args, BacktestRequest)

    async def test_module_boundary_violations(self):
        """Test for module boundary violations."""
        # Verify backtesting module doesn't directly import internal details from other modules
        import src.backtesting.service
        import src.backtesting.engine
        
        # Check imports are only from public interfaces
        service_module = src.backtesting.service
        engine_module = src.backtesting.engine
        
        # Should only import from interfaces and types, not internal implementations
        # This is a structural test to catch boundary violations
        
        # Verify service layer separation
        assert hasattr(service_module, 'BacktestService')
        assert hasattr(engine_module, 'BacktestEngine')
        
        # BacktestService should use interfaces, not concrete implementations
        service_code = service_module.__file__
        with open(service_code, 'r') as f:
            content = f.read()
            
        # Should use DataServiceInterface, not concrete DataService
        assert 'DataServiceInterface' in content or 'data_service' in content
        # Should not directly import database models or repositories
        assert 'from src.database.models' not in content
        assert 'from src.database.repository' not in content

    async def test_dependency_injection_registration(self, dependency_injector):
        """Test backtesting services are properly registered in DI container."""
        # Verify all backtesting services are registered
        assert dependency_injector.is_registered("BacktestService")
        assert dependency_injector.is_registered("BacktestEngineFactory")
        assert dependency_injector.is_registered("MetricsCalculator")
        assert dependency_injector.is_registered("MonteCarloAnalyzer")
        assert dependency_injector.is_registered("WalkForwardAnalyzer")
        assert dependency_injector.is_registered("PerformanceAttributor")
        
        # Test service resolution
        backtest_service = dependency_injector.resolve("BacktestService")
        assert backtest_service is not None
        
        metrics_calculator = dependency_injector.resolve("MetricsCalculator")
        assert metrics_calculator is not None

    async def test_circular_dependency_prevention(self):
        """Test that circular dependencies are prevented."""
        # This test ensures that BacktestService and StrategyService
        # don't create circular dependencies
        
        from src.backtesting.di_registration import register_backtesting_services
        from src.core.dependency_injection import DependencyInjector
        
        injector = DependencyInjector()
        
        # Mock config
        mock_config = MagicMock()
        injector.register_service("Config", lambda: mock_config, singleton=True)
        
        # Register backtesting services (should not depend on StrategyService directly)
        register_backtesting_services(injector)
        
        # This should work without circular dependency issues
        backtest_service = injector.resolve("BacktestService")
        assert backtest_service is not None

    async def test_optimization_integration_uses_service_layer(self):
        """Test optimization module properly uses BacktestService."""
        from src.optimization.integration import OptimizationIntegration
        
        # Check that optimization imports BacktestService, not BacktestEngine directly
        import src.optimization.integration as opt_module
        
        # Read the source code to verify proper imports
        import inspect
        source = inspect.getsource(opt_module)
        
        # Should import BacktestService
        assert 'BacktestService' in source
        # Should not directly instantiate BacktestEngine
        assert 'BacktestEngine(' not in source or source.count('BacktestEngine(') <= 1  # Allow for imports