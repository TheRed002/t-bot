"""
Simple integration tests for backtesting module integration verification.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

import pytest

from src.backtesting.service import BacktestRequest, BacktestService
from src.backtesting.interfaces import BacktestServiceInterface
from src.core.config import Config
from src.core.exceptions import BacktestError
from src.utils.decimal_utils import to_decimal


class TestBacktestingIntegrationSimple:
    """Simple integration tests for backtesting module."""

    def test_backtest_service_interface_compliance(self):
        """Test BacktestService implements required interface."""
        # Create minimal config
        config = MagicMock(spec=Config)
        config.get = MagicMock(return_value={})
        config.backtest_cache = {}
        
        # Create service with minimal dependencies
        service = BacktestService(config=config)
        
        # Verify interface compliance
        assert isinstance(service, BacktestServiceInterface)
        assert hasattr(service, 'run_backtest')
        assert hasattr(service, 'get_active_backtests')
        assert hasattr(service, 'cancel_backtest')
        assert hasattr(service, 'clear_cache')
        assert hasattr(service, 'health_check')

    def test_backtest_request_validation(self):
        """Test BacktestRequest validates input properly."""
        # Valid request
        request = BacktestRequest(
            strategy_config={'strategy_type': 'test'},
            symbols=['BTCUSDT'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_capital=to_decimal('10000')
        )
        assert request.symbols == ['BTCUSDT']
        assert request.initial_capital == to_decimal('10000')

        # Invalid date range should fail
        with pytest.raises(ValueError, match="End date must be after start date"):
            BacktestRequest(
                strategy_config={'strategy_type': 'test'},
                symbols=['BTCUSDT'],
                start_date=datetime(2023, 1, 31),
                end_date=datetime(2023, 1, 1),  # Before start date
                initial_capital=to_decimal('10000')
            )

    def test_service_layer_pattern_enforcement(self):
        """Test that service layer patterns are enforced."""
        # BacktestService should not have direct database access
        config = MagicMock(spec=Config)
        config.get = MagicMock(return_value={})
        config.backtest_cache = {}
        
        service = BacktestService(config=config)
        
        # Should not have direct database dependencies
        assert not hasattr(service, '_database')
        assert not hasattr(service, '_repository')
        assert not hasattr(service, '_db_connection')
        
        # Should use service layer dependencies
        assert hasattr(service, 'data_service')
        assert hasattr(service, 'execution_service')
        assert hasattr(service, 'strategy_service')

    def test_strategy_service_backtest_integration(self):
        """Test StrategyService properly integrates with BacktestService."""
        from src.strategies.service import StrategyService
        
        # Create strategy service with service manager
        mock_service_manager = MagicMock()
        mock_backtest_service = AsyncMock()
        mock_service_manager.get_service.return_value = mock_backtest_service
        
        strategy_service = StrategyService(
            name="test",
            config={},
            service_manager=mock_service_manager
        )
        
        # Mock strategy and result
        strategy_service._active_strategies['test'] = MagicMock()
        strategy_service._active_strategies['test'].config = MagicMock()
        strategy_service._active_strategies['test'].config.model_dump = MagicMock(return_value={})
        
        result = MagicMock()
        result.total_return = to_decimal('0.1')
        result.sharpe_ratio = 1.5
        result.model_dump = MagicMock(return_value={})
        mock_backtest_service.run_backtest.return_value = result
        
        # Test that service manager is used properly
        @pytest.mark.asyncio
        async def test_run():
            await strategy_service._run_backtest_impl('test', {
                'symbols': ['BTCUSDT'],
                'start_date': datetime.now(timezone.utc) - timedelta(days=30),
                'end_date': datetime.now(timezone.utc)
            })
            
        asyncio.run(test_run())
        
        # Verify service locator pattern was used
        mock_service_manager.get_service.assert_called_with("BacktestService")
        mock_backtest_service.run_backtest.assert_called_once()

    def test_module_boundary_violations_check(self):
        """Test for module boundary violations."""
        import src.backtesting.service as service_module
        
        # Read source code to check for boundary violations
        import inspect
        source = inspect.getsource(service_module)
        
        # Should not directly import database models or repositories
        assert 'from src.database.models' not in source
        assert 'from src.database.repository' not in source
        
        # Should use proper service interfaces
        assert 'DataService' in source or 'data_service' in source

    def test_circular_dependency_prevention(self):
        """Test circular dependency prevention."""
        from src.backtesting.di_registration import register_backtesting_services
        from src.core.dependency_injection import DependencyInjector
        
        # This should work without circular dependency issues
        injector = DependencyInjector()
        config = MagicMock()
        injector.register_service("Config", lambda: config, singleton=True)
        
        # Register backtesting services
        register_backtesting_services(injector)
        
        # Should be able to resolve services
        assert injector.has_service("BacktestService")
        assert injector.has_service("MetricsCalculator")

    def test_optimization_module_uses_service_layer(self):
        """Test optimization module uses proper service layer patterns."""
        import src.optimization.service as opt_module
        import inspect

        source = inspect.getsource(opt_module)

        # Check that the module uses service layer patterns
        # Should have a service class that inherits from BaseService
        assert 'BaseService' in source or 'class OptimizationService' in source

        # Should not directly instantiate engines - should use dependency injection
        direct_engine_instantiations = [
            line.strip() for line in source.split('\n')
            if 'Engine(' in line and not ('import' in line or 'from' in line or '#' in line)
        ]
        assert len(direct_engine_instantiations) == 0, f"Found direct engine instantiations: {direct_engine_instantiations}"