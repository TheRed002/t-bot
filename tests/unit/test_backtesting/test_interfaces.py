"""Tests for backtesting interfaces."""

import pytest
from typing import Any
from unittest.mock import MagicMock

from src.backtesting.interfaces import (
    BacktestServiceInterface,
    MetricsCalculatorInterface,
    BacktestAnalyzerInterface,
    BacktestEngineFactoryInterface,
    ComponentFactoryInterface
)
from src.core.base.interfaces import HealthCheckResult


class TestBacktestServiceInterface:
    """Test BacktestServiceInterface protocol."""
    
    def test_interface_methods_exist(self):
        """Test that interface methods are properly defined."""
        # Create a mock implementation
        mock_service = MagicMock(spec=BacktestServiceInterface)
        
        # Verify all methods exist
        assert hasattr(mock_service, 'initialize')
        assert hasattr(mock_service, 'run_backtest')
        assert hasattr(mock_service, 'get_active_backtests')
        assert hasattr(mock_service, 'cancel_backtest')
        assert hasattr(mock_service, 'clear_cache')
        assert hasattr(mock_service, 'get_cache_stats')
        assert hasattr(mock_service, 'health_check')
        assert hasattr(mock_service, 'cleanup')

    def test_interface_method_signatures(self):
        """Test interface method signatures are correctly defined."""
        # Test that methods can be called with expected parameters
        mock_service = MagicMock(spec=BacktestServiceInterface)
        
        # These should not raise errors for proper signatures
        mock_service.initialize.assert_not_called()
        mock_service.run_backtest.assert_not_called()
        mock_service.get_active_backtests.assert_not_called()
        mock_service.cancel_backtest.assert_not_called()
        mock_service.clear_cache.assert_not_called()
        mock_service.get_cache_stats.assert_not_called()
        mock_service.health_check.assert_not_called()
        mock_service.cleanup.assert_not_called()


class TestMetricsCalculatorInterface:
    """Test MetricsCalculatorInterface protocol."""
    
    def test_interface_methods_exist(self):
        """Test that interface methods are properly defined."""
        mock_calculator = MagicMock(spec=MetricsCalculatorInterface)
        
        # Verify method exists
        assert hasattr(mock_calculator, 'calculate_all')

    def test_calculate_all_signature(self):
        """Test calculate_all method signature."""
        mock_calculator = MagicMock(spec=MetricsCalculatorInterface)
        
        # Should accept the expected parameters
        equity_curve = [{"timestamp": "2024-01-01", "value": 1000.0}]
        trades = [{"id": "1", "pnl": 100.0}]
        daily_returns = [0.01, 0.02, -0.005]
        initial_capital = 10000.0
        
        # This should not raise an error
        mock_calculator.calculate_all(
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
            initial_capital=initial_capital
        )
        mock_calculator.calculate_all.assert_called_once()


class TestBacktestAnalyzerInterface:
    """Test BacktestAnalyzerInterface ABC."""
    
    def test_abstract_method_exists(self):
        """Test that abstract method is properly defined."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            BacktestAnalyzerInterface()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works."""
        class ConcreteAnalyzer(BacktestAnalyzerInterface):
            async def run_analysis(self, **kwargs) -> dict[str, Any]:
                return {"result": "success"}
        
        # Should be able to instantiate concrete implementation
        analyzer = ConcreteAnalyzer()
        assert analyzer is not None

    def test_missing_implementation_fails(self):
        """Test that missing implementation fails."""
        class IncompleteAnalyzer(BacktestAnalyzerInterface):
            pass
        
        # Should not be able to instantiate without implementing abstract method
        with pytest.raises(TypeError):
            IncompleteAnalyzer()


class TestBacktestEngineFactoryInterface:
    """Test BacktestEngineFactoryInterface protocol."""
    
    def test_interface_callable_exists(self):
        """Test that callable interface is properly defined."""
        mock_factory = MagicMock(spec=BacktestEngineFactoryInterface)
        
        # Should be callable
        assert callable(mock_factory)

    def test_factory_call_signature(self):
        """Test factory call signature."""
        mock_factory = MagicMock(spec=BacktestEngineFactoryInterface)
        
        # Should accept config, strategy, and kwargs
        config = MagicMock()
        strategy = MagicMock()
        
        mock_factory(config, strategy, extra_param="value")
        mock_factory.assert_called_once_with(config, strategy, extra_param="value")


class TestComponentFactoryInterface:
    """Test ComponentFactoryInterface protocol."""
    
    def test_interface_callable_exists(self):
        """Test that callable interface is properly defined."""
        mock_factory = MagicMock(spec=ComponentFactoryInterface)
        
        # Should be callable
        assert callable(mock_factory)

    def test_factory_call_no_params(self):
        """Test factory call with no parameters."""
        mock_factory = MagicMock(spec=ComponentFactoryInterface)
        
        # Should be callable with no parameters
        mock_factory()
        mock_factory.assert_called_once_with()


class TestInterfaceIntegration:
    """Test interface integration scenarios."""
    
    def test_interfaces_can_be_implemented_together(self):
        """Test that multiple interfaces can be implemented by same class."""
        class CombinedService:
            """Mock service implementing multiple interfaces."""
            
            async def initialize(self) -> None:
                pass
            
            async def run_backtest(self, request: Any) -> Any:
                return MagicMock()
            
            async def get_active_backtests(self) -> dict[str, dict[str, Any]]:
                return {}
            
            async def cancel_backtest(self, backtest_id: str) -> bool:
                return True
            
            async def clear_cache(self, pattern: str = "*") -> int:
                return 0
            
            async def get_cache_stats(self) -> dict[str, Any]:
                return {}
            
            async def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True, details={})
            
            async def cleanup(self) -> None:
                pass
            
            def calculate_all(
                self,
                equity_curve: list[dict[str, Any]],
                trades: list[dict[str, Any]],
                daily_returns: list[float],
                initial_capital: float,
            ) -> dict[str, Any]:
                return {}
        
        # Should be able to create instance
        service = CombinedService()
        assert service is not None

    def test_interface_type_checking(self):
        """Test that interfaces support proper type checking."""
        # Create mocks that conform to interfaces
        service_mock = MagicMock(spec=BacktestServiceInterface)
        calculator_mock = MagicMock(spec=MetricsCalculatorInterface)
        factory_mock = MagicMock(spec=BacktestEngineFactoryInterface)
        component_factory_mock = MagicMock(spec=ComponentFactoryInterface)
        
        # Should be able to assign to interface types
        service: BacktestServiceInterface = service_mock
        calculator: MetricsCalculatorInterface = calculator_mock
        engine_factory: BacktestEngineFactoryInterface = factory_mock
        comp_factory: ComponentFactoryInterface = component_factory_mock
        
        assert service is not None
        assert calculator is not None
        assert engine_factory is not None
        assert comp_factory is not None