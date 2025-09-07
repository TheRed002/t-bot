"""
Unit tests for optimization interfaces module.

Tests the protocol definitions and abstract base classes for the optimization
framework to ensure proper type checking and interface compliance.
"""

import pytest
from abc import ABC
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, AsyncMock
import asyncio

from src.optimization.interfaces import (
    OptimizationServiceProtocol,
    BacktestIntegrationProtocol,
    OptimizationAnalysisProtocol,
    OptimizationRepositoryProtocol,
    IOptimizationService,
    IBacktestIntegrationService,
)
from src.optimization.core import OptimizationObjective, OptimizationResult, ObjectiveDirection
from src.optimization.parameter_space import ParameterSpace
from src.core.types import StrategyConfig


class TestOptimizationServiceProtocol:
    """Test OptimizationServiceProtocol."""

    def test_protocol_structure(self):
        """Test protocol has required methods."""
        # Create mock that implements the protocol
        mock_service = Mock(spec=OptimizationServiceProtocol)
        
        # Check that protocol methods exist
        assert hasattr(mock_service, 'optimize_strategy')
        assert hasattr(mock_service, 'optimize_parameters')

    @pytest.mark.asyncio
    async def test_optimize_strategy_interface(self):
        """Test optimize_strategy method interface."""
        mock_service = AsyncMock(spec=OptimizationServiceProtocol)
        
        # Set up return value
        expected_result = {"best_params": {"param1": 0.5}, "score": 0.85}
        mock_service.optimize_strategy.return_value = expected_result
        
        # Test method call
        result = await mock_service.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space=Mock(),
            optimization_method="brute_force",
        )
        
        assert result == expected_result
        mock_service.optimize_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_parameters_interface(self):
        """Test optimize_parameters method interface."""
        mock_service = AsyncMock(spec=OptimizationServiceProtocol)
        
        # Set up return value
        expected_result = Mock(spec=OptimizationResult)
        mock_service.optimize_parameters.return_value = expected_result
        
        # Test method call
        objective_function = lambda x: 0.5
        parameter_space = Mock(spec=ParameterSpace)
        objectives = [Mock(spec=OptimizationObjective)]
        
        result = await mock_service.optimize_parameters(
            objective_function=objective_function,
            parameter_space=parameter_space,
            objectives=objectives,
        )
        
        assert result == expected_result
        mock_service.optimize_parameters.assert_called_once()


class TestBacktestIntegrationProtocol:
    """Test BacktestIntegrationProtocol."""

    def test_protocol_structure(self):
        """Test protocol has required methods."""
        mock_service = Mock(spec=BacktestIntegrationProtocol)
        
        assert hasattr(mock_service, 'evaluate_strategy')

    @pytest.mark.asyncio
    async def test_evaluate_strategy_interface(self):
        """Test evaluate_strategy method interface."""
        mock_service = AsyncMock(spec=BacktestIntegrationProtocol)
        
        # Set up return value
        expected_result = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
        }
        mock_service.evaluate_strategy.return_value = expected_result
        
        # Test method call
        strategy_config = Mock(spec=StrategyConfig)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        initial_capital = Decimal("100000")
        
        result = await mock_service.evaluate_strategy(
            strategy_config=strategy_config,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )
        
        assert result == expected_result
        mock_service.evaluate_strategy.assert_called_once()


class TestOptimizationAnalysisProtocol:
    """Test OptimizationAnalysisProtocol."""

    def test_protocol_structure(self):
        """Test protocol has required methods."""
        mock_service = Mock(spec=OptimizationAnalysisProtocol)
        
        assert hasattr(mock_service, 'analyze_results')
        assert hasattr(mock_service, 'calculate_parameter_importance')

    def test_analyze_results_interface(self):
        """Test analyze_results method interface."""
        mock_service = Mock(spec=OptimizationAnalysisProtocol)
        
        # Set up return value
        expected_result = {
            "parameter_sensitivity": {"param1": 0.8, "param2": 0.4},
            "optimization_quality": 0.95,
        }
        mock_service.analyze_results.return_value = expected_result
        
        # Test method call
        optimization_result = Mock(spec=OptimizationResult)
        parameter_space = Mock(spec=ParameterSpace)
        
        result = mock_service.analyze_results(
            optimization_result=optimization_result,
            parameter_space=parameter_space,
        )
        
        assert result == expected_result
        mock_service.analyze_results.assert_called_once()

    def test_calculate_parameter_importance_interface(self):
        """Test calculate_parameter_importance method interface."""
        mock_service = Mock(spec=OptimizationAnalysisProtocol)
        
        # Set up return value
        expected_result = {"param1": 0.8, "param2": 0.6, "param3": 0.3}
        mock_service.calculate_parameter_importance.return_value = expected_result
        
        # Test method call
        optimization_history = [
            {"param1": 0.1, "param2": 0.2, "score": 0.5},
            {"param1": 0.3, "param2": 0.4, "score": 0.7},
        ]
        parameter_names = ["param1", "param2", "param3"]
        
        result = mock_service.calculate_parameter_importance(
            optimization_history=optimization_history,
            parameter_names=parameter_names,
        )
        
        assert result == expected_result
        mock_service.calculate_parameter_importance.assert_called_once()


class TestOptimizationRepositoryProtocol:
    """Test OptimizationRepositoryProtocol."""

    def test_protocol_structure(self):
        """Test protocol has required methods."""
        mock_repository = Mock(spec=OptimizationRepositoryProtocol)
        
        assert hasattr(mock_repository, 'save_optimization_result')
        assert hasattr(mock_repository, 'get_optimization_result')
        assert hasattr(mock_repository, 'list_optimization_results')

    @pytest.mark.asyncio
    async def test_save_optimization_result_interface(self):
        """Test save_optimization_result method interface."""
        mock_repository = AsyncMock(spec=OptimizationRepositoryProtocol)
        
        # Set up return value
        expected_id = "opt_123456"
        mock_repository.save_optimization_result.return_value = expected_id
        
        # Test method call
        result = Mock(spec=OptimizationResult)
        metadata = {"strategy": "test_strategy", "version": "1.0"}
        
        saved_id = await mock_repository.save_optimization_result(
            result=result,
            metadata=metadata,
        )
        
        assert saved_id == expected_id
        mock_repository.save_optimization_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_optimization_result_interface(self):
        """Test get_optimization_result method interface."""
        mock_repository = AsyncMock(spec=OptimizationRepositoryProtocol)
        
        # Set up return value
        expected_result = Mock(spec=OptimizationResult)
        mock_repository.get_optimization_result.return_value = expected_result
        
        # Test method call
        optimization_id = "opt_123456"
        result = await mock_repository.get_optimization_result(optimization_id)
        
        assert result == expected_result
        mock_repository.get_optimization_result.assert_called_once_with(optimization_id)

    @pytest.mark.asyncio
    async def test_list_optimization_results_interface(self):
        """Test list_optimization_results method interface."""
        mock_repository = AsyncMock(spec=OptimizationRepositoryProtocol)
        
        # Set up return value
        expected_results = [Mock(spec=OptimizationResult), Mock(spec=OptimizationResult)]
        mock_repository.list_optimization_results.return_value = expected_results
        
        # Test method call
        results = await mock_repository.list_optimization_results(
            strategy_name="test_strategy",
            limit=50,
        )
        
        assert results == expected_results
        mock_repository.list_optimization_results.assert_called_once_with(
            strategy_name="test_strategy", limit=50
        )


class TestIOptimizationService:
    """Test IOptimizationService abstract base class."""

    def test_is_abstract_base_class(self):
        """Test that IOptimizationService is an abstract base class."""
        assert issubclass(IOptimizationService, ABC)
        
        # Cannot instantiate abstract base class
        with pytest.raises(TypeError):
            IOptimizationService()

    def test_abstract_methods(self):
        """Test that all methods are abstract."""
        # Create concrete implementation for testing
        class ConcreteOptimizationService(IOptimizationService):
            async def optimize_strategy(self, strategy_name, parameter_space, optimization_method="brute_force", **kwargs):
                return {}
            
            async def optimize_parameters(self, objective_function, parameter_space, objectives, method="brute_force", **kwargs):
                return Mock(spec=OptimizationResult)
            
            async def analyze_optimization_results(self, optimization_result, parameter_space):
                return {}
        
        # Should be able to instantiate concrete implementation
        service = ConcreteOptimizationService()
        assert isinstance(service, IOptimizationService)

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        class IncompleteOptimizationService(IOptimizationService):
            async def optimize_strategy(self, strategy_name, parameter_space, optimization_method="brute_force", **kwargs):
                return {}
            # Missing other abstract methods
        
        # Cannot instantiate incomplete implementation
        with pytest.raises(TypeError):
            IncompleteOptimizationService()


class TestIBacktestIntegrationService:
    """Test IBacktestIntegrationService abstract base class."""

    def test_is_abstract_base_class(self):
        """Test that IBacktestIntegrationService is an abstract base class."""
        assert issubclass(IBacktestIntegrationService, ABC)
        
        # Cannot instantiate abstract base class
        with pytest.raises(TypeError):
            IBacktestIntegrationService()

    def test_concrete_implementation(self):
        """Test concrete implementation works."""
        class ConcreteBacktestService(IBacktestIntegrationService):
            async def evaluate_strategy(self, strategy_config, start_date=None, end_date=None, initial_capital=Decimal("100000")):
                return {"total_return": 0.15}
            
            def create_objective_function(self, strategy_name, data_start_date=None, data_end_date=None, initial_capital=Decimal("100000")):
                return lambda params: 0.5
        
        # Should be able to instantiate concrete implementation
        service = ConcreteBacktestService()
        assert isinstance(service, IBacktestIntegrationService)


class TestInterfaceCompliance:
    """Test interface compliance and integration."""

    def test_protocol_compliance(self):
        """Test that concrete implementations comply with protocols."""
        # Create mock implementations
        optimization_service = Mock()
        optimization_service.optimize_strategy = AsyncMock(return_value={})
        optimization_service.optimize_parameters = AsyncMock(return_value=Mock())
        
        backtest_service = Mock()
        backtest_service.evaluate_strategy = AsyncMock(return_value={})
        
        # Test protocol compliance
        def use_optimization_service(service: OptimizationServiceProtocol):
            return service
        
        def use_backtest_service(service: BacktestIntegrationProtocol):
            return service
        
        # Should accept mocks that implement the protocol
        assert use_optimization_service(optimization_service) is optimization_service
        assert use_backtest_service(backtest_service) is backtest_service

    @pytest.mark.asyncio
    async def test_integration_workflow(self):
        """Test integration between different service protocols."""
        # Create mock services
        optimization_service = AsyncMock(spec=OptimizationServiceProtocol)
        backtest_service = AsyncMock(spec=BacktestIntegrationProtocol)
        analysis_service = Mock(spec=OptimizationAnalysisProtocol)
        repository = AsyncMock(spec=OptimizationRepositoryProtocol)
        
        # Set up mock returns
        optimization_result = Mock(spec=OptimizationResult)
        optimization_service.optimize_parameters.return_value = optimization_result
        
        backtest_service.evaluate_strategy.return_value = {"total_return": 0.15}
        
        analysis_service.analyze_results.return_value = {
            "parameter_sensitivity": {"param1": 0.8}
        }
        
        repository.save_optimization_result.return_value = "opt_123"
        
        # Test workflow
        # 1. Optimize parameters
        result = await optimization_service.optimize_parameters(
            objective_function=lambda x: 0.5,
            parameter_space=Mock(),
            objectives=[Mock()],
        )
        
        # 2. Analyze results
        analysis = analysis_service.analyze_results(result, Mock())
        
        # 3. Save results
        saved_id = await repository.save_optimization_result(result)
        
        # Verify workflow
        assert result == optimization_result
        assert analysis["parameter_sensitivity"]["param1"] == 0.8
        assert saved_id == "opt_123"
        
        # Verify all services were called
        optimization_service.optimize_parameters.assert_called_once()
        analysis_service.analyze_results.assert_called_once()
        repository.save_optimization_result.assert_called_once()


class TestFinancialEdgeCases:
    """Test financial-specific edge cases for interfaces."""

    @pytest.mark.asyncio
    async def test_high_precision_capital_handling(self):
        """Test handling of high-precision capital amounts."""
        mock_service = AsyncMock(spec=BacktestIntegrationProtocol)
        
        # High precision initial capital
        high_precision_capital = Decimal("123456.789012345678901234567890")
        
        mock_service.evaluate_strategy.return_value = {
            "total_return": 0.15,
            "final_capital": float(high_precision_capital * Decimal("1.15"))
        }
        
        result = await mock_service.evaluate_strategy(
            strategy_config=Mock(),
            initial_capital=high_precision_capital,
        )
        
        # Should handle high precision without loss
        assert "final_capital" in result
        mock_service.evaluate_strategy.assert_called_once()

    def test_parameter_importance_edge_cases(self):
        """Test parameter importance calculation with edge cases."""
        mock_service = Mock(spec=OptimizationAnalysisProtocol)
        
        # Empty parameter list
        mock_service.calculate_parameter_importance.return_value = {}
        
        result = mock_service.calculate_parameter_importance(
            optimization_history=[],
            parameter_names=[],
        )
        
        assert result == {}
        
        # Single parameter
        mock_service.calculate_parameter_importance.return_value = {"param1": 1.0}
        
        result = mock_service.calculate_parameter_importance(
            optimization_history=[{"param1": 0.5, "score": 0.8}],
            parameter_names=["param1"],
        )
        
        assert result == {"param1": 1.0}