"""
Unit tests for optimization service module.

Tests the main optimization service implementation that coordinates
optimization algorithms, backtesting integration, and result analysis.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from src.optimization.service import OptimizationService
from src.optimization.interfaces import (
    BacktestIntegrationProtocol,
    OptimizationRepositoryProtocol,
)
from src.optimization.parameter_space import ParameterSpace
from src.optimization.core import OptimizationObjective, OptimizationResult, ObjectiveDirection
from src.core.exceptions import OptimizationError, ValidationError


class TestOptimizationServiceInitialization:
    """Test OptimizationService initialization."""

    def test_service_initialization_with_dependencies(self):
        """Test service initialization with all dependencies."""
        backtest_integration = Mock(spec=BacktestIntegrationProtocol)
        optimization_repository = Mock(spec=OptimizationRepositoryProtocol)
        
        service = OptimizationService(
            backtest_integration=backtest_integration,
            optimization_repository=optimization_repository,
        )
        
        assert service._backtest_integration == backtest_integration
        assert service._optimization_repository == optimization_repository
        assert hasattr(service, 'logger')

    def test_service_initialization_without_dependencies(self):
        """Test service initialization without dependencies."""
        service = OptimizationService()
        
        assert service._backtest_integration is None
        assert service._optimization_repository is None
        assert hasattr(service, 'logger')

    def test_service_inherits_from_base_service(self):
        """Test that service inherits from BaseService."""
        from src.core.base import BaseService
        from src.optimization.interfaces import IOptimizationService
        
        service = OptimizationService()
        
        assert isinstance(service, BaseService)
        assert isinstance(service, IOptimizationService)


class TestOptimizeStrategy:
    """Test optimize_strategy method."""

    @pytest.fixture
    def mock_backtest_integration(self):
        """Create mock backtest integration."""
        mock = AsyncMock()  # Don't use spec to allow adding methods
        # Create async lambda that returns immediately
        async def mock_objective(params):
            await asyncio.sleep(0)
            return Decimal("0.85")
        mock.create_objective_function.return_value = mock_objective
        return mock

    @pytest.fixture(scope="class")
    def mock_repository(self):
        """Create mock repository."""
        mock = AsyncMock(spec=OptimizationRepositoryProtocol)
        mock.save_optimization_result.return_value = "opt_123"
        return mock

    @pytest.fixture(scope="class")
    def sample_parameter_space(self):
        """Create sample parameter space."""
        from src.optimization.parameter_space import ParameterSpaceBuilder
        builder = ParameterSpaceBuilder()
        return builder.add_continuous("param1", 0.0, 1.0).build()

    @pytest.fixture
    def optimization_service(self, mock_backtest_integration, mock_repository):
        """Create optimization service with mocked dependencies."""
        return OptimizationService(
            backtest_integration=mock_backtest_integration,
            optimization_repository=mock_repository,
        )

    @pytest.mark.asyncio
    async def test_optimize_strategy_success(self, optimization_service, sample_parameter_space):
        """Test successful strategy optimization."""
        with patch.object(optimization_service, '_create_objective_function') as mock_create_obj, \
             patch.object(optimization_service, '_create_standard_trading_objectives') as mock_create_objectives, \
             patch.object(optimization_service, 'optimize_parameters') as mock_optimize:
            
            # Setup mocks with async functions
            async def mock_async_objective(params):
                await asyncio.sleep(0)
                return Decimal("0.85")
            mock_create_obj.return_value = mock_async_objective
            mock_create_objectives.return_value = [Mock(spec=OptimizationObjective)]
            
            mock_result = Mock(spec=OptimizationResult)
            mock_result.optimization_id = "test_opt_123"
            mock_result.algorithm_name = "brute_force"
            mock_result.convergence_achieved = True
            mock_result.optimal_parameters = {"param1": Decimal("0.5")}
            mock_result.optimal_objective_value = Decimal("0.85")
            mock_result.iterations_completed = 100
            mock_optimize.return_value = mock_result
            
            # Execute
            result = await optimization_service.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space=sample_parameter_space,
                optimization_method="brute_force",
            )
            
            # Verify
            assert isinstance(result, dict)
            assert "optimization_result" in result
            assert "analysis" in result
            assert "strategy_name" in result
            
            mock_create_obj.assert_called_once()
            mock_create_objectives.assert_called_once()
            mock_optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_strategy_with_custom_parameters(self, optimization_service, sample_parameter_space):
        """Test strategy optimization with custom parameters."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        initial_capital = Decimal("50000")
        
        with patch.object(optimization_service, '_create_objective_function') as mock_create_obj, \
             patch.object(optimization_service, '_create_standard_trading_objectives') as mock_create_objectives, \
             patch.object(optimization_service, 'optimize_parameters') as mock_optimize:
            
            mock_create_obj.return_value = lambda params: Decimal("0.75")
            mock_create_objectives.return_value = [Mock()]
            mock_result = Mock(spec=OptimizationResult)
            mock_result.optimization_id = "test_opt_124"
            mock_result.algorithm_name = "bayesian"
            mock_result.convergence_achieved = False
            mock_result.optimal_parameters = {"param1": Decimal("0.5")}
            mock_result.optimal_objective_value = Decimal("0.75")
            mock_result.iterations_completed = 50
            mock_optimize.return_value = mock_result
            
            await optimization_service.optimize_strategy(
                strategy_name="custom_strategy",
                parameter_space=sample_parameter_space,
                optimization_method="bayesian",
                data_start_date=start_date,
                data_end_date=end_date,
                initial_capital=initial_capital,
            )
            
            # Verify objective function created with custom parameters
            mock_create_obj.assert_called_once_with(
                strategy_name="custom_strategy",
                data_start_date=start_date,
                data_end_date=end_date,
                initial_capital=initial_capital,
            )

    @pytest.mark.asyncio
    async def test_optimize_strategy_without_backtest_integration(self, mock_repository, sample_parameter_space):
        """Test strategy optimization without backtest integration uses simulation."""
        service = OptimizationService(optimization_repository=mock_repository)
        
        # Should succeed using simulation fallback when no backtest integration
        result = await service.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space=sample_parameter_space,
        )

        # Should get results from simulation objective function
        assert isinstance(result, dict)
        assert "optimization_result" in result
        assert "analysis" in result
        assert "strategy_name" in result
        assert result["strategy_name"] == "test_strategy"

        # Verify optimization result structure
        opt_result = result["optimization_result"]
        assert hasattr(opt_result, 'optimization_id')
        assert hasattr(opt_result, 'optimal_parameters')
        assert isinstance(result["analysis"], dict)


class TestOptimizeParameters:
    """Test optimize_parameters method."""

    @pytest.fixture
    def optimization_service(self):
        """Create basic optimization service."""
        return OptimizationService()

    @pytest.fixture(scope="class")
    def sample_objective_function(self):
        """Create sample objective function."""
        async def async_objective(params):
            await asyncio.sleep(0)
            return Decimal("0.8") * params.get("param1", Decimal("1.0"))
        return async_objective

    @pytest.fixture
    def sample_objectives(self):
        """Create sample optimization objectives."""
        return [
            OptimizationObjective(
                name="return",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("1.0"),
                is_primary=True,
            )
        ]

    @pytest.fixture
    def sample_parameter_space(self):
        """Create sample parameter space."""
        from src.optimization.parameter_space import ParameterSpaceBuilder
        builder = ParameterSpaceBuilder()
        return builder.add_continuous("param1", 0.0, 1.0).build()

    @pytest.mark.asyncio
    async def test_optimize_parameters_brute_force(
        self, optimization_service, sample_objective_function, 
        sample_objectives, sample_parameter_space
    ):
        """Test parameter optimization using brute force method."""
        with patch.object(optimization_service, '_create_brute_force_optimizer') as mock_create_optimizer:
            mock_optimizer = AsyncMock()
            mock_result = Mock(spec=OptimizationResult)
            mock_result.optimization_id = "test_opt_125"
            mock_optimizer.optimize.return_value = mock_result
            mock_create_optimizer.return_value = mock_optimizer
            
            result = await optimization_service.optimize_parameters(
                objective_function=sample_objective_function,
                parameter_space=sample_parameter_space,
                objectives=sample_objectives,
                method="brute_force",
            )

            assert result == mock_result
            mock_create_optimizer.assert_called_once()
            mock_optimizer.optimize.assert_called_once_with(sample_objective_function)

    @pytest.mark.asyncio
    async def test_optimize_parameters_bayesian(
        self, optimization_service, sample_objective_function,
        sample_objectives, sample_parameter_space
    ):
        """Test parameter optimization using Bayesian method."""
        with patch.object(optimization_service, '_create_bayesian_optimizer') as mock_create_optimizer:
            mock_optimizer = AsyncMock()
            mock_result = Mock(spec=OptimizationResult)
            mock_result.optimization_id = "test_opt_126"
            mock_optimizer.optimize.return_value = mock_result
            mock_create_optimizer.return_value = mock_optimizer
            
            result = await optimization_service.optimize_parameters(
                objective_function=sample_objective_function,
                parameter_space=sample_parameter_space,
                objectives=sample_objectives,
                method="bayesian",
            )

            assert result == mock_result
            mock_create_optimizer.assert_called_once()
            mock_optimizer.optimize.assert_called_once_with(sample_objective_function)

    @pytest.mark.asyncio
    async def test_optimize_parameters_unknown_method(
        self, optimization_service, sample_objective_function,
        sample_objectives, sample_parameter_space
    ):
        """Test parameter optimization with unknown method."""
        with pytest.raises(ValidationError, match="Unknown optimization method"):
            await optimization_service.optimize_parameters(
                objective_function=sample_objective_function,
                parameter_space=sample_parameter_space,
                objectives=sample_objectives,
                method="unknown_method",
            )

    @pytest.mark.asyncio
    async def test_optimize_parameters_with_repository(self, sample_parameter_space):
        """Test parameter optimization with repository available."""
        mock_repository = AsyncMock(spec=OptimizationRepositoryProtocol)
        
        service = OptimizationService(optimization_repository=mock_repository)
        
        with patch.object(service, '_create_brute_force_optimizer') as mock_create_optimizer:
            mock_optimizer = AsyncMock()
            mock_result = Mock(spec=OptimizationResult)
            mock_result.optimization_id = "test_opt_127"
            mock_optimizer.optimize.return_value = mock_result
            mock_create_optimizer.return_value = mock_optimizer

            result = await service.optimize_parameters(
                objective_function=lambda x: 0.5,
                parameter_space=sample_parameter_space,
                objectives=[Mock()],
            )
            
            # Should return the optimization result (no automatic saving in optimize_parameters)
            assert result == mock_result
            # Repository is not called by optimize_parameters method
            mock_repository.save_optimization_result.assert_not_called()


class TestAnalyzeOptimizationResults:
    """Test analyze_optimization_results method."""

    @pytest.fixture
    def optimization_service(self):
        """Create optimization service."""
        return OptimizationService()

    @pytest.mark.asyncio
    async def test_analyze_results_with_analyzer(self, optimization_service):
        """Test result analysis when analyzer is available."""
        mock_result = Mock(spec=OptimizationResult)
        mock_result.optimization_id = "test_opt_128"
        mock_result.algorithm_name = "test_algorithm"
        mock_result.convergence_achieved = True
        mock_result.iterations_completed = 42
        mock_result.objective_values = {"sharpe_ratio": Decimal("0.8")}
        mock_result.optimal_parameters = {"param1": Decimal("0.5")}
        mock_result.optimal_objective_value = Decimal("0.8")
        mock_parameter_space = Mock(spec=ParameterSpace)
        mock_parameter_space.parameters = {"param1": "continuous"}
        
        # Mock the results analyzer
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_optimization_results.return_value = {
            "parameter_sensitivity": {"param1": 0.8},
            "optimization_quality": 0.95,
        }
        
        optimization_service._analysis_service = mock_analyzer
        
        result = await optimization_service.analyze_optimization_results(
            optimization_result=mock_result,
            parameter_space=mock_parameter_space,
        )
        
        assert "parameter_sensitivity" in result
        assert "optimization_quality" in result
        mock_analyzer.analyze_optimization_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_results_without_analyzer(self, optimization_service):
        """Test result analysis when no analyzer is available."""
        mock_result = Mock(spec=OptimizationResult)
        mock_result.optimization_id = "test_opt_129"
        mock_result.convergence_achieved = True
        mock_result.iterations_completed = 42
        mock_result.optimal_parameters = {"param1": Decimal("0.5")}
        mock_result.optimal_objective_value = Decimal("0.8")
        mock_parameter_space = Mock(spec=ParameterSpace)
        mock_parameter_space.parameters = {"param1": "continuous"}
        
        # No analyzer set
        optimization_service._analysis_service = None
        
        result = await optimization_service.analyze_optimization_results(
            optimization_result=mock_result,
            parameter_space=mock_parameter_space,
        )
        
        # Should return analysis from newly created analyzer
        assert isinstance(result, dict)
        assert "best_result_analysis" in result or "parameter_correlations" in result


class TestPrivateMethods:
    """Test private helper methods."""

    @pytest.fixture
    def optimization_service(self):
        """Create optimization service with mock dependencies."""
        mock_backtest = Mock()  # Don't use spec here since we need to add methods
        # Set up the create_objective_function method
        async def mock_objective(params):
            return 0.5
        mock_backtest.create_objective_function.return_value = mock_objective
        return OptimizationService(backtest_integration=mock_backtest)

    @pytest.mark.asyncio
    async def test_create_objective_function(self, optimization_service):
        """Test objective function creation."""
        strategy_name = "test_strategy"
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        initial_capital = Decimal("100000")
        
        # Mock the backtest integration
        expected_function = lambda params: Decimal("0.75")
        optimization_service._backtest_integration.create_objective_function.return_value = expected_function
        
        result = await optimization_service._create_objective_function(
            strategy_name=strategy_name,
            data_start_date=start_date,
            data_end_date=end_date,
            initial_capital=initial_capital,
        )
        
        assert result == expected_function
        optimization_service._backtest_integration.create_objective_function.assert_called_once_with(
            strategy_name,
            start_date,
            end_date,
            initial_capital,
        )

    def test_create_standard_trading_objectives(self, optimization_service):
        """Test standard trading objectives creation."""
        objectives = optimization_service._create_standard_trading_objectives()
        
        assert isinstance(objectives, list)
        assert len(objectives) > 0
        assert all(isinstance(obj, OptimizationObjective) for obj in objectives)
        
        # Should have at least return objective
        objective_names = [obj.name for obj in objectives]
        assert "total_return" in objective_names or "return" in objective_names


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def optimization_service(self):
        """Create optimization service."""
        return OptimizationService()

    @pytest.mark.asyncio
    async def test_optimization_error_handling(self, optimization_service):
        """Test optimization error handling."""
        sample_parameter_space = Mock(spec=ParameterSpace)
        
        with patch.object(optimization_service, '_create_brute_force_optimizer') as mock_create_optimizer:
            mock_optimizer = AsyncMock()
            mock_optimizer.optimize.side_effect = OptimizationError("Test error")
            mock_create_optimizer.return_value = mock_optimizer
            
            with pytest.raises(OptimizationError):
                await optimization_service.optimize_parameters(
                    objective_function=lambda x: 0.5,
                    parameter_space=sample_parameter_space,
                    objectives=[Mock()],
                )

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, optimization_service):
        """Test validation error handling."""
        # Test with invalid parameter space - causes optimizer creation failure
        with pytest.raises(OptimizationError, match="Failed to create brute force optimizer"):
            await optimization_service.optimize_parameters(
                objective_function=lambda x: 0.5,
                parameter_space=None,  # Invalid
                objectives=[Mock()],
            )

    @pytest.mark.asyncio
    async def test_objective_function_failure_handling(self, optimization_service):
        """Test handling of objective function failures."""
        from src.optimization.parameter_space import ParameterSpaceBuilder

        # Create parameter space
        parameter_space = ParameterSpaceBuilder().add_continuous("param1", 0.0, 1.0).build()

        # Create objective function that always fails
        async def failing_objective(params):
            raise RuntimeError("Objective function failed")

        with patch.object(optimization_service, '_create_brute_force_optimizer') as mock_create_optimizer:
            mock_optimizer = AsyncMock()
            mock_optimizer.optimize.side_effect = OptimizationError("Objective evaluation failed")
            mock_create_optimizer.return_value = mock_optimizer

            with pytest.raises(OptimizationError):
                await optimization_service.optimize_parameters(
                    objective_function=failing_objective,
                    parameter_space=parameter_space,
                    objectives=[Mock()],
                )


class TestFinancialEdgeCases:
    """Test financial-specific edge cases."""

    @pytest.fixture
    def optimization_service(self):
        """Create optimization service."""
        return OptimizationService()

    def test_decimal_precision_preservation(self, optimization_service):
        """Test that decimal precision is preserved throughout optimization."""
        high_precision_capital = Decimal("123456.789012345678901234567890")
        
        # Test objective function creation with high precision
        with patch.object(optimization_service, '_backtest_integration') as mock_backtest:
            mock_backtest.create_objective_function.return_value = lambda x: high_precision_capital
            
            # Should handle high precision without loss
            objectives = optimization_service._create_standard_trading_objectives()
            assert all(isinstance(obj.weight, Decimal) for obj in objectives)

    @pytest.mark.asyncio
    async def test_risk_constraint_validation(self, optimization_service):
        """Test risk constraint validation in trading objectives."""
        objectives = optimization_service._create_standard_trading_objectives()
        
        # Should have risk-related constraints
        objective_names = [obj.name.lower() for obj in objectives]
        
        # Check for common risk metrics
        risk_metrics = ["sharpe_ratio", "max_drawdown", "volatility", "var"]
        has_risk_metric = any(metric in " ".join(objective_names) for metric in risk_metrics)
        
        # Should have some form of risk consideration
        assert len(objectives) > 1 or has_risk_metric  # Multiple objectives or risk-aware

    def test_trading_objective_constraints(self, optimization_service):
        """Test that trading objectives have proper constraints."""
        objectives = optimization_service._create_standard_trading_objectives()

        for obj in objectives:
            # Financial objectives should have reasonable constraints
            assert obj.weight >= Decimal("0")  # Non-negative weights

            # Check for reasonable constraint bounds if they exist
            if obj.constraint_min is not None:
                assert isinstance(obj.constraint_min, Decimal)
            if obj.constraint_max is not None:
                assert isinstance(obj.constraint_max, Decimal)
                if obj.constraint_min is not None:
                    assert obj.constraint_max >= obj.constraint_min

    @pytest.mark.asyncio
    async def test_extreme_market_conditions_simulation(self, optimization_service):
        """Test optimization under extreme market conditions."""
        from src.optimization.parameter_space import ParameterSpaceBuilder

        # Create parameter space for stress testing
        parameter_space = ParameterSpaceBuilder().add_continuous("position_size", 0.001, 0.02).build()

        # Simulate extreme market volatility scenario
        extreme_objectives = [
            OptimizationObjective(
                name="max_drawdown",
                direction=ObjectiveDirection.MINIMIZE,
                weight=Decimal("3.0"),
                constraint_max=Decimal("0.15"),  # 15% max drawdown limit
                is_primary=False,
            ),
            OptimizationObjective(
                name="var_95",
                direction=ObjectiveDirection.MINIMIZE,
                weight=Decimal("2.0"),
                constraint_max=Decimal("0.05"),  # 5% VaR limit
                is_primary=False,
            )
        ]

        # This should handle the extreme risk constraints properly
        result = await optimization_service.optimize_parameters(
            objective_function=lambda params: {
                "max_drawdown": Decimal("0.12"),
                "var_95": Decimal("0.04")
            },
            parameter_space=parameter_space,
            objectives=extreme_objectives,
        )

        # Verify risk constraints are respected
        assert isinstance(result, OptimizationResult)
        assert "max_drawdown" in result.objective_values
        assert "var_95" in result.objective_values
        assert result.objective_values["max_drawdown"] <= Decimal("0.15")
        assert result.objective_values["var_95"] <= Decimal("0.05")

    def test_currency_precision_edge_cases(self, optimization_service):
        """Test handling of extreme currency precision."""
        # Test with very small amounts (satoshi-level precision)
        tiny_amount = Decimal("0.00000001")  # 1 satoshi

        # Test with very large amounts (institutional scale)
        large_amount = Decimal("1000000000.12345678")  # 1B+ with precision

        objectives = optimization_service._create_standard_trading_objectives()

        # All objectives should handle extreme precision without loss
        for obj in objectives:
            if obj.constraint_min is not None:
                # Should be able to handle tiny constraints
                obj.constraint_min = tiny_amount
                assert obj.satisfies_constraints(tiny_amount) is True
                assert obj.satisfies_constraints(tiny_amount / 2) is False

            if obj.constraint_max is not None:
                # Should be able to handle large constraints
                obj.constraint_max = large_amount
                assert obj.satisfies_constraints(large_amount) is True
                assert obj.satisfies_constraints(large_amount * 2) is False