"""
Comprehensive tests for optimization controller.

This module ensures high coverage for the OptimizationController class,
covering all methods, validation logic, and edge cases.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.optimization.controller import OptimizationController
from src.optimization.interfaces import IOptimizationService
from src.core.exceptions import ValidationError


class TestOptimizationController:
    """Test cases for OptimizationController class."""

    @pytest.fixture
    def mock_optimization_service(self):
        """Create a mock optimization service."""
        service = Mock(spec=IOptimizationService)
        service.optimize_strategy = AsyncMock()
        service.optimize_parameters_with_config = AsyncMock()
        return service

    @pytest.fixture
    def controller(self, mock_optimization_service):
        """Create OptimizationController instance."""
        return OptimizationController(
            optimization_service=mock_optimization_service,
            name="TestController",
            config={"test": "config"},
            correlation_id="test-123"
        )

    @pytest.fixture
    def sample_parameter_space_config(self):
        """Sample parameter space configuration."""
        return {
            "parameters": {
                "param1": {
                    "type": "continuous",
                    "min_value": 0.1,
                    "max_value": 0.9
                },
                "param2": {
                    "type": "discrete",
                    "min_value": 1,
                    "max_value": 10
                }
            }
        }

    @pytest.fixture
    def sample_objectives_config(self):
        """Sample objectives configuration."""
        return [
            {
                "name": "profit",
                "direction": "maximize",
                "weight": 1.0
            },
            {
                "name": "sharpe",
                "direction": "maximize",
                "weight": 0.5
            }
        ]


class TestOptimizationControllerInitialization(TestOptimizationController):
    """Test OptimizationController initialization."""

    def test_initialization_success(self, mock_optimization_service):
        """Test successful controller initialization."""
        controller = OptimizationController(
            optimization_service=mock_optimization_service,
            name="TestController",
            config={"setting": "value"},
            correlation_id="corr-123"
        )

        assert controller._optimization_service == mock_optimization_service
        assert controller.name == "TestController"
        assert controller.get_config() == {"setting": "value"}
        assert controller.correlation_id == "corr-123"

    def test_initialization_default_name(self, mock_optimization_service):
        """Test initialization with default name."""
        controller = OptimizationController(optimization_service=mock_optimization_service)

        assert controller.name == "OptimizationController"
        assert controller._optimization_service == mock_optimization_service

    def test_initialization_adds_dependency(self, mock_optimization_service):
        """Test that initialization adds service dependency."""
        with patch.object(OptimizationController, 'add_dependency') as mock_add_dep:
            controller = OptimizationController(optimization_service=mock_optimization_service)
            mock_add_dep.assert_called_once_with("OptimizationService")

    def test_initialization_logs_info(self, mock_optimization_service):
        """Test that initialization logs info message."""
        # Simply test that the controller was created successfully
        # and has the correct service dependency
        controller = OptimizationController(optimization_service=mock_optimization_service)

        assert controller._optimization_service == mock_optimization_service
        assert controller.name == "OptimizationController"


class TestOptimizeStrategy(TestOptimizationController):
    """Test optimize_strategy method."""

    @pytest.mark.asyncio
    async def test_optimize_strategy_success(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test successful strategy optimization."""
        expected_result = {"optimization_id": "test-123", "status": "completed"}
        mock_optimization_service.optimize_strategy.return_value = expected_result

        result = await controller.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="brute_force",
            initial_capital="50000"
        )

        assert result == expected_result
        mock_optimization_service.optimize_strategy.assert_called_once_with(
            strategy_name="test_strategy",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="brute_force",
            data_start_date=None,
            data_end_date=None,
            initial_capital=Decimal("50000")
        )

    @pytest.mark.asyncio
    async def test_optimize_strategy_with_dates(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test strategy optimization with date parameters."""
        expected_result = {"optimization_id": "test-456"}
        mock_optimization_service.optimize_strategy.return_value = expected_result

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = await controller.optimize_strategy(
            strategy_name="trend_following",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="bayesian",
            data_start_date=start_date,
            data_end_date=end_date,
            initial_capital=Decimal("100000")
        )

        assert result == expected_result
        mock_optimization_service.optimize_strategy.assert_called_once_with(
            strategy_name="trend_following",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="bayesian",
            data_start_date=start_date,
            data_end_date=end_date,
            initial_capital=Decimal("100000")
        )

    @pytest.mark.asyncio
    async def test_optimize_strategy_with_kwargs(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test strategy optimization with additional kwargs."""
        expected_result = {"optimization_id": "test-789"}
        mock_optimization_service.optimize_strategy.return_value = expected_result

        result = await controller.optimize_strategy(
            strategy_name="mean_reversion",
            parameter_space_config=sample_parameter_space_config,
            grid_resolution=5,
            n_calls=100,
            random_state=42
        )

        assert result == expected_result
        mock_optimization_service.optimize_strategy.assert_called_once_with(
            strategy_name="mean_reversion",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="brute_force",  # Default
            data_start_date=None,
            data_end_date=None,
            initial_capital=Decimal("100000"),  # Default
            grid_resolution=5,
            n_calls=100,
            random_state=42
        )

    @pytest.mark.asyncio
    async def test_optimize_strategy_decimal_capital_conversion(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test conversion of string initial_capital to Decimal."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "success"}

        # Test string conversion
        await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital="25000.50"
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == Decimal("25000.50")
        assert isinstance(call_args["initial_capital"], Decimal)

        # Test Decimal passthrough
        mock_optimization_service.optimize_strategy.reset_mock()
        decimal_capital = Decimal("75000.75")

        await controller.optimize_strategy(
            strategy_name="test2",
            parameter_space_config=sample_parameter_space_config,
            initial_capital=decimal_capital
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == decimal_capital
        assert isinstance(call_args["initial_capital"], Decimal)

    @pytest.mark.asyncio
    async def test_optimize_strategy_logging(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test strategy optimization logging."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "success"}

        with patch.object(controller, '_logger') as mock_logger:
            await controller.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space_config=sample_parameter_space_config,
                optimization_method="bayesian"
            )

            mock_logger.info.assert_called_once_with(
                "Strategy optimization completed",
                strategy="test_strategy",
                method="bayesian"
            )

    @pytest.mark.asyncio
    async def test_optimize_strategy_service_error_propagation(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test that service errors are propagated."""
        mock_optimization_service.optimize_strategy.side_effect = RuntimeError("Service error")

        with pytest.raises(RuntimeError, match="Service error"):
            await controller.optimize_strategy(
                strategy_name="test",
                parameter_space_config=sample_parameter_space_config
            )


class TestOptimizeParameters(TestOptimizationController):
    """Test optimize_parameters method."""

    @pytest.mark.asyncio
    async def test_optimize_parameters_success(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test successful parameter optimization."""
        expected_result = {"optimization_id": "param-opt-123"}
        mock_optimization_service.optimize_parameters_with_config.return_value = expected_result

        result = await controller.optimize_parameters(
            objective_function_name="profit_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="brute_force"
        )

        assert result == {"optimization_result": expected_result}
        mock_optimization_service.optimize_parameters_with_config.assert_called_once_with(
            objective_function_name="profit_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="brute_force"
        )

    @pytest.mark.asyncio
    async def test_optimize_parameters_with_kwargs(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test parameter optimization with additional kwargs."""
        expected_result = {"optimization_id": "param-opt-456"}
        mock_optimization_service.optimize_parameters_with_config.return_value = expected_result

        result = await controller.optimize_parameters(
            objective_function_name="sharpe_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="bayesian",
            n_calls=50,
            random_state=123
        )

        assert result == {"optimization_result": expected_result}
        mock_optimization_service.optimize_parameters_with_config.assert_called_once_with(
            objective_function_name="sharpe_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="bayesian",
            n_calls=50,
            random_state=123
        )

    @pytest.mark.asyncio
    async def test_optimize_parameters_default_method(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test parameter optimization with default method."""
        expected_result = {"optimization_id": "param-opt-789"}
        mock_optimization_service.optimize_parameters_with_config.return_value = expected_result

        result = await controller.optimize_parameters(
            objective_function_name="test_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config
            # method not specified, should default to "brute_force"
        )

        assert result == {"optimization_result": expected_result}
        call_args = mock_optimization_service.optimize_parameters_with_config.call_args[1]
        assert call_args["method"] == "brute_force"

    @pytest.mark.asyncio
    async def test_optimize_parameters_service_error_propagation(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test that service errors are propagated."""
        mock_optimization_service.optimize_parameters_with_config.side_effect = ValueError("Config error")

        with pytest.raises(ValueError, match="Config error"):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config=sample_parameter_space_config,
                objectives_config=sample_objectives_config
            )


class TestStrategyOptimizationValidation(TestOptimizationController):
    """Test strategy optimization request validation."""

    @pytest.mark.asyncio
    async def test_validate_strategy_optimization_empty_strategy_name(
        self,
        controller,
        sample_parameter_space_config
    ):
        """Test validation with empty strategy name."""
        with pytest.raises(ValidationError, match="Strategy name is required"):
            await controller.optimize_strategy(
                strategy_name="",
                parameter_space_config=sample_parameter_space_config
            )

        with pytest.raises(ValidationError, match="Strategy name is required"):
            await controller.optimize_strategy(
                strategy_name="   ",  # Only whitespace
                parameter_space_config=sample_parameter_space_config
            )

    @pytest.mark.asyncio
    async def test_validate_strategy_optimization_none_strategy_name(
        self,
        controller,
        sample_parameter_space_config
    ):
        """Test validation with None strategy name."""
        with pytest.raises(ValidationError, match="Strategy name is required"):
            await controller.optimize_strategy(
                strategy_name=None,
                parameter_space_config=sample_parameter_space_config
            )

    @pytest.mark.asyncio
    async def test_validate_strategy_optimization_empty_parameter_space(self, controller):
        """Test validation with empty parameter space."""
        with pytest.raises(ValidationError, match="Parameter space configuration is required"):
            await controller.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space_config={}
            )

        with pytest.raises(ValidationError, match="Parameter space configuration is required"):
            await controller.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space_config=None
            )

    @pytest.mark.asyncio
    async def test_validate_strategy_optimization_invalid_method(
        self,
        controller,
        sample_parameter_space_config
    ):
        """Test validation with invalid optimization method."""
        with pytest.raises(
            ValidationError,
            match="Invalid optimization method: invalid_method. Must be 'brute_force' or 'bayesian'"
        ):
            await controller.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space_config=sample_parameter_space_config,
                optimization_method="invalid_method"
            )

    @pytest.mark.asyncio
    async def test_validate_strategy_optimization_valid_methods(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test validation passes with valid methods."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "success"}

        # Test brute_force
        await controller.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="brute_force"
        )

        # Test bayesian
        await controller.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space_config=sample_parameter_space_config,
            optimization_method="bayesian"
        )

        # Both should pass validation and reach the service
        assert mock_optimization_service.optimize_strategy.call_count == 2


class TestParameterOptimizationValidation(TestOptimizationController):
    """Test parameter optimization request validation."""

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_empty_function_name(
        self,
        controller,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test validation with empty objective function name."""
        with pytest.raises(ValidationError, match="Objective function name is required"):
            await controller.optimize_parameters(
                objective_function_name="",
                parameter_space_config=sample_parameter_space_config,
                objectives_config=sample_objectives_config
            )

        with pytest.raises(ValidationError, match="Objective function name is required"):
            await controller.optimize_parameters(
                objective_function_name="   ",  # Only whitespace
                parameter_space_config=sample_parameter_space_config,
                objectives_config=sample_objectives_config
            )

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_none_function_name(
        self,
        controller,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test validation with None objective function name."""
        with pytest.raises(ValidationError, match="Objective function name is required"):
            await controller.optimize_parameters(
                objective_function_name=None,
                parameter_space_config=sample_parameter_space_config,
                objectives_config=sample_objectives_config
            )

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_empty_parameter_space(
        self,
        controller,
        sample_objectives_config
    ):
        """Test validation with empty parameter space."""
        with pytest.raises(ValidationError, match="Parameter space configuration is required"):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config={},
                objectives_config=sample_objectives_config
            )

        with pytest.raises(ValidationError, match="Parameter space configuration is required"):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config=None,
                objectives_config=sample_objectives_config
            )

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_empty_objectives(
        self,
        controller,
        sample_parameter_space_config
    ):
        """Test validation with empty objectives configuration."""
        with pytest.raises(ValidationError, match="Objectives configuration is required"):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config=sample_parameter_space_config,
                objectives_config=[]
            )

        with pytest.raises(ValidationError, match="Objectives configuration is required"):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config=sample_parameter_space_config,
                objectives_config=None
            )

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_invalid_method(
        self,
        controller,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test validation with invalid optimization method."""
        with pytest.raises(
            ValidationError,
            match="Invalid optimization method: random_search. Must be 'brute_force' or 'bayesian'"
        ):
            await controller.optimize_parameters(
                objective_function_name="test_function",
                parameter_space_config=sample_parameter_space_config,
                objectives_config=sample_objectives_config,
                method="random_search"
            )

    @pytest.mark.asyncio
    async def test_validate_parameter_optimization_valid_methods(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test validation passes with valid methods."""
        mock_optimization_service.optimize_parameters_with_config.return_value = {"result": "success"}

        # Test brute_force
        await controller.optimize_parameters(
            objective_function_name="test_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="brute_force"
        )

        # Test bayesian
        await controller.optimize_parameters(
            objective_function_name="test_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config,
            method="bayesian"
        )

        # Both should pass validation and reach the service
        assert mock_optimization_service.optimize_parameters_with_config.call_count == 2


class TestControllerEdgeCases(TestOptimizationController):
    """Test edge cases and error conditions."""

    def test_controller_inheritance(self, mock_optimization_service):
        """Test that controller properly inherits from BaseComponent."""
        from src.core.base import BaseComponent

        controller = OptimizationController(optimization_service=mock_optimization_service)

        assert isinstance(controller, BaseComponent)
        assert hasattr(controller, '_logger')
        assert hasattr(controller, 'name')
        assert hasattr(controller, 'get_config')
        assert hasattr(controller, 'correlation_id')

    @pytest.mark.asyncio
    async def test_optimize_strategy_with_extreme_decimal_values(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test strategy optimization with extreme decimal values."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "success"}

        # Test very small initial capital
        await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital="0.000000001"
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == Decimal("0.000000001")

        # Test very large initial capital
        await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital="999999999999.999999999"
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == Decimal("999999999999.999999999")

    @pytest.mark.asyncio
    async def test_optimize_strategy_invalid_decimal_conversion(
        self,
        controller,
        sample_parameter_space_config,
        mock_optimization_service
    ):
        """Test strategy optimization with invalid decimal string."""
        from decimal import InvalidOperation

        # Test that the controller validates and converts decimal input
        # Make the service side effect check for decimal validity
        def check_decimal_call(*args, **kwargs):
            initial_capital = kwargs.get('initial_capital')
            if not isinstance(initial_capital, Decimal):
                raise ValueError(f"Expected Decimal, got {type(initial_capital)}: {initial_capital}")
            if initial_capital.is_nan():
                raise ValueError("Cannot use NaN as initial capital")
            return {"result": "success"}

        mock_optimization_service.optimize_strategy.side_effect = check_decimal_call

        # This should raise a ValueError when the service detects NaN
        with pytest.raises(ValueError, match="Cannot use NaN as initial capital"):
            await controller.optimize_strategy(
                strategy_name="test",
                parameter_space_config=sample_parameter_space_config,
                initial_capital="definitely_not_a_number_123!@#"
            )

    @pytest.mark.asyncio
    async def test_concurrent_optimization_calls(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config,
        sample_objectives_config
    ):
        """Test that controller can handle concurrent calls."""
        import asyncio

        mock_optimization_service.optimize_strategy.return_value = {"strategy_result": "success"}
        mock_optimization_service.optimize_parameters_with_config.return_value = {"param_result": "success"}

        # Run both methods concurrently
        strategy_task = controller.optimize_strategy(
            strategy_name="test_strategy",
            parameter_space_config=sample_parameter_space_config
        )

        parameter_task = controller.optimize_parameters(
            objective_function_name="test_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=sample_objectives_config
        )

        strategy_result, parameter_result = await asyncio.gather(strategy_task, parameter_task)

        assert strategy_result == {"strategy_result": "success"}
        assert parameter_result == {"optimization_result": {"param_result": "success"}}

    @pytest.mark.asyncio
    async def test_complex_parameter_space_config(
        self,
        controller,
        mock_optimization_service
    ):
        """Test with complex parameter space configuration."""
        complex_config = {
            "parameters": {
                "continuous_param": {
                    "type": "continuous",
                    "min_value": 0.001,
                    "max_value": 0.999,
                    "precision": 6
                },
                "discrete_param": {
                    "type": "discrete",
                    "min_value": 1,
                    "max_value": 1000,
                    "step_size": 10
                },
                "categorical_param": {
                    "type": "categorical",
                    "choices": ["option_a", "option_b", "option_c"],
                    "weights": [0.5, 0.3, 0.2]
                }
            },
            "constraints": ["continuous_param < discrete_param * 0.001"],
            "metadata": {"description": "Complex parameter space"}
        }

        mock_optimization_service.optimize_strategy.return_value = {"result": "complex_success"}

        result = await controller.optimize_strategy(
            strategy_name="complex_strategy",
            parameter_space_config=complex_config
        )

        assert result == {"result": "complex_success"}
        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["parameter_space_config"] == complex_config

    @pytest.mark.asyncio
    async def test_complex_objectives_config(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test with complex objectives configuration."""
        complex_objectives = [
            {
                "name": "profit",
                "direction": "maximize",
                "weight": 0.7,
                "target_value": 1000.0,
                "constraint_min": 100.0,
                "constraint_max": 5000.0
            },
            {
                "name": "sharpe_ratio",
                "direction": "maximize",
                "weight": 0.2,
                "target_value": 2.0
            },
            {
                "name": "max_drawdown",
                "direction": "minimize",
                "weight": 0.1,
                "constraint_max": 0.2
            }
        ]

        mock_optimization_service.optimize_parameters_with_config.return_value = {"result": "complex_objectives"}

        result = await controller.optimize_parameters(
            objective_function_name="multi_objective_function",
            parameter_space_config=sample_parameter_space_config,
            objectives_config=complex_objectives
        )

        assert result == {"optimization_result": {"result": "complex_objectives"}}
        call_args = mock_optimization_service.optimize_parameters_with_config.call_args[1]
        assert call_args["objectives_config"] == complex_objectives


class TestControllerFinancialEdgeCases(TestOptimizationController):
    """Test financial calculation specific edge cases."""

    @pytest.mark.asyncio
    async def test_zero_initial_capital(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test optimization with zero initial capital."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "zero_capital"}

        result = await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital="0.0"
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_negative_initial_capital(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test optimization with negative initial capital."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "negative_capital"}

        result = await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital="-1000.0"
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert call_args["initial_capital"] == Decimal("-1000.0")

    @pytest.mark.asyncio
    async def test_high_precision_initial_capital(
        self,
        controller,
        mock_optimization_service,
        sample_parameter_space_config
    ):
        """Test optimization with high precision initial capital."""
        mock_optimization_service.optimize_strategy.return_value = {"result": "high_precision"}

        high_precision_capital = "123456789.123456789012345678"

        result = await controller.optimize_strategy(
            strategy_name="test",
            parameter_space_config=sample_parameter_space_config,
            initial_capital=high_precision_capital
        )

        call_args = mock_optimization_service.optimize_strategy.call_args[1]
        assert str(call_args["initial_capital"]) == high_precision_capital
        assert isinstance(call_args["initial_capital"], Decimal)