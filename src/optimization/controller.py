"""
Optimization controller for handling optimization requests.

This module provides the controller layer for optimization operations,
following the Controller→Service→Repository pattern.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ValidationError
from src.optimization.interfaces import IOptimizationService
from src.optimization.parameter_space import ParameterSpace


class OptimizationController(BaseComponent):
    """
    Controller for optimization operations.

    Handles HTTP requests, validation, and orchestrates optimization
    operations through the service layer.
    """

    def __init__(self, optimization_service: IOptimizationService):
        """
        Initialize optimization controller.

        Args:
            optimization_service: Optimization service instance
        """
        super().__init__()
        self._optimization_service = optimization_service

        self.logger.info("OptimizationController initialized")

    async def optimize_strategy(
        self,
        strategy_name: str,
        parameter_space_config: dict[str, Any],
        optimization_method: str = "brute_force",
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: str | Decimal = "100000",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Handle strategy optimization request.

        Args:
            strategy_name: Name of strategy to optimize
            parameter_space_config: Parameter space configuration
            optimization_method: Method to use for optimization
            data_start_date: Start date for backtesting data
            data_end_date: End date for backtesting data
            initial_capital: Initial capital amount
            **kwargs: Additional optimization parameters

        Returns:
            Optimization results
        """
        # Validate inputs
        self._validate_strategy_optimization_request(
            strategy_name, parameter_space_config, optimization_method
        )

        # Convert string capital to Decimal if needed
        if isinstance(initial_capital, str):
            initial_capital = Decimal(initial_capital)

        # Build parameter space from configuration
        parameter_space = self._build_parameter_space(parameter_space_config)

        # Delegate to service layer
        result = await self._optimization_service.optimize_strategy(
            strategy_name=strategy_name,
            parameter_space=parameter_space,
            optimization_method=optimization_method,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            initial_capital=initial_capital,
            **kwargs,
        )

        self.logger.info(
            "Strategy optimization completed",
            strategy=strategy_name,
            method=optimization_method,
        )

        return result

    async def optimize_parameters(
        self,
        objective_function_name: str,
        parameter_space_config: dict[str, Any],
        objectives_config: list[dict[str, Any]],
        method: str = "brute_force",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Handle parameter optimization request.

        Args:
            objective_function_name: Name of objective function to use
            parameter_space_config: Parameter space configuration
            objectives_config: Optimization objectives configuration
            method: Optimization method to use
            **kwargs: Additional optimization parameters

        Returns:
            Optimization results
        """
        # Validate inputs
        self._validate_parameter_optimization_request(
            objective_function_name, parameter_space_config, objectives_config, method
        )

        # Build parameter space
        parameter_space = self._build_parameter_space(parameter_space_config)

        # Build objectives
        objectives = self._build_objectives(objectives_config)

        # Create objective function (placeholder - would be implemented based on needs)
        objective_function = self._create_objective_function(objective_function_name)

        # Delegate to service layer
        result = await self._optimization_service.optimize_parameters(
            objective_function=objective_function,
            parameter_space=parameter_space,
            objectives=objectives,
            method=method,
            **kwargs,
        )

        return {"optimization_result": result}

    def _validate_strategy_optimization_request(
        self,
        strategy_name: str,
        parameter_space_config: dict[str, Any],
        optimization_method: str,
    ) -> None:
        """Validate strategy optimization request."""
        if not strategy_name or not strategy_name.strip():
            raise ValidationError("Strategy name is required")

        if not parameter_space_config:
            raise ValidationError("Parameter space configuration is required")

        if optimization_method not in ["brute_force", "bayesian"]:
            raise ValidationError(
                f"Invalid optimization method: {optimization_method}. "
                "Must be 'brute_force' or 'bayesian'"
            )

    def _validate_parameter_optimization_request(
        self,
        objective_function_name: str,
        parameter_space_config: dict[str, Any],
        objectives_config: list[dict[str, Any]],
        method: str,
    ) -> None:
        """Validate parameter optimization request."""
        if not objective_function_name or not objective_function_name.strip():
            raise ValidationError("Objective function name is required")

        if not parameter_space_config:
            raise ValidationError("Parameter space configuration is required")

        if not objectives_config:
            raise ValidationError("Objectives configuration is required")

        if method not in ["brute_force", "bayesian"]:
            raise ValidationError(
                f"Invalid optimization method: {method}. " "Must be 'brute_force' or 'bayesian'"
            )

    def _build_parameter_space(self, config: dict[str, Any]) -> ParameterSpace:
        """Build parameter space from configuration."""
        from src.optimization.parameter_space import ParameterSpaceBuilder

        builder = ParameterSpaceBuilder()

        for param_name, param_config in config.items():
            param_type = param_config.get("type")

            if param_type == "continuous":
                builder.add_continuous(
                    name=param_name,
                    min_value=param_config["min_value"],
                    max_value=param_config["max_value"],
                    precision=param_config.get("precision", 3),
                )
            elif param_type == "discrete":
                builder.add_discrete(
                    name=param_name,
                    min_value=param_config["min_value"],
                    max_value=param_config["max_value"],
                    step_size=param_config.get("step_size", 1),
                )
            elif param_type == "categorical":
                builder.add_categorical(
                    name=param_name,
                    values=param_config["values"],
                )
            elif param_type == "boolean":
                builder.add_boolean(
                    name=param_name,
                    true_probability=param_config.get("true_probability", 0.5),
                )
            else:
                raise ValidationError(f"Invalid parameter type: {param_type}")

        return builder.build()

    def _build_objectives(self, objectives_config: list[dict[str, Any]]) -> list[Any]:
        """Build optimization objectives from configuration."""
        from src.optimization.core import ObjectiveDirection, OptimizationObjective

        objectives = []

        for obj_config in objectives_config:
            direction = ObjectiveDirection(obj_config["direction"])

            objective = OptimizationObjective(
                name=obj_config["name"],
                direction=direction,
                weight=Decimal(str(obj_config.get("weight", "1.0"))),
                constraint_min=Decimal(str(obj_config["constraint_min"]))
                if obj_config.get("constraint_min")
                else None,
                constraint_max=Decimal(str(obj_config["constraint_max"]))
                if obj_config.get("constraint_max")
                else None,
                is_primary=obj_config.get("is_primary", False),
                description=obj_config.get("description", ""),
            )
            objectives.append(objective)

        return objectives

    def _create_objective_function(self, function_name: str) -> Any:
        """Create objective function based on name (placeholder implementation)."""

        async def placeholder_objective(parameters: dict[str, Any]) -> dict[str, float]:
            """Placeholder objective function."""
            return {"objective_value": 0.0}

        return placeholder_objective
