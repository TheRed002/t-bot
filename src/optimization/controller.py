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


class OptimizationController(BaseComponent):
    """
    Controller for optimization operations.

    Handles HTTP requests, validation, and orchestrates optimization
    operations through the service layer.
    """

    def __init__(
        self,
        optimization_service: IOptimizationService,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize optimization controller.

        Args:
            optimization_service: Optimization service instance
            name: Controller name for identification
            config: Controller configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "OptimizationController", config, correlation_id)
        self._optimization_service = optimization_service

        # Add service dependency
        self.add_dependency("OptimizationService")

        self._logger.info("OptimizationController initialized")

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

        # Delegate to service layer - let service handle parameter space building
        result = await self._optimization_service.optimize_strategy(
            strategy_name=strategy_name,
            parameter_space_config=parameter_space_config,
            optimization_method=optimization_method,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            initial_capital=initial_capital,
            **kwargs,
        )

        self._logger.info(
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

        # Delegate to service layer - let service handle parameter space and objective building
        result = await self._optimization_service.optimize_parameters_with_config(
            objective_function_name=objective_function_name,
            parameter_space_config=parameter_space_config,
            objectives_config=objectives_config,
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
                f"Invalid optimization method: {method}. Must be 'brute_force' or 'bayesian'"
            )
