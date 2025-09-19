"""
Optimization service implementation.

This module provides the main optimization service that coordinates
optimization algorithms, backtesting integration, and result analysis
following the service layer pattern.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Union

from src.core.base import BaseService
from src.core.event_constants import OptimizationEvents
from src.core.exceptions import OptimizationError, ValidationError
from src.optimization.bayesian import BayesianConfig, BayesianOptimizer
from src.optimization.brute_force import BruteForceOptimizer, GridSearchConfig
from src.optimization.core import OptimizationObjective, OptimizationResult
from src.optimization.interfaces import (
    IAnalysisService,
    IBacktestIntegrationService,
    IOptimizationService,
    OptimizationRepositoryProtocol,
)
from src.optimization.parameter_space import ParameterSpace
from src.optimization.parameter_space_service import ParameterSpaceService
from src.optimization.result_transformation_service import ResultTransformationService
from src.utils.messaging_patterns import (
    ErrorPropagationMixin,
    MessagePattern,
)

# Production Configuration Constants
DEFAULT_INITIAL_CAPITAL = Decimal("100000")
DEFAULT_POSITION_SIZE_PCT = Decimal("0.02")
DEFAULT_STOP_LOSS_PCT = Decimal("0.02")
DEFAULT_TAKE_PROFIT_PCT = Decimal("0.04")
DEFAULT_RISK_MULTIPLIER = Decimal("10")
DEFAULT_BASE_RETURN = Decimal("0.1")
DEFAULT_BASE_VOLATILITY = Decimal("0.15")
DEFAULT_DRAWDOWN_FACTOR = Decimal("0.5")
DEFAULT_WIN_RATE_BASE = Decimal("0.5")


class OptimizationService(BaseService, IOptimizationService, ErrorPropagationMixin):
    """
    Main optimization service implementation.

    Provides high-level optimization capabilities while properly separating
    concerns between optimization algorithms, backtesting, and result storage.
    """

    def __init__(
        self,
        backtest_integration: "Union[IBacktestIntegrationService, None]" = None,
        optimization_repository: "Union[OptimizationRepositoryProtocol, None]" = None,
        analysis_service: "Union[IAnalysisService, None]" = None,
        parameter_space_service: "Union[ParameterSpaceService, None]" = None,
        result_transformation_service: "Union[ResultTransformationService, None]" = None,
        name: Union[str, None] = None,
        config: Union[dict[str, Any], None] = None,
        correlation_id: Union[str, None] = None,
    ):
        """
        Initialize optimization service.

        Args:
            backtest_integration: Backtesting integration service
            optimization_repository: Optimization result repository
            analysis_service: Analysis service for result analysis
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "OptimizationService", config, correlation_id)
        self._backtest_integration = backtest_integration
        self._optimization_repository = optimization_repository
        self._analysis_service = analysis_service
        self._parameter_space_service = parameter_space_service or ParameterSpaceService()
        self._result_transformation_service = result_transformation_service or ResultTransformationService()

        # Add dependencies
        if backtest_integration:
            self.add_dependency("BacktestIntegration")
        if optimization_repository:
            self.add_dependency("OptimizationRepository")
        if analysis_service:
            self.add_dependency("AnalysisService")
        self.add_dependency("ParameterSpaceService")
        self.add_dependency("ResultTransformationService")

        self._logger.info("OptimizationService initialized")

    async def optimize_strategy(
        self,
        strategy_name: str,
        parameter_space_config: dict[str, Any] | None = None,
        parameter_space: ParameterSpace | None = None,
        optimization_method: str = "brute_force",
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = DEFAULT_INITIAL_CAPITAL,
        **optimizer_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize a trading strategy.

        Args:
            strategy_name: Name of strategy to optimize
            parameter_space_config: Configuration for parameter space (alternative to parameter_space)
            parameter_space: Pre-built parameter space (alternative to parameter_space_config)
            optimization_method: Method to use ("brute_force" or "bayesian")
            data_start_date: Start date for backtesting data
            data_end_date: End date for backtesting data
            initial_capital: Initial capital for backtesting
            **optimizer_kwargs: Additional optimizer configuration

        Returns:
            Comprehensive optimization results
        """
        self._logger.info(
            "Starting strategy optimization",
            strategy=strategy_name,
            method=optimization_method,
            data_period=f"{data_start_date} to {data_end_date}",
        )

        # Emit optimization started event with consistent data transformation
        event_data = self._transform_event_data(
            {
                "strategy_name": strategy_name,
                "optimization_method": optimization_method,
                "data_start_date": data_start_date.isoformat() if data_start_date else None,
                "data_end_date": data_end_date.isoformat() if data_end_date else None,
                "initial_capital": str(initial_capital),
                "processing_mode": "batch",  # Optimization is batch-oriented
                "message_pattern": MessagePattern.REQ_REPLY.value,  # Use enum for consistency
                "target_processing_mode": self._determine_target_processing_mode_for_strategy(strategy_name),
                "source": "optimization_service",
                "operation": "strategy_optimization",
            }
        )

        # Validate event data
        self._validate_event_data(event_data)

        # Emit optimization started event
        try:
            if hasattr(self, "emit_event") and callable(self.emit_event):
                await self.emit_event(OptimizationEvents.STARTED, event_data)
        except Exception as e:
            # Graceful fallback - log but don't fail optimization
            self._logger.debug(f"Event emission failed: {e}")

        try:
            # Build parameter space if configuration provided
            if parameter_space is None:
                if parameter_space_config is None:
                    validation_error = ValidationError(
                        "Either parameter_space or parameter_space_config must be provided",
                        error_code="OPT_001",
                        field_name="parameter_space",
                    )
                    # Use consistent validation error propagation
                    self.propagate_validation_error(
                        validation_error, "strategy_optimization_parameter_validation"
                    )
                    raise validation_error
                # Delegate to specialized service
                parameter_space = self._parameter_space_service.build_parameter_space(parameter_space_config)

            # Create objective function
            objective_function = await self._create_objective_function(
                strategy_name=strategy_name,
                data_start_date=data_start_date,
                data_end_date=data_end_date,
                initial_capital=initial_capital,
            )

            # Create optimization objectives
            objectives = self._create_standard_trading_objectives()

            # Run optimization
            optimization_result = await self.optimize_parameters(
                objective_function=objective_function,
                parameter_space=parameter_space,
                objectives=objectives,
                method=optimization_method,
                **optimizer_kwargs,
            )

            # Analyze results
            analysis_results = await self.analyze_optimization_results(
                optimization_result, parameter_space
            )

            # Save results if repository available
            if self._optimization_repository:
                result_id = await self._optimization_repository.save_optimization_result(
                    optimization_result,
                    metadata={
                        "strategy_name": strategy_name,
                        "optimization_method": optimization_method,
                        "data_start_date": data_start_date,
                        "data_end_date": data_end_date,
                        "initial_capital": initial_capital,
                    },
                )
                # Emit result saved event at service layer
                await self._emit_result_saved_event(optimization_result, result_id)

            self._logger.info(
                "Strategy optimization completed",
                strategy=strategy_name,
                optimal_value=optimization_result.optimal_objective_value,
                iterations=optimization_result.iterations_completed,
            )

            # Emit optimization completed event with consistent data transformation
            completion_event_data = self._transform_event_data(
                {
                    "optimization_id": optimization_result.optimization_id,
                    "algorithm_name": optimization_result.algorithm_name,
                    "optimal_objective_value": str(optimization_result.optimal_objective_value),
                    "convergence_achieved": optimization_result.convergence_achieved,
                    "iterations_completed": optimization_result.iterations_completed,
                    "strategy_name": strategy_name,
                    "optimization_method": optimization_method,
                    "processing_mode": "batch",
                    "target_processing_mode": self._determine_target_processing_mode_for_strategy(strategy_name),
                    "message_pattern": MessagePattern.PUB_SUB.value,  # Completion events are broadcast
                    "source": "optimization_service",
                    "operation": "strategy_optimization_completed",
                    "cross_module_event": True,  # This event may be consumed by other modules
                }
            )

            # Validate completion event data
            self._validate_event_data(completion_event_data)

            # Emit completion event
            try:
                if hasattr(self, "emit_event") and callable(self.emit_event):
                    await self.emit_event(OptimizationEvents.COMPLETED, completion_event_data)
            except Exception as e:
                # Log but don't fail the optimization for WebSocket issues
                self._logger.warning(f"Failed to emit completion event: {e}")

            return {
                "optimization_result": optimization_result,
                "analysis": analysis_results,
                "strategy_name": strategy_name,
                "optimization_method": optimization_method,
                "metadata": {
                    "data_start_date": data_start_date,
                    "data_end_date": data_end_date,
                    "initial_capital": initial_capital,
                    "optimization_timestamp": datetime.now(),
                },
            }

        except Exception as e:
            self._logger.error(f"Strategy optimization failed: {e}")

            # Emit optimization failed event with consistent data transformation
            error_event_data = self._transform_event_data(
                {
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "strategy_name": strategy_name,
                    "optimization_method": optimization_method,
                    "processing_mode": "batch",
                    "message_pattern": MessagePattern.PUB_SUB.value,  # Error events are broadcast
                    "source": "optimization_service",
                    "operation": "strategy_optimization",
                    "error_context": "strategy_optimization_execution",
                    "cross_module_event": True,
                }
            )

            # Validate error event data
            self._validate_event_data(error_event_data)

            # Emit error event
            try:
                if hasattr(self, "emit_event") and callable(self.emit_event):
                    await self.emit_event(OptimizationEvents.FAILED, error_event_data)
            except Exception as e:
                # Log but don't fail the optimization for WebSocket issues
                self._logger.warning(f"Failed to emit error event: {e}")

            # Use consistent error propagation patterns
            self.propagate_service_error(e, "strategy_optimization")

            raise OptimizationError(
                f"Strategy optimization failed: {e}",
                error_code="OPT_002",
                optimization_stage="strategy_optimization",
            ) from e

    async def optimize_strategy_parameters(
        self,
        strategy_id: str,
        optimization_request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters based on request from strategies module.

        Args:
            strategy_id: Strategy identifier
            optimization_request: Optimization request data

        Returns:
            Optimization results compatible with strategies module
        """
        try:
            # Extract strategy name from request
            strategy_name = optimization_request.get("strategy_name") or strategy_id
            optimization_config = optimization_request.get("optimization_config", {})
            current_parameters = optimization_request.get("current_parameters", {})

            # Delegate parameter space building to specialized service
            parameter_space_config = self._parameter_space_service.build_parameter_space_from_current(
                current_parameters
            )

            # Extract optimization configuration
            optimization_method = optimization_config.get("method", "brute_force")
            initial_capital = Decimal(
                str(optimization_config.get("initial_capital", DEFAULT_INITIAL_CAPITAL))
            )

            # Call the main optimization method
            result = await self.optimize_strategy(
                strategy_name=strategy_name,
                parameter_space_config=parameter_space_config,
                optimization_method=optimization_method,
                initial_capital=initial_capital,
                data_start_date=optimization_config.get("data_start_date"),
                data_end_date=optimization_config.get("data_end_date"),
                **optimization_config.get("optimizer_kwargs", {}),
            )

            # Transform result using specialized service
            return self._result_transformation_service.transform_for_strategies_module(
                result, optimization_request.get("current_parameters", {})
            )

        except Exception as e:
            self._logger.error(f"Strategy parameter optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "optimized_parameters": optimization_request.get("current_parameters", {}),
            }

    async def optimize_parameters(
        self,
        objective_function: Callable[[dict[str, Any]], Any],
        parameter_space: ParameterSpace,
        objectives: list[OptimizationObjective],
        method: str = "brute_force",
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Optimize parameters using specified method.

        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            objectives: Optimization objectives
            method: Optimization method to use
            **kwargs: Additional optimizer configuration

        Returns:
            Optimization result
        """
        if method == "brute_force":
            optimizer = await self._create_brute_force_optimizer(
                objectives, parameter_space, **kwargs
            )
        elif method == "bayesian":
            optimizer = await self._create_bayesian_optimizer(objectives, parameter_space, **kwargs)
        else:
            validation_error = ValidationError(
                f"Unknown optimization method: {method}",
                error_code="OPT_003",
                field_name="optimization_method",
                field_value=method,
            )
            # Use consistent validation error propagation
            self.propagate_validation_error(validation_error, "optimization_method_validation")
            raise validation_error

        return await optimizer.optimize(objective_function)

    async def optimize_parameters_with_config(
        self,
        objective_function_name: str,
        parameter_space_config: dict[str, Any],
        objectives_config: list[dict[str, Any]],
        method: str = "brute_force",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize parameters using configuration objects.

        Args:
            objective_function_name: Name of objective function to create
            parameter_space_config: Parameter space configuration
            objectives_config: Objectives configuration
            method: Optimization method to use
            **kwargs: Additional optimizer configuration

        Returns:
            Optimization results
        """
        # Build parameter space from configuration using specialized service
        parameter_space = self._parameter_space_service.build_parameter_space(parameter_space_config)

        # Build objectives from configuration
        objectives = self._build_objectives(objectives_config)

        # Create objective function
        objective_function = self._create_objective_function_by_name(objective_function_name)

        # Run optimization
        result = await self.optimize_parameters(
            objective_function=objective_function,
            parameter_space=parameter_space,
            objectives=objectives,
            method=method,
            **kwargs,
        )

        return result

    async def analyze_optimization_results(
        self,
        optimization_result: OptimizationResult,
        parameter_space: ParameterSpace,
    ) -> dict[str, Any]:
        """
        Analyze optimization results.

        Args:
            optimization_result: Optimization result to analyze
            parameter_space: Parameter space used in optimization

        Returns:
            Analysis results
        """
        try:
            if not self._analysis_service:
                self._logger.warning("No analysis service available, creating basic analysis")
                # Provide basic analysis when no service is available
                return {
                    "best_result_analysis": {
                        "optimal_parameters": optimization_result.optimal_parameters,
                        "optimal_objective_value": optimization_result.optimal_objective_value,
                        "convergence_achieved": optimization_result.convergence_achieved,
                        "iterations_completed": optimization_result.iterations_completed,
                    },
                    "parameter_correlations": {},
                    "analysis_status": "basic_fallback",
                }

            # Analysis service is available - proceed with service call
            # (analysis_request_data prepared for potential future audit logging)

            # Create mock optimization history for analysis
            optimization_history = [
                {
                    "parameters": optimization_result.optimal_parameters,
                    "objective_value": optimization_result.optimal_objective_value,
                    "performance": optimization_result.optimal_objective_value,
                }
            ]

            # Use injected analysis service
            parameter_names = list(parameter_space.parameters.keys())
            analysis = await self._analysis_service.analyze_optimization_results(
                optimization_history,
                parameter_names,
                optimization_history[0],
            )

            return analysis

        except Exception as e:
            self._logger.error(f"Results analysis failed: {e}")
            # Still return error dict for backward compatibility but also propagate the error
            self.propagate_service_error(e, "optimization_results_analysis")
            return {"error": str(e), "analysis_status": "failed"}

    async def _create_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = DEFAULT_INITIAL_CAPITAL,
    ) -> Callable[[dict[str, Any]], Any]:
        """Create objective function for strategy optimization."""
        # Yield control to event loop
        await asyncio.sleep(0)

        # Prepare data for backtesting integration
        if self._backtest_integration:
            # Backtesting integration available - proceed with service call
            # (backtest_request_data prepared for potential future audit logging)

            try:
                return self._backtest_integration.create_objective_function(
                    strategy_name, data_start_date, data_end_date, initial_capital
                )
            except Exception as e:
                # Use consistent error propagation for backtesting integration failures
                self.propagate_service_error(e, "backtesting_objective_function_creation")
                raise
        else:
            # Create a simulation-based objective if no backtesting is available
            # This uses the same simulation logic as BacktestIntegrationService
            self._logger.warning("No backtesting integration available, using simulation")
            return self._create_simulation_objective_function(
                strategy_name, data_start_date, data_end_date, initial_capital
            )

    def _create_simulation_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = DEFAULT_INITIAL_CAPITAL,
    ) -> Callable[[dict[str, Any]], Any]:
        """Create simulation-based objective function delegating to BacktestIntegrationService."""
        if self._backtest_integration:
            return self._backtest_integration.create_objective_function(
                strategy_name=strategy_name,
                data_start_date=data_start_date,
                data_end_date=data_end_date,
                initial_capital=initial_capital,
            )

        # Fallback simulation when no backtest integration available
        async def simple_simulation_objective(parameters: dict[str, Any]) -> dict[str, Decimal]:
            """Simple simulation fallback for testing."""
            await asyncio.sleep(0)

            position_size = Decimal(
                str(parameters.get("position_size_pct", DEFAULT_POSITION_SIZE_PCT))
            )
            stop_loss = Decimal(str(parameters.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)))
            take_profit = Decimal(str(parameters.get("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)))

            # Simple risk-return simulation
            risk_factor = position_size * DEFAULT_RISK_MULTIPLIER
            risk_adjusted_return = (
                DEFAULT_BASE_RETURN
                * (Decimal("1") + risk_factor)
                * (Decimal("1") - stop_loss * Decimal("2"))
            )
            volatility = DEFAULT_BASE_VOLATILITY * (Decimal("1") + risk_factor)
            sharpe_ratio = risk_adjusted_return / volatility if volatility > 0 else Decimal("0")
            max_drawdown = volatility * DEFAULT_DRAWDOWN_FACTOR
            win_rate = DEFAULT_WIN_RATE_BASE * (take_profit / (take_profit + stop_loss))

            return {
                "total_return": risk_adjusted_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": Decimal("1") + risk_adjusted_return,
            }

        return simple_simulation_objective

    def _create_standard_trading_objectives(self) -> list[OptimizationObjective]:
        """Create standard trading optimization objectives."""
        from src.optimization.core import ObjectiveDirection

        return [
            OptimizationObjective(
                name="sharpe_ratio",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("0.4"),
                constraint_min=Decimal("1.0"),
                is_primary=True,
                description="Risk-adjusted returns (Sharpe ratio)",
            ),
            OptimizationObjective(
                name="total_return",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("0.3"),
                constraint_min=Decimal("0.05"),
                description="Total portfolio return",
            ),
            OptimizationObjective(
                name="max_drawdown",
                direction=ObjectiveDirection.MINIMIZE,
                weight=Decimal("0.2"),
                constraint_max=Decimal("0.2"),
                description="Maximum drawdown",
            ),
            OptimizationObjective(
                name="win_rate",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("0.1"),
                constraint_min=Decimal("0.4"),
                description="Win rate percentage",
            ),
        ]

    async def _create_brute_force_optimizer(
        self,
        objectives: list[OptimizationObjective],
        parameter_space: ParameterSpace,
        **kwargs: Any,
    ) -> Any:
        """Create brute force optimizer with configuration."""
        # Yield control to event loop
        await asyncio.sleep(0)

        try:
            # Validate parameter space
            if parameter_space is None:
                raise OptimizationError("Parameter space cannot be None", error_code="OPT_004")
            # Create GridSearchConfig from kwargs
            grid_config_data = {
                k.replace("grid_", ""): v
                for k, v in kwargs.items()
                if k.startswith("grid_") and k != "grid_config"
            }
            grid_config = kwargs.get("grid_config") or GridSearchConfig(**grid_config_data)

            # Create real brute force optimizer
            return BruteForceOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                config=self._config,
                grid_config=grid_config,
                validation_config=kwargs.get("validation_config"),
            )
        except Exception as e:
            self._logger.error(f"Failed to create brute force optimizer: {e}")
            raise OptimizationError(
                f"Failed to create brute force optimizer: {e}",
                error_code="OPT_004",
                optimization_algorithm="brute_force",
            ) from e

    async def _create_bayesian_optimizer(
        self,
        objectives: list[OptimizationObjective],
        parameter_space: ParameterSpace,
        **kwargs: Any,
    ) -> Any:
        """Create Bayesian optimizer with configuration."""
        # Yield control to event loop
        await asyncio.sleep(0)

        try:
            # Validate parameter space
            if parameter_space is None:
                raise OptimizationError("Parameter space cannot be None", error_code="OPT_005")
            # Create BayesianConfig from kwargs
            bayesian_config_data = {
                k.replace("bayesian_", ""): v
                for k, v in kwargs.items()
                if k.startswith("bayesian_") and k != "bayesian_config"
            }
            bayesian_config = kwargs.get("bayesian_config") or BayesianConfig(
                **bayesian_config_data
            )

            # Create real Bayesian optimizer
            return BayesianOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                config=self._config,
                bayesian_config=bayesian_config,
            )
        except Exception as e:
            self._logger.error(f"Failed to create Bayesian optimizer: {e}")
            raise OptimizationError(
                f"Failed to create Bayesian optimizer: {e}",
                error_code="OPT_005",
                optimization_algorithm="bayesian",
            ) from e

    # Parameter space and result transformation methods moved to specialized services

    def _build_objectives(
        self, objectives_config: list[dict[str, Any]]
    ) -> list[OptimizationObjective]:
        """Build optimization objectives from configuration."""
        from src.optimization.core import ObjectiveDirection

        objectives = []

        for obj_config in objectives_config:
            try:
                direction = ObjectiveDirection(obj_config["direction"])

                # Safely convert constraint values to Decimal
                constraint_min = None
                constraint_max = None
                weight = Decimal("1.0")  # Default weight

                if obj_config.get("constraint_min"):
                    constraint_min = Decimal(str(obj_config["constraint_min"]))
                if obj_config.get("constraint_max"):
                    constraint_max = Decimal(str(obj_config["constraint_max"]))
                if obj_config.get("weight"):
                    weight = Decimal(str(obj_config["weight"]))

                objective = OptimizationObjective(
                    name=obj_config["name"],
                    direction=direction,
                    weight=weight,
                    constraint_min=constraint_min,
                    constraint_max=constraint_max,
                    is_primary=obj_config.get("is_primary", False),
                    description=obj_config.get("description", ""),
                )
                objectives.append(objective)
            except (ValueError, KeyError, TypeError) as e:
                self._logger.error(f"Failed to create objective from config {obj_config}: {e}")
                raise ValidationError(f"Invalid objective configuration: {e}") from e

        return objectives

    def _create_objective_function_by_name(
        self, function_name: str
    ) -> Callable[[dict[str, Any]], Any]:
        """Create objective function based on name."""

        async def placeholder_objective(parameters: dict[str, Any]) -> dict[str, Decimal]:
            """Placeholder objective function."""
            # Yield control to event loop
            await asyncio.sleep(0)
            return {"objective_value": 0.0}

        # In a real implementation, this would create different objective functions
        # based on the function_name parameter
        return placeholder_objective

    async def _emit_result_saved_event(
        self, optimization_result: OptimizationResult, result_id: str
    ) -> None:
        """Emit result saved event at service layer."""
        try:
            if hasattr(self, "emit_event") and callable(self.emit_event):
                event_data = self._transform_event_data(
                    {
                        "optimization_id": optimization_result.optimization_id,
                        "algorithm_name": optimization_result.algorithm_name,
                        "result_id": result_id,
                        "optimal_objective_value": str(optimization_result.optimal_objective_value),
                        "processing_mode": "batch",
                        "target_processing_mode": self._determine_target_processing_mode(
                            optimization_result
                        ),
                        "message_pattern": MessagePattern.PUB_SUB.value,
                        "source": "optimization_service",
                        "operation": "result_saved",
                        "cross_module_event": True,
                    }
                )

                await asyncio.wait_for(
                    self.emit_event(OptimizationEvents.RESULT_SAVED, event_data),
                    timeout=5.0,
                )
        except (AttributeError, asyncio.TimeoutError) as e:
            self._logger.debug(f"Event emission not available or timed out: {e}")
        except Exception as e:
            self._logger.warning(f"Failed to emit result saved event: {e}")

    def _determine_target_processing_mode(self, result: OptimizationResult) -> str:
        """Determine target processing mode based on optimization result."""
        # Check if this optimization is for real-time strategies
        if hasattr(result, "strategy_name") and result.strategy_name:
            strategy_name = result.strategy_name.lower()
            real_time_strategies = {
                "adaptive", "momentum", "volatility", "breakout",
                "trend", "mean_reversion", "market_making", "arbitrage"
            }
            for strategy_type in real_time_strategies:
                if strategy_type in strategy_name:
                    return "stream"

        # Check algorithm type
        if hasattr(result, "algorithm_name") and result.algorithm_name:
            algorithm_name = result.algorithm_name.lower()
            if "online" in algorithm_name or "adaptive" in algorithm_name:
                return "stream"

        return "batch"

    def _determine_target_processing_mode_for_strategy(self, strategy_name: str) -> str:
        """
        Determine the target processing mode for strategy communication.

        Args:
            strategy_name: Name of the strategy to optimize

        Returns:
            Target processing mode for cross-module alignment
        """
        # Most strategies operate in stream mode for real-time processing
        # but may accept batch mode for optimization operations
        strategy_stream_types = {
            "adaptive_momentum", "volatility_breakout", "trend_following",
            "mean_reversion", "market_making", "arbitrage"
        }

        # Check if strategy name contains stream-oriented strategy types
        for stream_type in strategy_stream_types:
            if stream_type in strategy_name.lower():
                return "stream"

        # Default to batch mode for optimization compatibility
        return "batch"

    def _apply_cross_module_alignment(self, data: dict[str, Any]) -> None:
        """
        Apply cross-module data alignment for strategies integration.

        Args:
            data: Event data to align
        """
        from src.utils.messaging_patterns import ProcessingParadigmAligner, BoundaryValidator

        # Check if this is cross-module data
        if data.get("cross_module_event") or data.get("target_processing_mode"):
            source_mode = data.get("processing_mode", "batch")
            target_mode = data.get("target_processing_mode", source_mode)

            if source_mode != target_mode:
                # Apply paradigm alignment
                aligned_data = ProcessingParadigmAligner.align_processing_modes(
                    source_mode, target_mode, data
                )
                data.update(aligned_data)

        # Apply boundary validation for optimization-to-strategies flow
        if data.get("operation_category") == "optimization":
            try:
                BoundaryValidator.validate_optimization_to_strategy_boundary(data)
            except Exception as e:
                self._logger.warning(f"Boundary validation warning: {e}")
                # Add fallback validation metadata
                data["boundary_validation_status"] = "warning"
                data["boundary_validation_message"] = str(e)

    async def shutdown(self) -> None:
        """Shutdown the optimization service and cleanup resources."""
        try:
            self._logger.info("Shutting down OptimizationService")

            # Call parent shutdown if available
            if hasattr(super(), "shutdown"):
                await super().shutdown()

            self._logger.info("OptimizationService shutdown completed")

        except Exception as e:
            self._logger.error(f"Error during OptimizationService shutdown: {e}")

    def _transform_event_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform event data to consistent format aligned with core module patterns.

        This method ensures all optimization events have consistent structure,
        financial data transformations, and required metadata fields.

        Args:
            data: Raw event data dictionary

        Returns:
            Transformed event data with consistent structure
        """
        from datetime import datetime, timezone

        if not isinstance(data, dict):
            # Transform non-dict data to standard format
            transformed = {
                "payload": data,
                "type": type(data).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "batch",  # Default for optimization
                "data_format": "event_data_v1",
                "module": "optimization",
            }
            return transformed

        # Ensure required metadata fields for cross-module consistency
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if "processing_mode" not in data:
            data["processing_mode"] = "batch"  # Optimization is typically batch-oriented

        if "data_format" not in data:
            data["data_format"] = "event_data_v1"

        if "module" not in data:
            data["module"] = "optimization"

        # Apply consistent financial data transformations for optimization events
        financial_fields = ["initial_capital", "optimal_objective_value", "capital", "value"]
        for field in financial_fields:
            if field in data and data[field] is not None:
                # Ensure financial values are decimal-formatted strings
                try:
                    if isinstance(data[field], (int, float)):
                        from decimal import Decimal

                        data[field] = str(Decimal(str(data[field])))
                    elif not isinstance(data[field], str):
                        data[field] = str(data[field])
                except (ValueError, TypeError):
                    # Leave as-is if conversion fails, but log it
                    self._logger.warning(
                        f"Could not convert financial field {field}: {data[field]}"
                    )

        # Add optimization-specific metadata
        if "operation" in data and not data.get("operation_category"):
            if "optimization" in data["operation"]:
                data["operation_category"] = "optimization"
            elif "analysis" in data["operation"]:
                data["operation_category"] = "analysis"
            else:
                data["operation_category"] = "unknown"

        # Apply cross-module boundary validation and transformation
        self._apply_cross_module_alignment(data)

        # Ensure message pattern is set for boundary validation
        if "message_pattern" not in data:
            # Default to PUB_SUB for events, REQ_REPLY for operations
            if data.get("cross_module_event", False):
                data["message_pattern"] = MessagePattern.PUB_SUB.value
            else:
                data["message_pattern"] = MessagePattern.REQ_REPLY.value

        return data

    def _validate_event_data(self, data: Any) -> None:
        """
        Basic validation for event data.

        Args:
            data: Event data to validate

        Raises:
            ValidationError: If data is invalid
        """
        if data is not None and not isinstance(data, dict):
            raise ValidationError(
                "Event data must be dict or None",
                field_name="event_data",
                field_value=type(data).__name__,
                expected_type="dict or None",
            )

    def _propagate_error_with_boundary_validation(
        self,
        error: Exception,
        context: str,
        source_module: str = "optimization",
        target_module: str = "backtesting",
    ) -> None:
        """
        Enhanced error propagation with boundary validation aligned with backtesting module.

        Args:
            error: Exception to propagate
            context: Error context/description
            source_module: Source module name
            target_module: Target module name
        """
        try:
            # Apply data transformation for error propagation
            from src.optimization.data_transformer import OptimizationDataTransformer

            # Create error data with boundary validation
            error_data = {
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context,
                "source_module": source_module,
                "target_module": target_module,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "stream",  # Align with backtesting error handling
            }

            # Apply cross-module validation
            validated_error_data = OptimizationDataTransformer.apply_cross_module_validation(
                error_data, source_module, target_module
            )

            # Log enhanced error with validation
            self._logger.error(
                f"Enhanced error propagation in {context}: {error}",
                extra={"validated_error_data": validated_error_data},
            )

        except Exception:
            # Fallback to standard error propagation
            pass

        # Use standard error propagation as fallback
        self.propagate_service_error(error, context)
