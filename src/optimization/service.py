"""
Optimization service implementation.

This module provides the main optimization service that coordinates
optimization algorithms, backtesting integration, and result analysis
following the service layer pattern.
"""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.base import BaseService
from src.core.exceptions import OptimizationError, ValidationError
from src.optimization.bayesian import BayesianConfig, BayesianOptimizer
from src.optimization.brute_force import BruteForceOptimizer, GridSearchConfig, ValidationConfig
from src.optimization.core import OptimizationObjective, OptimizationResult
from src.optimization.interfaces import (
    BacktestIntegrationProtocol,
    IOptimizationService,
    OptimizationRepositoryProtocol,
)
from src.optimization.parameter_space import ParameterSpace


class OptimizationService(BaseService, IOptimizationService):
    """
    Main optimization service implementation.

    Provides high-level optimization capabilities while properly separating
    concerns between optimization algorithms, backtesting, and result storage.
    """

    def __init__(
        self,
        backtest_integration: BacktestIntegrationProtocol | None = None,
        optimization_repository: OptimizationRepositoryProtocol | None = None,
    ):
        """
        Initialize optimization service.

        Args:
            backtest_integration: Backtesting integration service
            optimization_repository: Optimization result repository
        """
        super().__init__()
        self._backtest_integration = backtest_integration
        self._optimization_repository = optimization_repository
        # Don't create ResultsAnalyzer directly - should be injected if needed
        self._results_analyzer = None

        self.logger.info("OptimizationService initialized")

    async def optimize_strategy(
        self,
        strategy_name: str,
        parameter_space: ParameterSpace,
        optimization_method: str = "brute_force",
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
        **optimizer_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize a trading strategy.

        Args:
            strategy_name: Name of strategy to optimize
            parameter_space: Parameter space for optimization
            optimization_method: Method to use ("brute_force" or "bayesian")
            data_start_date: Start date for backtesting data
            data_end_date: End date for backtesting data
            initial_capital: Initial capital for backtesting
            **optimizer_kwargs: Additional optimizer configuration

        Returns:
            Comprehensive optimization results
        """
        self.logger.info(
            "Starting strategy optimization",
            strategy=strategy_name,
            method=optimization_method,
            data_period=f"{data_start_date} to {data_end_date}",
        )

        try:
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
                await self._optimization_repository.save_optimization_result(
                    optimization_result,
                    metadata={
                        "strategy_name": strategy_name,
                        "optimization_method": optimization_method,
                        "data_start_date": data_start_date,
                        "data_end_date": data_end_date,
                        "initial_capital": initial_capital,
                    },
                )

            self.logger.info(
                "Strategy optimization completed",
                strategy=strategy_name,
                optimal_value=float(optimization_result.optimal_objective_value),
                iterations=optimization_result.iterations_completed,
            )

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
            self.logger.error(f"Strategy optimization failed: {e}")
            raise OptimizationError(f"Strategy optimization failed: {e}") from e

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
            raise ValidationError(f"Unknown optimization method: {method}")

        return await optimizer.optimize(objective_function)

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
            # Create mock optimization history for analysis
            optimization_history = [
                {
                    "parameters": optimization_result.optimal_parameters,
                    "objective_value": float(optimization_result.optimal_objective_value),
                    "performance": float(optimization_result.optimal_objective_value),
                }
            ]

            # Create results analyzer if not available
            if self._results_analyzer is None:
                from src.optimization.analysis import ResultsAnalyzer
                self._results_analyzer = ResultsAnalyzer()

            # Analyze results
            parameter_names = list(parameter_space.parameters.keys())
            analysis = self._results_analyzer.analyze_optimization_results(
                optimization_history,
                parameter_names,
                optimization_history[0],
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Results analysis failed: {e}")
            return {"error": str(e)}

    async def _create_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> Callable[[dict[str, Any]], Any]:
        """Create objective function for strategy optimization."""
        if self._backtest_integration:
            return self._backtest_integration.create_objective_function(
                strategy_name, data_start_date, data_end_date, initial_capital
            )
        else:
            # Return simulation function if no backtesting available
            return self._create_simulation_objective_function()

    def _create_simulation_objective_function(self) -> Callable[[dict[str, Any]], Any]:
        """Create simulation-based objective function for testing."""

        async def simulation_objective(parameters: dict[str, Any]) -> dict[str, float]:
            """Simulate strategy performance for testing."""
            position_size = float(parameters.get("position_size_pct", 0.02))
            stop_loss = float(parameters.get("stop_loss_pct", 0.02))
            take_profit = float(parameters.get("take_profit_pct", 0.04))

            # Simulate risk-return tradeoff
            risk_factor = position_size * 10
            risk_adjusted_return = 0.1 * (1 + risk_factor) * (1 - stop_loss * 2)

            # Simulate Sharpe ratio
            volatility = 0.15 * (1 + risk_factor)
            sharpe_ratio = risk_adjusted_return / volatility if volatility > 0 else 0

            # Simulate drawdown
            max_drawdown = volatility * 0.5

            # Simulate win rate
            win_rate = 0.5 * (take_profit / (take_profit + stop_loss))

            return {
                "total_return": risk_adjusted_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": 1.0 + risk_adjusted_return,
            }

        return simulation_objective

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
    ) -> BruteForceOptimizer:
        """Create brute force optimizer with configuration."""
        try:
            grid_config = GridSearchConfig(
                grid_resolution=kwargs.get("grid_resolution", 5),
                adaptive_refinement=kwargs.get("adaptive_refinement", True),
                batch_size=kwargs.get("batch_size", 10),
                early_stopping_enabled=kwargs.get("early_stopping", True),
            )

            validation_config = ValidationConfig(
                enable_cross_validation=kwargs.get("enable_cv", False),
                enable_walk_forward=kwargs.get("enable_wf", False),
            )

            return BruteForceOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                grid_config=grid_config,
                validation_config=validation_config,
            )
        except Exception as e:
            self.logger.error(f"Failed to create brute force optimizer: {e}")
            raise OptimizationError(f"Failed to create brute force optimizer: {e}") from e

    async def _create_bayesian_optimizer(
        self,
        objectives: list[OptimizationObjective],
        parameter_space: ParameterSpace,
        **kwargs: Any,
    ) -> BayesianOptimizer:
        """Create Bayesian optimizer with configuration."""
        try:
            bayesian_config = BayesianConfig(
                n_initial_points=kwargs.get("n_initial", 10),
                n_calls=kwargs.get("n_calls", 50),
                batch_size=kwargs.get("batch_size", 1),
            )

            return BayesianOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                bayesian_config=bayesian_config,
            )
        except Exception as e:
            self.logger.error(f"Failed to create Bayesian optimizer: {e}")
            raise OptimizationError(f"Failed to create Bayesian optimizer: {e}") from e
