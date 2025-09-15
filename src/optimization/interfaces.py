"""
Optimization service interfaces and protocols.

This module defines the service layer interfaces for the optimization framework,
ensuring proper separation of concerns and testability.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from src.core.types import StrategyConfig
from src.optimization.core import OptimizationObjective, OptimizationResult
from src.optimization.parameter_space import ParameterSpace


class OptimizationServiceProtocol(Protocol):
    """Protocol for optimization services."""

    async def optimize_strategy(
        self,
        strategy_name: str,
        parameter_space_config: dict[str, Any] | None = None,
        parameter_space: ParameterSpace | None = None,
        optimization_method: str = "brute_force",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize a trading strategy."""
        ...

    async def optimize_parameters(
        self,
        objective_function: Callable[[dict[str, Any]], Any],
        parameter_space: ParameterSpace,
        objectives: list[OptimizationObjective],
        method: str = "brute_force",
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize parameters using specified method."""
        ...


class BacktestIntegrationProtocol(Protocol):
    """Protocol for backtesting integration."""

    async def evaluate_strategy(
        self,
        strategy_config: StrategyConfig,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> dict[str, Decimal]:
        """Evaluate strategy performance."""
        ...

    def create_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> Callable[[dict[str, Any]], Any]:
        """Create objective function for strategy optimization."""
        ...


class OptimizationAnalysisProtocol(Protocol):
    """Protocol for optimization result analysis."""

    def analyze_results(
        self,
        optimization_result: OptimizationResult,
        parameter_space: ParameterSpace,
        objective_function: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze optimization results."""
        ...

    def calculate_parameter_importance(
        self,
        optimization_history: list[dict[str, Any]],
        parameter_names: list[str],
    ) -> dict[str, Decimal]:
        """Calculate parameter importance scores."""
        ...


class OptimizationRepositoryProtocol(Protocol):
    """Protocol for optimization result storage."""

    async def save_optimization_result(
        self,
        result: OptimizationResult,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save optimization result."""
        ...

    async def get_optimization_result(self, optimization_id: str) -> OptimizationResult | None:
        """Retrieve optimization result by ID."""
        ...

    async def list_optimization_results(
        self,
        strategy_name: str | None = None,
        limit: int = 100,
    ) -> list[OptimizationResult]:
        """List optimization results with optional filtering."""
        ...


class IOptimizationService(ABC):
    """Abstract base class for optimization services."""

    @abstractmethod
    async def optimize_strategy(
        self,
        strategy_name: str,
        parameter_space_config: dict[str, Any] | None = None,
        parameter_space: ParameterSpace | None = None,
        optimization_method: str = "brute_force",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize a trading strategy."""
        raise NotImplementedError("Subclasses must implement optimize_strategy")

    @abstractmethod
    async def optimize_parameters(
        self,
        objective_function: Callable[[dict[str, Any]], Any],
        parameter_space: ParameterSpace,
        objectives: list[OptimizationObjective],
        method: str = "brute_force",
        **kwargs: Any,
    ) -> OptimizationResult:
        """Optimize parameters using specified method."""
        raise NotImplementedError("Subclasses must implement optimize_parameters")

    @abstractmethod
    async def analyze_optimization_results(
        self,
        optimization_result: OptimizationResult,
        parameter_space: ParameterSpace,
    ) -> dict[str, Any]:
        """Analyze optimization results."""
        raise NotImplementedError("Subclasses must implement analyze_optimization_results")


class IBacktestIntegrationService(ABC):
    """Abstract base class for backtesting integration services."""

    @abstractmethod
    async def evaluate_strategy(
        self,
        strategy_config: StrategyConfig,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> dict[str, Decimal]:
        """Evaluate strategy performance using backtesting."""
        raise NotImplementedError("Subclasses must implement evaluate_strategy")

    @abstractmethod
    def create_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> Callable[[dict[str, Any]], Any]:
        """Create objective function for strategy optimization."""
        raise NotImplementedError("Subclasses must implement create_objective_function")


class IAnalysisService(ABC):
    """Abstract base class for optimization analysis services."""

    @abstractmethod
    async def analyze_optimization_results(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
        best_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze optimization results."""
        raise NotImplementedError("Subclasses must implement analyze_optimization_results")

    @abstractmethod
    async def analyze_parameter_importance(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
    ) -> list[Any]:
        """Analyze parameter importance."""
        raise NotImplementedError("Subclasses must implement analyze_parameter_importance")


class AnalysisServiceProtocol(Protocol):
    """Protocol for optimization analysis services."""

    async def analyze_optimization_results(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
        best_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze optimization results."""
        ...

    async def analyze_parameter_importance(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
    ) -> list[Any]:
        """Analyze parameter importance."""
        ...
