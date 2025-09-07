"""
Integration utilities for optimization with existing T-Bot systems.

This module provides legacy integration utilities and convenience functions
for backward compatibility. New code should use the service layer directly.

DEPRECATED: Use OptimizationService and related services instead.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.optimization.interfaces import IOptimizationService
from src.optimization.parameter_space import ParameterSpace


class OptimizationIntegration(BaseComponent):
    """
    DEPRECATED: Legacy integration class for optimization.

    Use OptimizationService directly instead of this wrapper class.
    This class is maintained for backward compatibility only.
    """

    def __init__(
        self,
        optimization_service: IOptimizationService | None = None,
    ):
        """
        Initialize optimization integration.

        Args:
            optimization_service: Optimization service instance
        """
        super().__init__()
        self._optimization_service = optimization_service

        if not optimization_service:
            # Create a default service for backward compatibility
            from src.optimization.factory import create_optimization_service

            self._optimization_service = create_optimization_service(use_repository=False)

        self.logger.warning(
            "OptimizationIntegration is deprecated. Use OptimizationService directly."
        )

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
        DEPRECATED: Optimize a trading strategy.

        Use OptimizationService.optimize_strategy() directly instead.
        """
        self.logger.warning("Use OptimizationService.optimize_strategy() directly")

        return await self._optimization_service.optimize_strategy(
            strategy_name=strategy_name,
            parameter_space=parameter_space,
            optimization_method=optimization_method,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            initial_capital=initial_capital,
            **optimizer_kwargs,
        )


# Factory functions for common optimization scenarios


def create_strategy_optimization_space() -> ParameterSpace:
    """Create parameter space for general strategy optimization."""
    from src.optimization.parameter_space import ParameterSpaceBuilder

    builder = ParameterSpaceBuilder()

    return (
        builder.add_continuous("position_size_pct", Decimal("0.01"), Decimal("0.05"), precision=3)
        .add_continuous("stop_loss_pct", Decimal("0.005"), Decimal("0.03"), precision=3)
        .add_continuous("take_profit_pct", Decimal("0.01"), Decimal("0.08"), precision=3)
        .add_discrete("lookback_period", 5, 50, step_size=5)
        .add_categorical("timeframe", ["1m", "5m", "15m", "30m", "1h", "4h"])
        .add_continuous("confidence_threshold", Decimal("0.5"), Decimal("0.9"), precision=2)
        .build()
    )


def create_risk_optimization_space() -> ParameterSpace:
    """Create parameter space for risk management optimization."""
    from src.optimization.parameter_space import ParameterSpaceBuilder

    builder = ParameterSpaceBuilder()

    return (
        builder.add_continuous(
            "max_portfolio_exposure", Decimal("0.5"), Decimal("0.95"), precision=2
        )
        .add_discrete("max_positions", 1, 20)
        .add_continuous("max_drawdown_limit", Decimal("0.05"), Decimal("0.25"), precision=3)
        .add_continuous("var_confidence_level", Decimal("0.9"), Decimal("0.99"), precision=3)
        .add_continuous("correlation_threshold", Decimal("0.7"), Decimal("0.95"), precision=2)
        .add_boolean("enable_correlation_breaker", true_probability=Decimal("0.8"))
        .build()
    )


async def optimize_strategy_demo(
    strategy_name: str = "mean_reversion", optimization_method: str = "brute_force"
) -> dict[str, Any]:
    """
    DEPRECATED: Demo function showing strategy optimization.

    Use OptimizationService directly instead.
    """
    # Create integration instance (deprecated)
    integration = OptimizationIntegration()

    # Create parameter space
    parameter_space = create_strategy_optimization_space()

    # Run optimization
    results = await integration.optimize_strategy(
        strategy_name=strategy_name,
        parameter_space=parameter_space,
        optimization_method=optimization_method,
        grid_resolution=3,  # Small for demo
        n_calls=20,  # Small for demo
        initial_capital=Decimal("100000"),
    )

    return results
