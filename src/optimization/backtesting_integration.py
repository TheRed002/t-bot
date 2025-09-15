"""
Backtesting integration service for optimization.

This module provides a dedicated service for integrating optimization
with the backtesting system, following proper service layer patterns.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, cast

import structlog

from src.backtesting.interfaces import BacktestServiceInterface
from src.backtesting.service import BacktestRequest
from src.core.base import BaseService
from src.core.exceptions import OptimizationError, ServiceError, ValidationError
from src.core.types import StrategyConfig, TradingMode
from src.optimization.interfaces import IBacktestIntegrationService


class BacktestIntegrationService(BaseService, IBacktestIntegrationService):
    """
    Service for integrating optimization with backtesting.

    Provides clean abstraction between optimization and backtesting systems,
    handling strategy evaluation and performance metric extraction.
    """

    def __init__(
        self,
        backtest_service: BacktestServiceInterface | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize backtesting integration service.

        Args:
            backtest_service: Backtesting service instance
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "BacktestIntegrationService", config, correlation_id)
        self._backtest_service = backtest_service

        # Add dependencies
        if backtest_service:
            self.add_dependency("BacktestService")

        cast(structlog.BoundLogger, self._logger).info("BacktestIntegrationService initialized")

    async def evaluate_strategy(
        self,
        strategy_config: StrategyConfig,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> dict[str, Decimal]:
        """
        Evaluate strategy performance using backtesting.

        Args:
            strategy_config: Strategy configuration
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital amount

        Returns:
            Dictionary of performance metrics
        """
        if not self._backtest_service:
            # Return simulated performance if no backtesting service available
            return self._simulate_performance(strategy_config.parameters)

        try:
            # Run backtest
            result = await self._run_backtest(
                strategy_config, start_date, end_date, initial_capital
            )

            # Extract performance metrics
            return self._extract_performance_metrics(result)

        except (ServiceError, ValidationError) as e:
            # Handle backtesting-specific exceptions
            cast(structlog.BoundLogger, self._logger).error(
                f"Backtesting service error during strategy evaluation: {e}"
            )
            raise OptimizationError(f"Backtesting failed: {e}") from e
        except Exception as e:
            cast(structlog.BoundLogger, self._logger).warning(
                f"Strategy evaluation failed with unexpected error: {e}"
            )
            # Return poor performance for failed evaluations
            return {
                "total_return": Decimal("-0.1"),
                "sharpe_ratio": Decimal("-1.0"),
                "max_drawdown": Decimal("0.5"),
                "win_rate": Decimal("0.3"),
                "profit_factor": Decimal("0.5"),
            }

    def create_objective_function(
        self,
        strategy_name: str,
        data_start_date: datetime | None = None,
        data_end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> Callable[[dict[str, Any]], Any]:
        """
        Create objective function for strategy optimization.

        Args:
            strategy_name: Name of strategy to optimize
            data_start_date: Start date for backtesting data
            data_end_date: End date for backtesting data
            initial_capital: Initial capital amount

        Returns:
            Async objective function for optimization
        """

        async def strategy_objective(parameters: dict[str, Any]) -> dict[str, Decimal]:
            """
            Evaluate strategy performance with given parameters.

            Args:
                parameters: Strategy parameters to evaluate

            Returns:
                Dictionary of performance metrics
            """
            try:
                # Create strategy configuration with proper required fields
                from src.core.types.strategy import StrategyType

                strategy_config = StrategyConfig(
                    strategy_id=f"optimization_{strategy_name}",
                    strategy_type=StrategyType.CUSTOM,
                    name=strategy_name,
                    symbol=parameters.get("symbol", "BTCUSDT"),
                    timeframe=parameters.get("timeframe", "1h"),
                    enabled=True,
                    parameters=parameters,
                )

                # Evaluate strategy
                return await self.evaluate_strategy(
                    strategy_config=strategy_config,
                    start_date=data_start_date,
                    end_date=data_end_date,
                    initial_capital=initial_capital,
                )

            except (ServiceError, ValidationError) as e:
                # Re-raise backtesting exceptions as optimization errors
                cast(structlog.BoundLogger, self._logger).error(
                    f"Backtesting error in objective function: {e}"
                )
                raise OptimizationError(f"Objective function evaluation failed: {e}") from e
            except Exception as e:
                cast(structlog.BoundLogger, self._logger).warning(
                    f"Strategy evaluation failed: {e}"
                )
                # Return poor performance for failed evaluations
                return {
                    "total_return": Decimal("-0.1"),
                    "sharpe_ratio": Decimal("-1.0"),
                    "max_drawdown": Decimal("0.5"),
                    "win_rate": Decimal("0.3"),
                    "profit_factor": Decimal("0.5"),
                }

        return strategy_objective

    async def _run_backtest(
        self,
        strategy_config: StrategyConfig,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> Any:
        """Run backtesting for strategy evaluation."""
        if not self._backtest_service:
            raise OptimizationError("Backtesting service not available")

        # Configure backtesting parameters
        backtest_config = {
            "strategy_config": strategy_config,
            "start_date": start_date or (datetime.now(timezone.utc) - timedelta(days=365)),
            "end_date": end_date or datetime.now(timezone.utc),
            "initial_capital": initial_capital,
            "trading_mode": TradingMode.BACKTEST,
        }

        # Create BacktestRequest with correct field names and proper type handling
        try:
            # Convert StrategyConfig to dict properly
            if isinstance(strategy_config, StrategyConfig):
                strategy_dict = {
                    "strategy_id": strategy_config.strategy_id,
                    "strategy_type": strategy_config.strategy_type.value
                    if hasattr(strategy_config.strategy_type, "value")
                    else str(strategy_config.strategy_type),
                    "name": strategy_config.name,
                    "symbol": strategy_config.symbol,
                    "timeframe": getattr(strategy_config, "timeframe", "1h"),
                    "enabled": strategy_config.enabled,
                    "parameters": strategy_config.parameters,
                }
            else:
                # Fallback for dict-like objects
                strategy_dict = (
                    dict(strategy_config) if hasattr(strategy_config, "keys") else strategy_config
                )
                # Ensure required fields are present
                if "timeframe" not in strategy_dict:
                    strategy_dict["timeframe"] = "1h"

            request = BacktestRequest(
                strategy_config=strategy_dict,
                symbols=[strategy_config.symbol],
                start_date=cast(datetime, backtest_config["start_date"]),
                end_date=cast(datetime, backtest_config["end_date"]),
                initial_capital=cast(Decimal, backtest_config["initial_capital"]),
                max_open_positions=10,
                exchange="binance",  # Add missing required field
                timeframe=strategy_config.timeframe
                if hasattr(strategy_config, "timeframe")
                else "1h",
            )
        except (AttributeError, TypeError) as e:
            raise OptimizationError(f"Invalid strategy configuration for backtesting: {e}") from e

        # Execute backtest
        result = await self._backtest_service.run_backtest(request)
        return result

    def _extract_performance_metrics(self, backtest_result: Any) -> dict[str, Decimal]:
        """Extract performance metrics from backtest result."""
        try:
            # Extract key metrics from backtest result
            total_return = Decimal(str(getattr(backtest_result, "total_return", 0)))
            sharpe_ratio = Decimal(str(getattr(backtest_result, "sharpe_ratio", 0)))
            max_drawdown = Decimal(str(getattr(backtest_result, "max_drawdown", 0)))

            # Calculate additional metrics
            trades = getattr(backtest_result, "trades", [])
            if trades:
                winning_trades = [t for t in trades if Decimal(str(getattr(t, "pnl", 0))) > 0]
                losing_trades = [t for t in trades if Decimal(str(getattr(t, "pnl", 0))) < 0]

                win_rate = Decimal(str(len(winning_trades))) / Decimal(str(len(trades)))

                # Profit factor
                gross_profit = sum(Decimal(str(getattr(t, "pnl", 0))) for t in winning_trades)
                gross_loss = abs(sum(Decimal(str(getattr(t, "pnl", 0))) for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("1")
            else:
                win_rate = Decimal("0")
                profit_factor = Decimal("1")

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": abs(max_drawdown),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": Decimal(str(len(trades))),
                "winning_trades": Decimal(
                    str(len([t for t in trades if Decimal(str(getattr(t, "pnl", 0))) > 0]))
                ),
                "losing_trades": Decimal(
                    str(len([t for t in trades if Decimal(str(getattr(t, "pnl", 0))) <= 0]))
                ),
            }

        except Exception as e:
            cast(structlog.BoundLogger, self._logger).warning(
                f"Failed to extract performance metrics: {e}"
            )
            return {
                "total_return": Decimal("0"),
                "sharpe_ratio": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "win_rate": Decimal("0"),
                "profit_factor": Decimal("1"),
            }

    def _simulate_performance(self, parameters: dict[str, Any]) -> dict[str, Decimal]:
        """Simulate strategy performance for testing."""
        # Simple simulation based on parameters - use Decimal for financial precision
        position_size = Decimal(str(parameters.get("position_size_pct", 0.02)))
        stop_loss = Decimal(str(parameters.get("stop_loss_pct", 0.02)))
        take_profit = Decimal(str(parameters.get("take_profit_pct", 0.04)))

        # Simulate risk-return tradeoff
        risk_factor = position_size * Decimal("10")
        risk_adjusted_return = (
            Decimal("0.1")
            * (Decimal("1") + risk_factor)
            * (Decimal("1") - stop_loss * Decimal("2"))
        )

        # Simulate Sharpe ratio
        volatility = Decimal("0.15") * (Decimal("1") + risk_factor)
        sharpe_ratio = risk_adjusted_return / volatility if volatility > 0 else Decimal("0")

        # Simulate drawdown
        max_drawdown = volatility * Decimal("0.5")

        # Simulate win rate based on stop loss / take profit ratio
        win_rate = Decimal("0.5") * (take_profit / (take_profit + stop_loss))

        return {
            "total_return": risk_adjusted_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": Decimal("1") + risk_adjusted_return,
        }
