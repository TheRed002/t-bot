"""
Backtesting integration service for optimization.

This module provides a dedicated service for integrating optimization
with the backtesting system, following proper service layer patterns.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.backtesting.service import BacktestRequest, BacktestService
from src.core.base import BaseService
from src.core.exceptions import OptimizationError
from src.core.types import StrategyConfig, TradingMode
from src.optimization.interfaces import IBacktestIntegrationService


class BacktestIntegrationService(BaseService, IBacktestIntegrationService):
    """
    Service for integrating optimization with backtesting.

    Provides clean abstraction between optimization and backtesting systems,
    handling strategy evaluation and performance metric extraction.
    """

    def __init__(self, backtest_service: BacktestService | None = None):
        """
        Initialize backtesting integration service.

        Args:
            backtest_service: Backtesting service instance
        """
        super().__init__()
        self._backtest_service = backtest_service

        self.logger.info("BacktestIntegrationService initialized")

    async def evaluate_strategy(
        self,
        strategy_config: StrategyConfig,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_capital: Decimal = Decimal("100000"),
    ) -> dict[str, float]:
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

        except Exception as e:
            self.logger.warning(f"Strategy evaluation failed: {e}")
            # Return poor performance for failed evaluations
            return {
                "total_return": -0.1,
                "sharpe_ratio": -1.0,
                "max_drawdown": 0.5,
                "win_rate": 0.3,
                "profit_factor": 0.5,
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

        async def strategy_objective(parameters: dict[str, Any]) -> dict[str, float]:
            """
            Evaluate strategy performance with given parameters.

            Args:
                parameters: Strategy parameters to evaluate

            Returns:
                Dictionary of performance metrics
            """
            try:
                # Create strategy configuration
                strategy_config = StrategyConfig(
                    name=strategy_name,
                    strategy_type="static",
                    enabled=True,
                    symbol=parameters.get("symbol", "BTCUSDT"),
                    timeframe=parameters.get("timeframe", "1h"),
                    parameters=parameters,
                )

                # Evaluate strategy
                return await self.evaluate_strategy(
                    strategy_config=strategy_config,
                    start_date=data_start_date,
                    end_date=data_end_date,
                    initial_capital=initial_capital,
                )

            except Exception as e:
                self.logger.warning(f"Strategy evaluation failed: {e}")
                # Return poor performance for failed evaluations
                return {
                    "total_return": -0.1,
                    "sharpe_ratio": -1.0,
                    "max_drawdown": 0.5,
                    "win_rate": 0.3,
                    "profit_factor": 0.5,
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

        # Create BacktestRequest
        request = BacktestRequest(
            parameters=backtest_config,
            symbols=[getattr(strategy_config, "symbol", "BTCUSDT")],
            strategy_name=strategy_config.name,
            start_date=backtest_config["start_date"],
            end_date=backtest_config["end_date"],
            initial_capital=backtest_config["initial_capital"],
            dry_run=True,
            max_positions=10,
        )

        # Execute backtest
        result = await self._backtest_service.run_backtest(request)
        return result

    def _extract_performance_metrics(self, backtest_result: Any) -> dict[str, float]:
        """Extract performance metrics from backtest result."""
        try:
            # Extract key metrics from backtest result
            total_return = float(getattr(backtest_result, "total_return", 0))
            sharpe_ratio = float(getattr(backtest_result, "sharpe_ratio", 0))
            max_drawdown = float(getattr(backtest_result, "max_drawdown", 0))

            # Calculate additional metrics
            trades = getattr(backtest_result, "trades", [])
            if trades:
                winning_trades = [t for t in trades if getattr(t, "pnl", 0) > 0]
                losing_trades = [t for t in trades if getattr(t, "pnl", 0) < 0]

                win_rate = len(winning_trades) / len(trades)

                # Profit factor
                gross_profit = sum(getattr(t, "pnl", 0) for t in winning_trades)
                gross_loss = abs(sum(getattr(t, "pnl", 0) for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            else:
                win_rate = 0.0
                profit_factor = 1.0

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": abs(max_drawdown),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades),
                "winning_trades": len([t for t in trades if getattr(t, "pnl", 0) > 0]),
                "losing_trades": len([t for t in trades if getattr(t, "pnl", 0) <= 0]),
            }

        except Exception as e:
            self.logger.warning(f"Failed to extract performance metrics: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 1.0,
            }

    def _simulate_performance(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Simulate strategy performance for testing."""
        # Simple simulation based on parameters
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

        # Simulate win rate based on stop loss / take profit ratio
        win_rate = 0.5 * (take_profit / (take_profit + stop_loss))

        return {
            "total_return": risk_adjusted_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": 1.0 + risk_adjusted_return,
        }
