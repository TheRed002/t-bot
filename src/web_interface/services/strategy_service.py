"""
Strategy service for web interface business logic.

This service handles all strategy-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.types import HealthCheckResult

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError
from src.web_interface.interfaces import WebStrategyServiceInterface


class WebStrategyService(BaseComponent, WebStrategyServiceInterface):
    """Service handling strategy business logic for web interface."""

    def __init__(self, strategy_facade=None):
        super().__init__()
        self.strategy_facade = strategy_facade

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web strategy service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web strategy service cleaned up")

    async def get_formatted_strategies(self) -> list[dict[str, Any]]:
        """Get strategies with web-specific formatting."""
        try:
            if self.strategy_facade:
                # Get strategies from actual service
                raw_strategies = await self.strategy_facade.list_strategies()

                # Format for web interface
                formatted_strategies = []
                for strategy in raw_strategies:
                    formatted_strategy = {
                        "strategy_name": strategy.get("name", "unknown"),
                        "strategy_type": strategy.get("type", "unknown"),
                        "description": strategy.get("description", "No description available"),
                        "category": strategy.get("category", "general"),
                        "supported_exchanges": strategy.get("exchanges", ["binance"]),
                        "supported_symbols": strategy.get("symbols", ["BTCUSDT"]),
                        "risk_level": strategy.get("risk_level", "medium"),
                        "minimum_capital": Decimal(str(strategy.get("min_capital", 1000))),
                        "maximum_capital": Decimal(str(strategy.get("max_capital", 100000))),
                        "default_parameters": strategy.get("parameters", {}),
                        "performance_metrics": strategy.get("metrics", {}),
                        "is_active": strategy.get("active", True),
                        "created_at": strategy.get("created_at", datetime.now(timezone.utc)),
                        "updated_at": strategy.get("updated_at", datetime.now(timezone.utc)),
                    }
                    formatted_strategies.append(formatted_strategy)

                return formatted_strategies
            else:
                # Mock data for development - business logic in service
                mock_strategies = [
                    {
                        "strategy_name": "trend_following",
                        "strategy_type": "momentum",
                        "description": "Follows market trends using moving averages and momentum indicators",
                        "category": "trend",
                        "supported_exchanges": ["binance", "coinbase", "okx"],
                        "supported_symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                        "risk_level": "medium",
                        "minimum_capital": Decimal("1000.00"),
                        "maximum_capital": Decimal("50000.00"),
                        "default_parameters": {
                            "fast_ma": 12,
                            "slow_ma": 26,
                            "signal_line": 9,
                            "stop_loss": 0.02,
                            "take_profit": 0.04,
                        },
                        "performance_metrics": {
                            "sharpe_ratio": 1.25,
                            "max_drawdown": 0.15,
                            "win_rate": 0.62,
                            "avg_return": 0.08,
                        },
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc) - timedelta(days=30),
                        "updated_at": datetime.now(timezone.utc) - timedelta(days=5),
                    },
                    {
                        "strategy_name": "mean_reversion",
                        "strategy_type": "statistical_arbitrage",
                        "description": "Trades based on mean reversion using Bollinger Bands and RSI",
                        "category": "statistical",
                        "supported_exchanges": ["binance", "coinbase"],
                        "supported_symbols": ["BTCUSDT", "ETHUSDT"],
                        "risk_level": "low",
                        "minimum_capital": Decimal("500.00"),
                        "maximum_capital": Decimal("25000.00"),
                        "default_parameters": {
                            "bb_period": 20,
                            "bb_std": 2.0,
                            "rsi_period": 14,
                            "overbought": 70,
                            "oversold": 30,
                        },
                        "performance_metrics": {
                            "sharpe_ratio": 0.95,
                            "max_drawdown": 0.08,
                            "win_rate": 0.58,
                            "avg_return": 0.05,
                        },
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc) - timedelta(days=45),
                        "updated_at": datetime.now(timezone.utc) - timedelta(days=10),
                    },
                    {
                        "strategy_name": "breakout",
                        "strategy_type": "breakout",
                        "description": "Trades breakouts from consolidation patterns with volume confirmation",
                        "category": "breakout",
                        "supported_exchanges": ["binance", "okx"],
                        "supported_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                        "risk_level": "high",
                        "minimum_capital": Decimal("2000.00"),
                        "maximum_capital": Decimal("100000.00"),
                        "default_parameters": {
                            "lookback_period": 20,
                            "volume_multiplier": 1.5,
                            "breakout_threshold": 0.005,
                            "stop_loss": 0.03,
                        },
                        "performance_metrics": {
                            "sharpe_ratio": 1.45,
                            "max_drawdown": 0.22,
                            "win_rate": 0.48,
                            "avg_return": 0.12,
                        },
                        "is_active": False,
                        "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                        "updated_at": datetime.now(timezone.utc) - timedelta(days=15),
                    },
                ]
                return mock_strategies

        except Exception as e:
            self.logger.error(f"Error getting formatted strategies: {e}")
            raise ServiceError(f"Failed to get strategies: {e}")

    async def validate_strategy_parameters(
        self, strategy_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate strategy parameters with web-specific business logic."""
        try:
            validation_errors = []

            # Business logic: validate common parameters
            if "stop_loss" in parameters:
                stop_loss = parameters["stop_loss"]
                if isinstance(stop_loss, (int, float, Decimal)):
                    if Decimal(str(stop_loss)) <= 0 or Decimal(str(stop_loss)) > 1:
                        validation_errors.append("Stop loss must be between 0 and 1")

            if "take_profit" in parameters:
                take_profit = parameters["take_profit"]
                if isinstance(take_profit, (int, float, Decimal)):
                    if Decimal(str(take_profit)) <= 0:
                        validation_errors.append("Take profit must be greater than 0")

            # Strategy-specific validations
            if strategy_name == "trend_following":
                if "fast_ma" in parameters and "slow_ma" in parameters:
                    fast_ma = parameters["fast_ma"]
                    slow_ma = parameters["slow_ma"]
                    if isinstance(fast_ma, int) and isinstance(slow_ma, int):
                        if fast_ma >= slow_ma:
                            validation_errors.append("Fast MA must be less than slow MA")

            elif strategy_name == "mean_reversion":
                if "overbought" in parameters and "oversold" in parameters:
                    overbought = parameters["overbought"]
                    oversold = parameters["oversold"]
                    if isinstance(overbought, int) and isinstance(oversold, int):
                        if overbought <= oversold:
                            validation_errors.append(
                                "Overbought level must be greater than oversold"
                            )

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "validated_parameters": parameters if len(validation_errors) == 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Error validating strategy parameters: {e}")
            raise ServiceError(f"Failed to validate strategy parameters: {e}")

    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy performance data with web-specific metrics."""
        try:
            if self.strategy_facade:
                # Get performance from actual service
                raw_performance = await self.strategy_facade.get_strategy_performance(strategy_name)

                # Format for web interface
                formatted_performance = {
                    "strategy_name": strategy_name,
                    "total_return": raw_performance.get("total_return", 0.0),
                    "annualized_return": raw_performance.get("annualized_return", 0.0),
                    "sharpe_ratio": raw_performance.get("sharpe_ratio", 0.0),
                    "max_drawdown": raw_performance.get("max_drawdown", 0.0),
                    "win_rate": raw_performance.get("win_rate", 0.0),
                    "profit_factor": raw_performance.get("profit_factor", 1.0),
                    "total_trades": raw_performance.get("total_trades", 0),
                    "winning_trades": raw_performance.get("winning_trades", 0),
                    "losing_trades": raw_performance.get("losing_trades", 0),
                    "avg_win": raw_performance.get("avg_win", 0.0),
                    "avg_loss": raw_performance.get("avg_loss", 0.0),
                    "largest_win": raw_performance.get("largest_win", 0.0),
                    "largest_loss": raw_performance.get("largest_loss", 0.0),
                    "consecutive_wins": raw_performance.get("consecutive_wins", 0),
                    "consecutive_losses": raw_performance.get("consecutive_losses", 0),
                    "period_start": raw_performance.get("period_start"),
                    "period_end": raw_performance.get("period_end"),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
                return formatted_performance
            else:
                # Mock data for development based on strategy
                base_metrics = {
                    "trend_following": {
                        "total_return": 0.15,
                        "annualized_return": 0.12,
                        "sharpe_ratio": 1.25,
                        "max_drawdown": 0.15,
                        "win_rate": 0.62,
                        "profit_factor": 1.8,
                    },
                    "mean_reversion": {
                        "total_return": 0.08,
                        "annualized_return": 0.06,
                        "sharpe_ratio": 0.95,
                        "max_drawdown": 0.08,
                        "win_rate": 0.58,
                        "profit_factor": 1.4,
                    },
                    "breakout": {
                        "total_return": 0.22,
                        "annualized_return": 0.18,
                        "sharpe_ratio": 1.45,
                        "max_drawdown": 0.22,
                        "win_rate": 0.48,
                        "profit_factor": 2.1,
                    },
                }.get(
                    strategy_name,
                    {
                        "total_return": 0.10,
                        "annualized_return": 0.08,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": 0.12,
                        "win_rate": 0.55,
                        "profit_factor": 1.5,
                    },
                )

                # Calculate derived metrics
                total_trades = 150
                winning_trades = int(total_trades * base_metrics["win_rate"])
                losing_trades = total_trades - winning_trades

                return {
                    "strategy_name": strategy_name,
                    "total_return": base_metrics["total_return"],
                    "annualized_return": base_metrics["annualized_return"],
                    "sharpe_ratio": base_metrics["sharpe_ratio"],
                    "max_drawdown": base_metrics["max_drawdown"],
                    "win_rate": base_metrics["win_rate"],
                    "profit_factor": base_metrics["profit_factor"],
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "avg_win": 0.025,
                    "avg_loss": -0.015,
                    "largest_win": 0.08,
                    "largest_loss": -0.04,
                    "consecutive_wins": 8,
                    "consecutive_losses": 5,
                    "period_start": (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
                    "period_end": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error getting strategy performance for {strategy_name}: {e}")
            raise ServiceError(f"Failed to get strategy performance: {e}")

    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]:
        """Format backtest results for web display."""
        try:
            # Business logic: format backtest data for web interface
            formatted_results = {
                "backtest_id": backtest_data.get("backtest_id", "unknown"),
                "strategy_name": backtest_data.get("strategy_name", "unknown"),
                "start_date": backtest_data.get("start_date"),
                "end_date": backtest_data.get("end_date"),
                "initial_capital": backtest_data.get("initial_capital", Decimal("10000")),
                "final_capital": backtest_data.get("final_capital", Decimal("10000")),
                "total_return": backtest_data.get("total_return", 0.0),
                "annualized_return": backtest_data.get("annualized_return", 0.0),
                "max_drawdown": backtest_data.get("max_drawdown", 0.0),
                "sharpe_ratio": backtest_data.get("sharpe_ratio", 0.0),
                "sortino_ratio": backtest_data.get("sortino_ratio", 0.0),
                "calmar_ratio": backtest_data.get("calmar_ratio", 0.0),
                "total_trades": backtest_data.get("total_trades", 0),
                "winning_trades": backtest_data.get("winning_trades", 0),
                "losing_trades": backtest_data.get("losing_trades", 0),
                "win_rate": backtest_data.get("win_rate", 0.0),
                "profit_factor": backtest_data.get("profit_factor", 1.0),
                "avg_trade": backtest_data.get("avg_trade", 0.0),
                "best_trade": backtest_data.get("best_trade", 0.0),
                "worst_trade": backtest_data.get("worst_trade", 0.0),
                "equity_curve": backtest_data.get("equity_curve", []),
                "monthly_returns": backtest_data.get("monthly_returns", []),
                "trade_distribution": backtest_data.get("trade_distribution", {}),
                "risk_metrics": backtest_data.get("risk_metrics", {}),
                "completed_at": backtest_data.get(
                    "completed_at", datetime.now(timezone.utc).isoformat()
                ),
            }

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error formatting backtest results: {e}")
            raise ServiceError(f"Failed to format backtest results: {e}")

    async def health_check(self) -> "HealthCheckResult":
        """Perform health check and return status."""
        return {
            "service": "WebStrategyService",
            "status": "healthy",
            "strategy_facade_available": self.strategy_facade is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebStrategyService",
            "description": "Web strategy service handling strategy business logic",
            "capabilities": [
                "strategy_listing",
                "strategy_parameter_validation",
                "strategy_performance_data",
                "backtest_result_formatting",
            ],
            "version": "1.0.0",
        }

    async def get_strategy_config_through_service(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration through service layer (wraps facade call)."""
        try:
            if self.strategy_facade:
                return await self.strategy_facade.get_strategy_config(strategy_name)
            else:
                # Mock implementation for development
                return {
                    "name": strategy_name,
                    "type": "momentum",
                    "display_name": f"{strategy_name.title()} Strategy",
                    "description": f"Trading strategy: {strategy_name}",
                    "category": "technical",
                    "supported_exchanges": ["binance", "coinbase"],
                    "supported_symbols": ["BTCUSDT", "ETHUSDT"],
                    "risk_level": "medium",
                    "minimum_capital": 10000,
                    "recommended_timeframes": ["1h", "4h"],
                    "parameters": {"lookback_period": 20, "threshold": 0.02, "stop_loss": 0.05},
                    "performance_metrics": {
                        "win_rate": 0.65,
                        "profit_factor": 1.8,
                        "max_drawdown": 0.15,
                    },
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
        except Exception as e:
            self.logger.error(f"Error getting strategy config through service: {e}")
            raise ServiceError(f"Failed to get strategy config: {e}")

    async def validate_strategy_config_through_service(
        self, strategy_name: str, parameters: dict[str, Any]
    ) -> bool:
        """Validate strategy configuration through service layer (wraps facade call)."""
        try:
            if self.strategy_facade:
                return await self.strategy_facade.validate_strategy_config(
                    strategy_name, parameters
                )
            else:
                # Mock implementation for development - basic validation
                required_params = ["lookback_period", "threshold", "stop_loss"]
                for param in required_params:
                    if param not in parameters:
                        return False

                # Check parameter ranges
                if (
                    parameters.get("lookback_period", 0) < 1
                    or parameters.get("lookback_period", 0) > 200
                ):
                    return False
                if parameters.get("threshold", 0) < 0.001 or parameters.get("threshold", 0) > 0.1:
                    return False
                if parameters.get("stop_loss", 0) < 0.001 or parameters.get("stop_loss", 0) > 0.5:
                    return False

                return True
        except Exception as e:
            self.logger.error(f"Error validating strategy config through service: {e}")
            raise ServiceError(f"Failed to validate strategy config: {e}")
