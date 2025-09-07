"""
Service layer implementations for T-Bot Trading System.

This module implements the service interfaces defined in the API facade,
providing concrete implementations that interact with the core system.
"""

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.types import (
    BotConfiguration,
    MarketData,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)
from src.core.base.interfaces import (
    BotManagementServiceInterface,
    MarketDataServiceInterface,
    PortfolioServiceInterface,
    RiskServiceInterface,
    StrategyServiceInterface,
    TradingServiceInterface,
)

from src.core.exceptions import ServiceError, ValidationError


class TradingServiceImpl(TradingServiceInterface, BaseComponent):
    """Implementation of trading service."""

    def __init__(self, execution_engine=None):
        super().__init__()
        self.execution_engine = execution_engine

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.execution_engine is None and injector:
            try:
                self.execution_engine = injector.resolve("ExecutionEngine")
                self.logger.info("ExecutionEngine resolved from DI container")
            except Exception as e:
                # Execution engine is optional, log but continue
                self.logger.warning(f"ExecutionEngine not available in DI container: {e}")
                self.execution_engine = None

    async def initialize(self) -> None:
        """Initialize trading service."""
        self.logger.info("Trading service initialized")

    async def cleanup(self) -> None:
        """Cleanup trading service."""
        self.logger.info("Trading service cleaned up")

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Decimal | None = None,
    ) -> str:
        """Place a trading order."""
        try:
            if self.execution_engine:
                # Use actual execution engine
                order = await self.execution_engine.place_order(
                    symbol=symbol,
                    side=side.value,
                    order_type=order_type.value,
                    amount=amount,
                    price=price if price else None,
                )
                return order.get("order_id", "unknown")
            else:
                # Mock implementation
                order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
                self.logger.info(f"Mock order placed: {order_id}")
                return order_id
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if self.execution_engine:
                # Use actual execution engine
                result = await self.execution_engine.cancel_order(order_id)
                return result.get("success", False)
            else:
                # Mock implementation
                self.logger.info(f"Mock order cancelled: {order_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def get_positions(self) -> list[Position]:
        """Get current positions."""
        try:
            if self.execution_engine:
                # Use actual execution engine
                positions_data = await self.execution_engine.get_positions()
                positions = []
                for pos_data in positions_data:
                    position = Position(
                        symbol=pos_data.get("symbol", ""),
                        side=PositionSide.LONG,  # Default to LONG, should be determined from data
                        quantity=Decimal(str(pos_data.get("size", 0))),
                        entry_price=Decimal(str(pos_data.get("entry_price", 0))),
                        current_price=Decimal(str(pos_data.get("current_price", 0))),
                        unrealized_pnl=Decimal(str(pos_data.get("unrealized_pnl", 0))),
                        status=PositionStatus.OPEN,
                        opened_at=datetime.now(timezone.utc),
                        exchange="binance",  # Default exchange
                    )
                    positions.append(position)
                return positions
            else:
                # Mock implementation
                return [
                    Position(
                        symbol="BTC/USDT",
                        side=PositionSide.LONG,
                        quantity=Decimal("0.1"),
                        entry_price=Decimal("45000"),
                        current_price=Decimal("46000"),
                        unrealized_pnl=Decimal("100"),
                        status=PositionStatus.OPEN,
                        opened_at=datetime.now(timezone.utc),
                        exchange="binance",
                    )
                ]
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []


class BotManagementServiceImpl(BotManagementServiceInterface, BaseComponent):
    """Implementation of bot management service."""

    def __init__(self, bot_orchestrator=None):
        super().__init__()
        self.bot_orchestrator = bot_orchestrator

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.bot_orchestrator is None and injector:
            try:
                self.bot_orchestrator = injector.resolve("BotOrchestrator")
                self.logger.info("BotOrchestrator resolved from DI container")
            except Exception as e:
                # Bot orchestrator is optional, but log the issue
                self.logger.warning(f"BotOrchestrator not available in DI container: {e}")
                self.bot_orchestrator = None

    async def initialize(self) -> None:
        """Initialize bot management service."""
        self.logger.info("Bot management service initialized")

    async def cleanup(self) -> None:
        """Cleanup bot management service."""
        self.logger.info("Bot management service cleaned up")

    async def create_bot(self, config: BotConfiguration) -> str:
        """Create a new trading bot."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            bot_id = await self.bot_orchestrator.create_bot(config)
            return bot_id
        except Exception as e:
            self.logger.error(f"Error creating bot: {e}")
            raise

    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            success = await self.bot_orchestrator.start_bot(bot_id)
            return success
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            raise

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            success = await self.bot_orchestrator.stop_bot(bot_id)
            return success
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            raise

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator's public method
            if hasattr(self.bot_orchestrator, "get_bot_status"):
                status = await self.bot_orchestrator.get_bot_status(bot_id)
                return status
            else:
                # Fallback for compatibility - still throw error if method not available
                raise ServiceError("Bot orchestrator does not support get_bot_status method")
        except Exception as e:
            self.logger.error(f"Error getting bot status: {e}")
            raise

    async def list_bots(self) -> list[dict[str, Any]]:
        """List all bots."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            bot_list = await self.bot_orchestrator.get_bot_list()
            return bot_list
        except AttributeError:
            # Method not available, try alternative approach
            try:
                if hasattr(self.bot_orchestrator, "get_all_bots_status"):
                    all_status = await self.bot_orchestrator.get_all_bots_status()
                    # Convert to list format
                    bot_list = []
                    for bot_id, bot_data in all_status.get("bots", {}).items():
                        bot_list.append(
                            {
                                "bot_id": bot_id,
                                "bot_name": bot_data.get("state", {})
                                .get("configuration", {})
                                .get("bot_name", bot_id),
                                "status": bot_data.get("state", {}).get("status", "unknown"),
                                "allocated_capital": str(
                                    bot_data.get("state", {})
                                    .get("configuration", {})
                                    .get("allocated_capital", 0)
                                ),
                                "metrics": bot_data.get("metrics", {}),
                            }
                        )
                    return bot_list
                else:
                    raise ServiceError("Bot orchestrator does not support listing bots")
            except Exception as e:
                self.logger.error(f"Error using alternative list method: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Error listing bots: {e}")
            raise

    async def get_all_bots_status(self) -> dict[str, Any]:
        """Get status of all bots."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            if hasattr(self.bot_orchestrator, "get_all_bots_status"):
                return await self.bot_orchestrator.get_all_bots_status()
            else:
                # Fallback: construct from list_bots
                bots = await self.list_bots()
                bots_data = {}
                for bot in bots:
                    bot_id = bot.get("bot_id")
                    bots_data[bot_id] = {
                        "state": {
                            "status": bot.get("status", "unknown"),
                            "configuration": {
                                "bot_name": bot.get("bot_name"),
                                "allocated_capital": bot.get("allocated_capital"),
                            },
                        },
                        "metrics": bot.get("metrics", {}),
                        "uptime": None,
                    }
                return {"bots": bots_data}
        except Exception as e:
            self.logger.error(f"Error getting all bots status: {e}")
            raise

    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """Delete a bot."""
        if not self.bot_orchestrator:
            self.logger.error("Bot orchestrator not available")
            raise ServiceError(
                "Bot orchestrator service is not available. Please check system configuration."
            )

        try:
            # Use actual bot orchestrator
            if hasattr(self.bot_orchestrator, "delete_bot"):
                return await self.bot_orchestrator.delete_bot(bot_id, force=force)
            else:
                # Fallback: stop and remove bot
                if force or await self.stop_bot(bot_id):
                    # Actual removal requires bot_orchestrator delete_bot method implementation
                    self.logger.warning(
                        f"Bot deletion not fully implemented - bot {bot_id} stopped but not removed"
                    )
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error deleting bot: {e}")
            raise


class MarketDataServiceImpl(MarketDataServiceInterface, BaseComponent):
    """Implementation of market data service."""

    def __init__(self, data_service=None):
        super().__init__()
        self.data_service = data_service
        self.subscribers = {}
        self._cache = {}
        self._cache_ttl = 30  # 30 seconds cache TTL

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.data_service is None and injector:
            try:
                self.data_service = injector.resolve("DataService")
                self.logger.info("DataService resolved from DI container")
            except Exception as e:
                # Data service is optional, but log the issue
                self.logger.warning(f"DataService not available in DI container: {e}")
                self.data_service = None

    async def initialize(self) -> None:
        """Initialize market data service."""
        self.logger.info("Market data service initialized")

    async def cleanup(self) -> None:
        """Cleanup market data service."""
        self.logger.info("Market data service cleaned up")

    async def get_ticker(self, symbol: str) -> MarketData:
        """Get current ticker data with caching."""
        try:
            # Check cache first
            cache_key = f"ticker_{symbol}"
            current_time = datetime.now(timezone.utc)
            
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if (current_time - timestamp).total_seconds() < self._cache_ttl:
                    return cached_data
            
            if self.data_service:
                # Use actual data service - get recent data instead of get_ticker
                recent_data = await self.data_service.get_recent_data(symbol, limit=1)
                if recent_data:
                    # Return the most recent market data
                    latest = recent_data[0]
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=latest.timestamp,
                        open=latest.open,
                        high=latest.high,
                        low=latest.low,
                        close=latest.close,
                        volume=latest.volume,
                        exchange="binance",
                        bid_price=latest.close * Decimal("0.9995"),  # Approximate bid
                        ask_price=latest.close * Decimal("1.0005"),  # Approximate ask
                    )
                    # Cache the result
                    self._cache[cache_key] = (market_data, current_time)
                    return market_data

            # Mock implementation or fallback
            market_data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                open=Decimal("44800.00"),
                high=Decimal("45200.00"),
                low=Decimal("44500.00"),
                close=Decimal("45000.00"),
                volume=Decimal("1234567.89"),
                exchange="binance",
                bid_price=Decimal("44999.50"),
                ask_price=Decimal("45000.50"),
            )
            # Cache the mock result as well
            self._cache[cache_key] = (market_data, current_time)
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            raise

    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker updates."""
        try:
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(callback)
            self.logger.info(f"Subscribed to ticker updates for {symbol}")
        except Exception as e:
            self.logger.error(f"Error subscribing to ticker {symbol}: {e}")

    async def unsubscribe_from_ticker(self, symbol: str) -> None:
        """Unsubscribe from ticker updates."""
        try:
            if symbol in self.subscribers:
                del self.subscribers[symbol]
                self.logger.info(f"Unsubscribed from ticker updates for {symbol}")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from ticker {symbol}: {e}")


class PortfolioServiceImpl(PortfolioServiceInterface, BaseComponent):
    """Implementation of portfolio service."""

    def __init__(self, portfolio_manager=None):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.portfolio_manager is None and injector:
            try:
                self.portfolio_manager = injector.resolve("PortfolioManager")
                self.logger.info("PortfolioManager resolved from DI container")
            except Exception as e:
                # Portfolio manager is optional, but log the issue
                self.logger.warning(f"PortfolioManager not available in DI container: {e}")
                self.portfolio_manager = None

    async def initialize(self) -> None:
        """Initialize portfolio service."""
        self.logger.info("Portfolio service initialized")

    async def cleanup(self) -> None:
        """Cleanup portfolio service."""
        self.logger.info("Portfolio service cleaned up")

    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        try:
            if self.portfolio_manager:
                # Use actual portfolio manager
                balances = await self.portfolio_manager.get_balances()
                return {asset: Decimal(str(amount)) for asset, amount in balances.items()}
            else:
                # Mock implementation
                return {"USDT": Decimal("5000.00"), "BTC": Decimal("0.1"), "ETH": Decimal("2.5")}
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            return {}

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        try:
            if self.portfolio_manager:
                # Use actual portfolio manager
                summary = await self.portfolio_manager.get_summary()
                return summary
            else:
                # Mock implementation
                return {
                    "total_value": Decimal("10000.00"),
                    "available_balance": Decimal("5000.00"),
                    "unrealized_pnl": Decimal("123.45"),
                    "daily_pnl": Decimal("67.89"),
                    "daily_pnl_percent": Decimal("0.68"),
                    "positions": [
                        {
                            "symbol": "BTC/USDT",
                            "size": Decimal("0.1"),
                            "entry_price": Decimal("45000"),
                            "current_price": Decimal("46000"),
                            "pnl": Decimal("100.00"),
                        }
                    ],
                }
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}

    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Get P&L report for date range."""
        try:
            if self.portfolio_manager:
                # Use actual portfolio manager
                report = await self.portfolio_manager.get_pnl_report(start_date, end_date)
                return report
            else:
                # Mock implementation
                return {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_pnl": Decimal("567.89"),
                    "total_trades": 25,
                    "win_rate": Decimal("0.72"),
                    "max_drawdown": Decimal("-123.45"),
                    "sharpe_ratio": Decimal("1.25"),
                }
        except Exception as e:
            self.logger.error(f"Error getting P&L report: {e}")
            return {}


class RiskServiceImpl(RiskServiceInterface, BaseComponent):
    """Implementation of risk management service."""

    def __init__(self, risk_manager=None):
        super().__init__()
        self.risk_manager = risk_manager

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.risk_manager is None and injector:
            try:
                self.risk_manager = injector.resolve("RiskManager")
                self.logger.info("RiskManager resolved from DI container")
            except Exception as e:
                # Risk manager is optional, but log the issue
                self.logger.warning(f"RiskManager not available in DI container: {e}")
                self.risk_manager = None

    async def initialize(self) -> None:
        """Initialize risk management service."""
        self.logger.info("Risk management service initialized")

    async def cleanup(self) -> None:
        """Cleanup risk management service."""
        self.logger.info("Risk management service cleaned up")

    async def validate_order(
        self, symbol: str, side: OrderSide, amount: Decimal, price: Decimal | None = None
    ) -> dict[str, Any]:
        """Validate an order against risk rules."""
        try:
            if self.risk_manager:
                # Use actual risk manager
                validation_result = await self.risk_manager.validate_order(
                    symbol, side.value, amount, price if price else None
                )
                return validation_result
            else:
                # Mock implementation
                return {
                    "valid": True,
                    "risk_score": Decimal("0.25"),
                    "warnings": [],
                    "max_position_size": amount * Decimal("2"),
                    "suggested_stop_loss": price * Decimal("0.95") if price else None,
                }
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return {"valid": False, "error": str(e)}

    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        try:
            if self.risk_manager:
                # Use actual risk manager
                metrics = await self.risk_manager.get_risk_metrics()
                return metrics
            else:
                # Mock implementation
                return {
                    "portfolio_var": Decimal("1250.00"),
                    "max_drawdown": Decimal("0.15"),
                    "sharpe_ratio": Decimal("1.35"),
                    "volatility": Decimal("0.22"),
                    "risk_utilization": Decimal("0.65"),
                    "position_concentration": Decimal("0.45"),
                }
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}

    async def update_risk_limits(self, limits: dict[str, Any]) -> bool:
        """Update risk limits."""
        try:
            if self.risk_manager:
                # Use actual risk manager
                success = await self.risk_manager.update_risk_limits(limits)
                return success
            else:
                # Mock implementation
                self.logger.info(f"Mock risk limits updated: {limits}")
                return True
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")
            return False


class StrategyServiceImpl(StrategyServiceInterface, BaseComponent):
    """Implementation of strategy service."""

    def __init__(self, strategy_service=None, strategy_factory=None):
        super().__init__()
        self.strategy_service = (
            strategy_service  # Use proper StrategyService from strategies module
        )
        self.strategy_factory = strategy_factory  # Use StrategyFactory for strategy creation

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.strategy_service is None and injector:
            try:
                self.strategy_service = injector.resolve("StrategyService")
                self.logger.info("StrategyService resolved from DI container")
            except Exception as e:
                # Strategy service is optional, but log the issue
                self.logger.warning(f"StrategyService not available in DI container: {e}")
                self.strategy_service = None

        if self.strategy_factory is None and injector:
            try:
                self.strategy_factory = injector.resolve("StrategyFactory")
                self.logger.info("StrategyFactory resolved from DI container")
            except Exception as e:
                # Strategy factory is optional, but log the issue
                self.logger.warning(f"StrategyFactory not available in DI container: {e}")
                self.strategy_factory = None

    async def initialize(self) -> None:
        """Initialize strategy service."""
        self.logger.info("Strategy service initialized")

    async def cleanup(self) -> None:
        """Cleanup strategy service."""
        self.logger.info("Strategy service cleaned up")

    async def list_strategies(self) -> list[dict[str, Any]]:
        """List available strategies."""
        try:
            if self.strategy_factory:
                # Use actual strategy factory to get available strategies
                available_strategies = self.strategy_factory.list_available_strategies()
                return [
                    {
                        "name": strategy_type,
                        "display_name": info.get("class_name", strategy_type),
                        "description": f"Strategy type: {strategy_type}",
                        "parameters": info.get("required_parameters", []),
                        "risk_level": "medium",
                        "strategy_info": info,
                    }
                    for strategy_type, info in available_strategies.items()
                ]
            elif self.strategy_service:
                # Use strategy service to get registered strategies
                strategies = await self.strategy_service.get_all_strategies()
                return [
                    {
                        "name": strategy_id,
                        "display_name": strategy_data.get("strategy_id", strategy_id),
                        "description": f"Active strategy: {strategy_id}",
                        "parameters": [],
                        "risk_level": "medium",
                        "status": strategy_data.get("status", "unknown"),
                    }
                    for strategy_id, strategy_data in strategies.items()
                ]
            else:
                # Mock implementation
                return [
                    {
                        "name": "mean_reversion",
                        "display_name": "Mean Reversion Strategy",
                        "description": "Trades on price reversals to the mean",
                        "parameters": ["lookback_period", "threshold", "position_size"],
                        "risk_level": "medium",
                    },
                    {
                        "name": "momentum",
                        "display_name": "Momentum Strategy",
                        "description": "Follows price trends and momentum",
                        "parameters": ["fast_ma", "slow_ma", "momentum_threshold"],
                        "risk_level": "high",
                    },
                ]
        except Exception as e:
            self.logger.error(f"Error listing strategies: {e}")
            return []

    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration."""
        try:
            if self.strategy_service:
                # Use strategy service to get performance data for registered strategy
                performance = await self.strategy_service.get_strategy_performance(strategy_name)
                return performance.get("config", {})
            elif self.strategy_factory:
                # Get strategy info from factory
                from src.core.types import StrategyType

                try:
                    strategy_type = StrategyType(strategy_name)
                    return self.strategy_factory.get_strategy_info(strategy_type)
                except ValueError:
                    # Strategy name doesn't match enum, return default
                    pass
            # Mock implementation
            return {
                "name": strategy_name,
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02,
                    "position_size": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.10,
                },
                "constraints": {"max_position_size": 0.25, "max_daily_trades": 10},
            }
        except Exception as e:
            self.logger.error(f"Error getting strategy config: {e}")
            return {}

    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool:
        """Validate strategy configuration."""
        try:
            if self.strategy_factory:
                # Use factory validation
                from src.core.types import StrategyConfig, StrategyType

                try:
                    strategy_type = StrategyType(strategy_name)
                    strategy_config = StrategyConfig(**config)
                    return self.strategy_factory.validate_strategy_requirements(
                        strategy_type, strategy_config
                    )
                except (ValueError, ValidationError):
                    return False
            elif self.strategy_service:
                # Use service validation if strategy is registered
                from src.core.types import StrategyConfig

                try:
                    strategy_config = StrategyConfig(**config)
                    return await self.strategy_service.validate_strategy_config(strategy_config)
                except (ValueError, ValidationError):
                    return False
            else:
                # Mock implementation - basic validation
                required_params = ["lookback_period", "threshold", "position_size"]
                return all(param in config for param in required_params)
        except Exception as e:
            self.logger.error(f"Error validating strategy config: {e}")
            return False
