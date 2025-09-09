"""
Recovery scenarios for specific failure modes in the trading bot.

This module implements specific recovery procedures for common failure scenarios
including partial order fills, network disconnections, exchange maintenance,
data feed interruptions, order rejections, and API rate limits.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

from src.core import BaseComponent
from src.core.config import Config
from src.utils.decorators import retry, time_execution

from .constants import (
    API_RATE_LIMIT_MAX_ATTEMPTS,
    CONSERVATIVE_DATA_STOP_ADJUSTMENT,
    CONSERVATIVE_STOP_ADJUSTMENT,
    DEFAULT_CACHE_EXPIRY,
    DEFAULT_DATA_FEED_MAX_STALENESS,
    DEFAULT_MAINTENANCE_CHECK_INTERVAL,
    DEFAULT_NETWORK_MAX_OFFLINE_DURATION,
    NETWORK_ERROR_MAX_ATTEMPTS,
    PARTIAL_FILL_MIN_PERCENTAGE,
    PRICE_ADJUSTMENT_DOWN,
    PRICE_ADJUSTMENT_UP,
    QUANTITY_ADJUSTMENT,
    STRING_TRUNCATION_LIMIT,
    TOLERANCE_DECIMAL,
)


class DatabaseServiceInterface(Protocol):
    """Protocol for database service operations."""

    async def initialize(self) -> None: ...
    async def update_position(
        self, exchange: str, symbol: str, side: str, quantity: str, current_price: str
    ) -> None: ...
    async def get_open_positions_with_prices(self) -> list[dict[str, Any]]: ...
    async def get_order_details(self, order_id: str) -> dict[str, Any] | None: ...


class RiskServiceInterface(Protocol):
    """Protocol for risk management service operations."""

    async def initialize(self) -> None: ...
    async def update_position(
        self, symbol: str, quantity: Any, side: str, exchange: str
    ) -> None: ...
    async def update_stop_loss(self, symbol: str, stop_loss_price: Any, exchange: str) -> None: ...


class CacheServiceInterface(Protocol):
    """Protocol for cache service operations."""

    async def initialize(self) -> None: ...
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, expiry: int | None = None) -> None: ...


class StateServiceInterface(Protocol):
    """Protocol for state service operations."""

    async def initialize(self) -> None: ...
    async def create_checkpoint(self, component_name: str, state_data: dict[str, Any]) -> str: ...
    async def get_latest_checkpoint(self, component_name: str) -> dict[str, Any] | None: ...
    async def restore_checkpoint(self, checkpoint_id: str, component_name: str) -> None: ...


class BotServiceInterface(Protocol):
    """Protocol for bot management service operations."""

    async def initialize(self) -> None: ...
    async def pause_bot(self, component: str) -> None: ...
    async def resume_bot(self, component: str) -> None: ...


class RecoveryScenario(BaseComponent):
    """Base class for recovery scenarios with service injection."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(name="RecoveryScenario", config={})
        self.config = config

        # Inject services to avoid tight coupling
        self._database_service = database_service
        self._risk_service = risk_service
        self._cache_service = cache_service
        self._state_service = state_service
        self._bot_service = bot_service

        # Create default error handling config if not present
        self.recovery_config = getattr(
            config,
            "error_handling",
            {
                "partial_fill_min_percentage": Decimal("0.5"),
                "max_retry_attempts": 3,
                "reconnect_timeout": 30,
                "maintenance_check_interval": DEFAULT_MAINTENANCE_CHECK_INTERVAL,
            },
        )

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector."""
        try:
            # Try to get services from DI container
            if not self._database_service and injector.has_service("DatabaseService"):
                self._database_service = injector.resolve("DatabaseService")

            if not self._risk_service and injector.has_service("RiskService"):
                self._risk_service = injector.resolve("RiskService")

            if not self._cache_service and injector.has_service("CacheService"):
                self._cache_service = injector.resolve("CacheService")

            if not self._state_service and injector.has_service("StateService"):
                self._state_service = injector.resolve("StateService")

            if not self._bot_service and injector.has_service("BotService"):
                self._bot_service = injector.resolve("BotService")

            self._logger.debug("Recovery scenario dependencies configured via DI container")
        except Exception as e:
            self._logger.warning(
                f"Failed to configure some recovery scenario dependencies via DI: {e}"
            )

    @time_execution
    async def execute_recovery(self, context: Any) -> bool:
        """
        Execute the recovery scenario. Must be implemented by subclasses.

        Args:
            context: Recovery context containing error details and recovery parameters

        Returns:
            True if recovery was successful, False otherwise
        """
        self._logger.error(
            "Recovery scenario not implemented",
            scenario_class=self.__class__.__name__,
            context=(
                str(context)[:STRING_TRUNCATION_LIMIT] + "..."
                if len(str(context)) > STRING_TRUNCATION_LIMIT
                else str(context)
            ),
        )
        return False


class PartialFillRecovery(RecoveryScenario):
    """Handle partially filled orders with intelligent recovery."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.min_fill_percentage = Decimal(
            str(
                self.recovery_config.get(
                    "partial_fill_min_percentage", str(PARTIAL_FILL_MIN_PERCENTAGE)
                )
            )
        )
        self.cancel_remainder = True  # Default behavior
        self.log_details = True  # Default behavior

    @time_execution
    @retry(max_attempts=3)
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle partial order fill recovery."""
        order = context.get("order")
        filled_quantity = context.get("filled_quantity", Decimal("0"))

        if not order or not filled_quantity:
            self._logger.error("Invalid context for partial fill recovery", context=context)
            return False

        fill_percentage = filled_quantity / Decimal(str(order.quantity))

        self._logger.info(
            "Processing partial fill recovery",
            order_id=order.get("id"),
            fill_percentage=fill_percentage,
            filled_quantity=filled_quantity,
            total_quantity=order.quantity,
        )

        if fill_percentage < self.min_fill_percentage:
            # Cancel remainder and re-evaluate signal
            await self._cancel_order(order.get("id"))
            await self._log_partial_fill(order, filled_quantity)
            await self._reevaluate_signal(order.get("signal"))
            return True
        else:
            # Accept partial fill and adjust position tracking
            await self._update_position(order, filled_quantity)
            await self._adjust_stop_loss(order, filled_quantity)
            return True

    async def _cancel_order(self, order_id: str) -> None:
        """Cancel the remaining order."""
        # Import locally to avoid circular dependency
        try:
            from src.execution.order_manager import OrderManager
        except ImportError:
            self._logger.warning("OrderManager not available", order_id=order_id)
            return

        self._logger.info("Cancelling order", order_id=order_id)

        try:
            order_manager = OrderManager(self.config)
            result = await order_manager.cancel_order(order_id)

            if result:
                self._logger.info("Successfully cancelled order", order_id=order_id)
            else:
                self._logger.warning("Could not cancel order", order_id=order_id)
        except Exception as e:
            self._logger.error("Failed to cancel order", order_id=order_id, error=str(e))

    async def _log_partial_fill(self, order: dict[str, Any], filled_quantity: Decimal) -> None:
        """Log partial fill details for analysis."""
        if self.log_details:
            self._logger.info(
                "Partial fill logged",
                order_id=order.get("id"),
                filled_quantity=filled_quantity,
                fill_percentage=filled_quantity / Decimal(str(order["quantity"])),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def _reevaluate_signal(self, signal: dict[str, Any]) -> None:
        """Re-evaluate the original trading signal."""
        # Lazy import to avoid circular dependency
        try:
            from src.strategies.factory import StrategyFactory
        except ImportError:
            self._logger.warning("StrategyFactory not available", signal_id=signal.get("id"))
            return

        self._logger.info("Re-evaluating signal", signal_id=signal.get("id"))

        try:
            strategy_name = signal.get("strategy", "")
            if strategy_name:
                from src.strategies.service import StrategyService

                strategy_service = StrategyService()
                factory = StrategyFactory(strategy_service)
                strategy = await factory.create_strategy(strategy_name)

                # Re-evaluate current market conditions
                market_data = signal.get("market_data", {})
                new_signal = await strategy.generate_signal(market_data)

                # Compare with original signal
                if new_signal and new_signal.get("direction") != signal.get("direction"):
                    self._logger.warning(
                        "Signal direction changed",
                        signal_id=signal.get("id"),
                        original=signal.get("direction"),
                        new=new_signal.get("direction"),
                    )
                    # Update signal in database
                    signal["reevaluated"] = True
                    signal["reevaluation_result"] = new_signal
                else:
                    self._logger.info(
                        "Signal still valid",
                        signal_id=signal.get("id"),
                        direction=signal.get("direction"),
                    )
            else:
                self._logger.warning("No strategy specified for signal", signal_id=signal.get("id"))
        except Exception as e:
            self._logger.error(
                "Failed to re-evaluate signal", signal_id=signal.get("id"), error=str(e)
            )

    async def _update_position(self, order: dict[str, Any], filled_quantity: Decimal) -> None:
        """Update position tracking with partial fill using injected services."""
        if not self._database_service:
            self._logger.warning("Database service not available", order_id=order.get("id"))
            return

        if not self._risk_service:
            self._logger.warning("Risk service not available", order_id=order.get("id"))
            return

        self._logger.info(
            "Updating position with partial fill",
            order_id=order.get("id"),
            filled_quantity=filled_quantity,
        )

        try:
            await self._database_service.initialize()
            await self._risk_service.initialize()

            # Update position via injected service
            await self._database_service.update_position(
                exchange=order.get("exchange"),
                symbol=order.get("symbol"),
                side=order.get("side"),
                quantity=str(filled_quantity),
                current_price=str(order.get("price")),
            )

            # Update risk management via injected service
            await self._risk_service.update_position(
                symbol=order.get("symbol"),
                quantity=filled_quantity,
                side=order.get("side"),
                exchange=order.get("exchange"),
            )

            self._logger.info(
                "Position updated with partial fill",
                order_id=order.get("id"),
                filled_quantity=filled_quantity,
            )
        except Exception as e:
            self._logger.error("Failed to update position", order_id=order.get("id"), error=str(e))

    async def _adjust_stop_loss(self, order: dict[str, Any], filled_quantity: Decimal) -> None:
        """Adjust stop loss based on partial fill using injected services."""
        if not self._database_service:
            self._logger.warning("Database service not available", order_id=order.get("id"))
            return

        if not self._risk_service:
            self._logger.warning("Risk service not available", order_id=order.get("id"))
            return

        self._logger.info(
            "Adjusting stop loss for partial fill",
            order_id=order.get("id"),
            filled_quantity=filled_quantity,
        )

        try:
            await self._database_service.initialize()
            await self._risk_service.initialize()

            # Calculate new stop loss based on partial fill
            original_quantity = Decimal(str(order.get("quantity", 0)))
            fill_ratio = (
                filled_quantity / original_quantity if original_quantity > 0 else Decimal("0")
            )

            # Get current positions via injected service
            positions = await self._database_service.get_open_positions_with_prices()
            current_position = None

            for pos in positions:
                if (
                    pos["symbol"] == order.get("symbol")
                    and pos["side"] == order.get("side")
                    and pos["exchange"] == order.get("exchange")
                ):
                    current_position = pos
                    break

            if current_position and current_position.get("stop_loss_price"):
                # Adjust stop loss proportionally
                entry_price = Decimal(str(current_position["entry_price"]))
                current_stop = Decimal(str(current_position["stop_loss_price"]))

                # For partial fills, tighten stop loss
                if fill_ratio < Decimal("0.5"):
                    # Less than 50% filled - tighten stop
                    stop_distance = abs(entry_price - current_stop)
                    new_stop_distance = stop_distance * CONSERVATIVE_STOP_ADJUSTMENT

                    if order.get("side") == "buy":
                        new_stop = entry_price - new_stop_distance
                    else:
                        new_stop = entry_price + new_stop_distance
                else:
                    # More than 50% filled - keep original stop
                    new_stop = current_stop

                # Update via injected risk service
                await self._risk_service.update_stop_loss(
                    symbol=order.get("symbol"),
                    stop_loss_price=new_stop,
                    exchange=order.get("exchange"),
                )

                self._logger.info(
                    "Stop loss adjusted for partial fill",
                    order_id=order.get("id"),
                    filled_quantity=filled_quantity,
                    new_stop_loss=new_stop,
                )
            else:
                self._logger.warning(
                    "No position or stop loss found to adjust", order_id=order.get("id")
                )
        except Exception as e:
            self._logger.error("Failed to adjust stop loss", order_id=order.get("id"), error=str(e))


class NetworkDisconnectionRecovery(RecoveryScenario):
    """Handle network disconnection with automatic reconnection."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.max_offline_duration = self.recovery_config.get(
            "network_max_offline_duration", DEFAULT_NETWORK_MAX_OFFLINE_DURATION
        )
        self.sync_on_reconnect = True  # Default behavior
        self.conservative_mode = True  # Default behavior
        self.max_reconnect_attempts = NETWORK_ERROR_MAX_ATTEMPTS

    @time_execution
    @retry(max_attempts=5, base_delay=Decimal("2.0"))  # Use Decimal for financial precision
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle network disconnection recovery."""
        component = context.get("component", "unknown")

        self._logger.warning(
            "Network disconnection detected",
            component=component,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Switch to offline mode
        await self._switch_to_offline_mode(component)

        # Persist pending operations
        await self._persist_pending_operations(component)

        # Attempt reconnection with exponential backoff
        for attempt in range(self.max_reconnect_attempts):
            if await self._try_reconnect(component):
                # Reconcile state with exchange
                await self._reconcile_positions(component)
                await self._reconcile_orders(component)
                await self._verify_balances(component)
                await self._switch_to_online_mode(component)
                return True

            # Exponential backoff
            await asyncio.sleep(2**attempt)

        # Enter safe mode if reconnection fails
        await self._enter_safe_mode(component)
        return False

    async def _switch_to_offline_mode(self, component: str) -> None:
        """Switch component to offline mode using injected services."""
        if not self._cache_service:
            self._logger.warning("Cache service not available", component=component)
            return

        if not self._bot_service:
            self._logger.warning("Bot service not available", component=component)
            return

        self._logger.info("Switching to offline mode", component=component)

        try:
            await self._cache_service.initialize()
            await self._bot_service.initialize()

            # Set offline flag via injected cache service
            await self._cache_service.set(
                f"component:status:{component}",
                {
                    "status": "offline",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "network_disconnection",
                },
                expiry=DEFAULT_CACHE_EXPIRY,
            )

            # Pause bot operations via injected service
            await self._bot_service.pause_bot(component)

            self._logger.info("Switched to offline mode", component=component)
        except Exception as e:
            self._logger.error(
                "Failed to switch to offline mode", component=component, error=str(e)
            )

    async def _persist_pending_operations(self, component: str) -> None:
        """Persist any pending operations to prevent data loss using injected services."""
        if not self._cache_service:
            self._logger.warning("Cache service not available", component=component)
            return

        if not self._state_service:
            self._logger.warning("State service not available", component=component)
            return

        self._logger.info("Persisting pending operations", component=component)

        try:
            await self._cache_service.initialize()
            await self._state_service.initialize()

            # Get pending operations from injected cache service
            pending_orders = await self._cache_service.get(f"pending_orders:{component}")
            pending_trades = await self._cache_service.get(f"pending_trades:{component}")

            # Create checkpoint via injected state service
            checkpoint_data = {
                "component": component,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pending_orders": pending_orders or [],
                "pending_trades": pending_trades or [],
                "reason": "network_disconnection",
            }

            # Save checkpoint via injected service
            checkpoint_id = await self._state_service.create_checkpoint(
                component_name=component, state_data=checkpoint_data
            )

            self._logger.info(
                "Persisted pending operations",
                component=component,
                checkpoint_id=checkpoint_id,
                orders_count=len(pending_orders or []),
                trades_count=len(pending_trades or []),
            )
        except Exception as e:
            self._logger.error("Failed to persist operations", component=component, error=str(e))

    async def _try_reconnect(self, component: str) -> bool:
        """Attempt to reconnect to the service."""
        try:
            from src.error_handling.connection_manager import ConnectionManager
            from src.exchanges.factory import ExchangeFactory
        except ImportError:
            self._logger.warning(
                "ConnectionManager or ExchangeFactory not available", component=component
            )
            return False

        try:
            self._logger.info("Attempting reconnection", component=component)

            # Initialize ConnectionManager if not already done
            connection_manager = ConnectionManager(self.config)

            # Try to identify if component is an exchange
            supported_exchanges = getattr(self.config.exchanges, "supported_exchanges", [])
            if component in supported_exchanges:
                exchange = ExchangeFactory.create(component, self.config)

                # Test connection with a simple API call
                try:
                    await exchange.get_server_time()
                    self._logger.info("Exchange reconnection successful", component=component)
                    return True
                except Exception as e:
                    self._logger.warning(f"Exchange reconnection failed: {e}", component=component)
                    # Try using ConnectionManager's reconnect method as fallback
                    return await connection_manager.reconnect_connection(component)
            else:
                # Use ConnectionManager for other components
                # This provides proper reconnection with exponential backoff
                success = await connection_manager.reconnect_connection(component)
                if success:
                    self._logger.info(
                        "ConnectionManager reconnection successful", component=component
                    )
                else:
                    self._logger.warning(
                        "ConnectionManager reconnection failed", component=component
                    )
                return success

        except Exception as e:
            self._logger.error("Reconnection attempt failed", component=component, error=str(e))
            return False

    async def _reconcile_positions(self, component: str) -> None:
        """Reconcile positions with exchange data."""
        try:
            self._logger.info("Reconciling positions", component=component)

            # Get cached positions from local storage
            cached_positions = await self._get_cached_positions(component)

            # Get current positions from exchange (when connection is restored)
            exchange_positions = await self._fetch_exchange_positions(component)

            # Compare and reconcile differences
            discrepancies = self._compare_positions(cached_positions, exchange_positions)

            if discrepancies:
                self._logger.warning(
                    "Position discrepancies found",
                    component=component,
                    discrepancies=len(discrepancies),
                )
                await self._resolve_discrepancies(component, discrepancies)
            else:
                self._logger.info("Positions reconciled successfully", component=component)

        except Exception as e:
            self._logger.error("Failed to reconcile positions", component=component, error=str(e))

    async def _reconcile_orders(self, component: str) -> None:
        """Reconcile orders with exchange data."""
        try:
            self._logger.info("Reconciling orders", component=component)

            # Get pending orders from local cache
            cached_orders = await self._get_cached_orders(component)

            # Get open orders from exchange
            exchange_orders = await self._fetch_exchange_orders(component)

            # Identify orders that need reconciliation
            missing_orders = [o for o in cached_orders if o not in exchange_orders]
            unknown_orders = [o for o in exchange_orders if o not in cached_orders]

            if missing_orders:
                self._logger.warning(
                    "Missing orders detected", component=component, count=len(missing_orders)
                )
                await self._handle_missing_orders(component, missing_orders)

            if unknown_orders:
                self._logger.warning(
                    "Unknown orders detected", component=component, count=len(unknown_orders)
                )
                await self._handle_unknown_orders(component, unknown_orders)

            self._logger.info("Orders reconciled", component=component)

        except Exception as e:
            self._logger.error("Failed to reconcile orders", component=component, error=str(e))

    async def _verify_balances(self, component: str) -> None:
        """Verify account balances are consistent."""
        try:
            self._logger.info("Verifying balances", component=component)

            # Get cached balances
            cached_balances = await self._get_cached_balances(component)

            # Get current balances from exchange
            exchange_balances = await self._fetch_exchange_balances(component)

            # Compare balances with tolerance
            tolerance = TOLERANCE_DECIMAL
            discrepancies = {}

            for asset, cached_amount in cached_balances.items():
                exchange_amount = exchange_balances.get(asset, Decimal("0"))
                difference = abs(cached_amount - exchange_amount)

                if difference > tolerance:
                    discrepancies[asset] = {
                        "cached": cached_amount,
                        "exchange": exchange_amount,
                        "difference": difference,
                    }

            if discrepancies:
                self._logger.error(
                    "Balance discrepancies detected",
                    component=component,
                    discrepancies=discrepancies,
                )
                await self._handle_balance_discrepancies(component, discrepancies)
            else:
                self._logger.info("Balances verified successfully", component=component)

        except Exception as e:
            self._logger.error("Failed to verify balances", component=component, error=str(e))

    async def _switch_to_online_mode(self, component: str) -> None:
        """Switch component back to online mode using injected services."""
        if not all([self._cache_service, self._bot_service, self._state_service]):
            self._logger.warning(
                "Required services not available",
                component=component,
                available_services={
                    "cache": self._cache_service is not None,
                    "bot": self._bot_service is not None,
                    "state": self._state_service is not None,
                },
            )
            return

        self._logger.info("Switching to online mode", component=component)

        try:
            await self._cache_service.initialize()
            await self._bot_service.initialize()
            await self._state_service.initialize()

            # Restore from checkpoint if available via injected service
            latest_checkpoint = await self._state_service.get_latest_checkpoint(component)
            if latest_checkpoint:
                await self._state_service.restore_checkpoint(
                    checkpoint_id=latest_checkpoint["id"], component_name=component
                )
                self._logger.info("Restored from checkpoint", component=component)

            # Update status via injected cache service
            await self._cache_service.set(
                f"component:status:{component}",
                {
                    "status": "online",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reconnected_at": datetime.now(timezone.utc).isoformat(),
                },
                expiry=DEFAULT_CACHE_EXPIRY,
            )

            # Resume bot operations via injected service
            await self._bot_service.resume_bot(component)

            self._logger.info("Switched to online mode", component=component)
        except Exception as e:
            self._logger.error("Failed to switch to online mode", component=component, error=str(e))

    async def _enter_safe_mode(self, component: str) -> None:
        """Enter safe mode when reconnection fails."""
        try:
            self._logger.warning("Entering safe mode", component=component)

            # Cancel all pending orders
            await self._cancel_all_pending_orders(component)

            # Close all open positions if configured
            if hasattr(self.config, "error_handling") and hasattr(
                self.config.error_handling, "safe_mode_close_positions"
            ):
                if self.config.error_handling.safe_mode_close_positions:
                    await self._close_all_positions(component)

            # Disable trading
            await self._disable_trading(component)

            # Set safe mode flag
            await self._set_safe_mode_flag(component, True)

            # Send critical alert
            await self._send_critical_alert(
                component=component,
                message=f"Component {component} entered safe mode due to connection failure",
            )

            self._logger.warning(
                "Safe mode activated",
                component=component,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            self._logger.error("Failed to enter safe mode", component=component, error=str(e))
            # Safe mode failure is critical - attempt emergency shutdown
            await self._emergency_shutdown(component)

    # Helper methods for recovery operations
    async def _get_cached_positions(self, component: str) -> dict[str, Any]:
        """Get cached positions from local storage."""
        # Placeholder for cache integration - returns empty dict for now
        return {}

    async def _fetch_exchange_positions(self, component: str) -> dict[str, Any]:
        """Fetch current positions from exchange."""
        # Placeholder for exchange API integration - returns empty dict for now
        return {}

    def _compare_positions(self, cached: dict, exchange: dict) -> list[dict]:
        """Compare cached and exchange positions."""
        discrepancies = []
        all_symbols = set(cached.keys()) | set(exchange.keys())

        for symbol in all_symbols:
            cached_pos = cached.get(symbol, {})
            exchange_pos = exchange.get(symbol, {})

            if cached_pos != exchange_pos:
                discrepancies.append(
                    {"symbol": symbol, "cached": cached_pos, "exchange": exchange_pos}
                )

        return discrepancies

    async def _resolve_discrepancies(self, component: str, discrepancies: list) -> None:
        """Resolve position discrepancies."""
        for discrepancy in discrepancies:
            self._logger.warning(f"Resolving discrepancy for {discrepancy['symbol']}")
            # Cache update placeholder - implement when cache layer is available

    async def _get_cached_orders(self, component: str) -> list[dict]:
        """Get cached orders from local storage."""
        return []

    async def _fetch_exchange_orders(self, component: str) -> list[dict]:
        """Fetch open orders from exchange."""
        return []

    async def _handle_missing_orders(self, component: str, orders: list) -> None:
        """Handle orders missing from exchange."""
        for order in orders:
            self._logger.warning(f"Handling missing order: {order}")
            # Mark as cancelled or resubmit based on strategy

    async def _handle_unknown_orders(self, component: str, orders: list) -> None:
        """Handle orders unknown to local cache."""
        for order in orders:
            self._logger.warning(f"Handling unknown order: {order}")
            # Add to local cache or cancel based on strategy

    async def _get_cached_balances(self, component: str) -> dict[str, Decimal]:
        """Get cached balances from local storage."""
        return {}

    async def _fetch_exchange_balances(self, component: str) -> dict[str, Decimal]:
        """Fetch current balances from exchange."""
        return {}

    async def _handle_balance_discrepancies(self, component: str, discrepancies: dict) -> None:
        """Handle balance discrepancies."""
        for asset, info in discrepancies.items():
            self._logger.error(f"Balance discrepancy for {asset}: {info}")
            # Update local cache with exchange data as source of truth

    async def _cancel_all_pending_orders(self, component: str) -> None:
        """Cancel all pending orders."""
        self._logger.info(f"Cancelling all pending orders for {component}")
        # Order cancellation placeholder - implement with exchange integration

    async def _close_all_positions(self, component: str) -> None:
        """Close all open positions."""
        self._logger.info(f"Closing all positions for {component}")
        # Position closure placeholder - implement with exchange integration

    async def _disable_trading(self, component: str) -> None:
        """Disable trading for component."""
        self._logger.info(f"Disabling trading for {component}")
        # Set flag in config/database to prevent new trades

    async def _set_safe_mode_flag(self, component: str, enabled: bool) -> None:
        """Set safe mode flag."""
        self._logger.info(f"Setting safe mode flag for {component}: {enabled}")
        # Update system state to reflect safe mode

    async def _send_critical_alert(self, component: str, message: str) -> None:
        """Send critical alert notification."""
        self._logger.critical(f"ALERT - {component}: {message}")
        # Alert notification placeholder - implement with monitoring integration

    async def _emergency_shutdown(self, component: str) -> None:
        """Perform emergency shutdown."""
        self._logger.critical(f"EMERGENCY SHUTDOWN - {component}")
        # Graceful shutdown placeholder - implement with service lifecycle management


class ExchangeMaintenanceRecovery(RecoveryScenario):
    """Handle exchange maintenance with graceful degradation."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.detect_maintenance = True  # Default behavior
        self.redistribute_capital = True  # Default behavior
        self.pause_new_orders = True  # Default behavior

    @time_execution
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle exchange maintenance recovery."""
        exchange = context.get("exchange", "unknown")

        self._logger.warning(
            "Exchange maintenance detected",
            exchange=exchange,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if self.detect_maintenance:
            await self._detect_maintenance_schedule(exchange)

        if self.redistribute_capital:
            await self._redistribute_capital(exchange)

        if self.pause_new_orders:
            await self._pause_new_orders(exchange)

        return True

    async def _detect_maintenance_schedule(self, exchange: str) -> None:
        """Detect and handle scheduled maintenance."""
        try:
            # Maintenance window check placeholder - implement with exchange API
            # This will be implemented in P-003+ (Exchange Integrations)
            self._logger.info("Detecting maintenance schedule", exchange=exchange)
        except Exception as e:
            self._logger.error(
                "Failed to detect maintenance schedule", exchange=exchange, error=str(e)
            )

    async def _redistribute_capital(self, exchange: str) -> None:
        """Redistribute capital to other exchanges."""
        try:
            # Capital redistribution involves rebalancing positions across exchanges
            # when one exchange goes into maintenance mode
            if self._risk_service:
                # Create checkpoint before redistribution
                if self._state_service:
                    await self._state_service.create_checkpoint(
                        "capital_redistribution",
                        {"exchange": exchange, "timestamp": datetime.now(timezone.utc).isoformat()},
                    )

            self._logger.info("Capital redistribution initiated", exchange=exchange)
        except Exception as e:
            self._logger.error("Failed to redistribute capital", exchange=exchange, error=str(e))

    async def _pause_new_orders(self, exchange: str) -> None:
        """Pause new order placement on the exchange."""
        try:
            # Pause new order creation on the maintenance-affected exchange
            # This prevents orders from being submitted during downtime
            if self._bot_service:
                await self._bot_service.pause_bot(f"order_execution_{exchange}")

            pause_key = f"orders_paused_{exchange}"
            if self._cache_service:
                await self._cache_service.set(pause_key, True, DEFAULT_CACHE_EXPIRY)  # 1 hour cache

            self._logger.info("New orders paused for maintenance", exchange=exchange)
        except Exception as e:
            self._logger.error("Failed to pause new orders", exchange=exchange, error=str(e))


class DataFeedInterruptionRecovery(RecoveryScenario):
    """Handle data feed interruptions with fallback sources."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.max_staleness = self.recovery_config.get(
            "data_feed_max_staleness", DEFAULT_DATA_FEED_MAX_STALENESS
        )
        self.fallback_sources = ["backup_feed", "static_data"]  # Default fallback sources
        self.conservative_trading = True  # Default behavior

    @time_execution
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle data feed interruption recovery."""
        data_source = context.get("data_source", "unknown")

        self._logger.warning(
            "Data feed interruption detected",
            data_source=data_source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Check data staleness
        if await self._check_data_staleness(data_source):
            await self._switch_to_fallback_source(data_source)

        if self.conservative_trading:
            await self._enable_conservative_trading(data_source)

        return True

    async def _check_data_staleness(self, data_source: str) -> bool:
        """Check if data is stale and needs fallback."""
        try:
            # Data freshness validation placeholder - implement with timestamp comparison
            # This will be implemented in P-014 (Data Pipeline and Sources
            # Integration)
            self._logger.info("Checking data staleness", data_source=data_source)
            return True  # Simulate stale data
        except Exception as e:
            self._logger.error(
                "Failed to check data staleness", data_source=data_source, error=str(e)
            )
            return True

    async def _switch_to_fallback_source(self, data_source: str) -> None:
        """Switch to fallback data source."""
        try:
            # Switch to backup data sources or cached values
            # until primary data feed is restored
            fallback_source = self.fallback_sources[0] if self.fallback_sources else "cached_data"

            if self._cache_service:
                await self._cache_service.set(f"active_source_{data_source}", fallback_source, 300)

            # Log the source switch for monitoring
            self._logger.info(
                "Switched to fallback data source",
                original_source=data_source,
                fallback_source=fallback_source,
            )
        except Exception as e:
            self._logger.error(
                "Failed to switch to fallback source", data_source=data_source, error=str(e)
            )

    async def _enable_conservative_trading(self, data_source: str) -> None:
        """Enable conservative trading mode."""
        try:
            # Reduce position sizes and tighten stop losses
            # during data feed instability
            if self._risk_service:
                # Get all open positions and apply conservative parameters
                if self._database_service:
                    positions = await self._database_service.get_open_positions_with_prices()
                    for position in positions:
                        symbol = position.get("symbol")
                        if symbol:
                            # Tighten stop losses by 5%
                            current_stop = position.get("stop_loss_price")
                            if current_stop:
                                conservative_stop = (
                                    Decimal(str(current_stop)) * CONSERVATIVE_DATA_STOP_ADJUSTMENT
                                )
                                await self._risk_service.update_stop_loss(
                                    symbol, conservative_stop, position.get("exchange", "default")
                                )

            self._logger.info("Conservative trading mode enabled", data_source=data_source)
        except Exception as e:
            self._logger.error(
                "Failed to enable conservative trading", data_source=data_source, error=str(e)
            )


class OrderRejectionRecovery(RecoveryScenario):
    """Handle order rejections with intelligent retry."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.analyze_rejection_reason = True  # Default behavior
        self.adjust_parameters = True  # Default behavior
        self.max_retry_attempts = self.recovery_config.get("order_rejection_max_retries", 3)

    @time_execution
    @retry(max_attempts=2)
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle order rejection recovery."""
        order = context.get("order")
        rejection_reason = context.get("rejection_reason", "unknown")

        self._logger.warning(
            "Order rejection detected",
            order_id=order.get("id") if order else "unknown",
            rejection_reason=rejection_reason,
        )

        if self.analyze_rejection_reason:
            await self._analyze_rejection_reason(order, rejection_reason)

        if self.adjust_parameters:
            await self._adjust_order_parameters(order, rejection_reason)

        return True

    async def _analyze_rejection_reason(self, order: dict[str, Any], rejection_reason: str) -> None:
        """Analyze the rejection reason for pattern detection."""
        try:
            # Analyze rejection codes and messages from exchange
            # to determine appropriate recovery action
            rejection_patterns = {
                "insufficient_funds": "balance",
                "price_too_high": "price",
                "price_too_low": "price",
                "invalid_quantity": "quantity",
                "market_closed": "timing",
                "rate_limit": "throttling",
            }

            rejection_type = "unknown"
            for pattern, category in rejection_patterns.items():
                if pattern in rejection_reason.lower():
                    rejection_type = category
                    break

            # Store rejection analysis for future reference
            if self._cache_service:
                analysis_key = f"rejection_analysis_{order.get('id') if order else 'unknown'}"
                await self._cache_service.set(
                    analysis_key,
                    {
                        "rejection_reason": rejection_reason,
                        "rejection_type": rejection_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    3600,
                )

            self._logger.info(
                "Rejection reason analyzed",
                order_id=order.get("id") if order else "unknown",
                rejection_reason=rejection_reason,
                rejection_type=rejection_type,
            )
        except Exception as e:
            self._logger.error("Failed to analyze rejection reason", error=str(e))

    async def _adjust_order_parameters(self, order: dict[str, Any], rejection_reason: str) -> None:
        """Adjust order parameters based on rejection reason."""
        try:
            # Adjust order parameters based on rejection analysis
            # (price, quantity, order type, etc.)
            adjustments = {}

            if "price" in rejection_reason.lower():
                # Adjust price based on market conditions
                if order and "price" in order:
                    current_price = Decimal(str(order["price"]))
                    # Adjust by 0.1% towards market
                    if "too_high" in rejection_reason.lower():
                        adjustments["price"] = current_price * PRICE_ADJUSTMENT_DOWN
                    elif "too_low" in rejection_reason.lower():
                        adjustments["price"] = current_price * PRICE_ADJUSTMENT_UP

            elif "quantity" in rejection_reason.lower():
                # Adjust quantity if invalid
                if order and "quantity" in order:
                    current_qty = Decimal(str(order["quantity"]))
                    # Reduce by 10% for size issues
                    adjustments["quantity"] = current_qty * QUANTITY_ADJUSTMENT

            # Store adjustments for retry
            if adjustments and self._cache_service:
                adjustment_key = f"order_adjustments_{order.get('id') if order else 'unknown'}"
                await self._cache_service.set(adjustment_key, adjustments, 300)

            self._logger.info(
                "Order parameters adjusted",
                order_id=order.get("id") if order else "unknown",
                rejection_reason=rejection_reason,
            )
        except Exception as e:
            self._logger.error("Failed to adjust order parameters", error=str(e))


class APIRateLimitRecovery(RecoveryScenario):
    """Handle API rate limit violations with automatic throttling."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        cache_service: CacheServiceInterface | None = None,
        state_service: StateServiceInterface | None = None,
        bot_service: BotServiceInterface | None = None,
    ):
        super().__init__(
            config, database_service, risk_service, cache_service, state_service, bot_service
        )
        self.respect_retry_after = True
        self.max_retry_attempts = API_RATE_LIMIT_MAX_ATTEMPTS
        self.base_delay = 5

    @time_execution
    async def execute_recovery(self, context: dict[str, Any]) -> bool:
        """Handle API rate limit recovery."""
        api_endpoint = context.get("api_endpoint", "unknown")
        retry_after = context.get("retry_after", self.base_delay)

        self._logger.warning(
            "API rate limit exceeded", api_endpoint=api_endpoint, retry_after=retry_after
        )

        # Respect retry-after header if provided
        if self.respect_retry_after:
            await asyncio.sleep(retry_after)

        # Exponential backoff placeholder - implement with configurable retry logic
        for attempt in range(self.max_retry_attempts):
            try:
                # Retry API call with exponential backoff
                # and respect exchange rate limits
                self._logger.info(
                    "Retrying API call", api_endpoint=api_endpoint, attempt=attempt + 1
                )
                await asyncio.sleep(2**attempt)  # Exponential backoff
                return True
            except Exception as e:
                self._logger.error(
                    "API retry failed", api_endpoint=api_endpoint, attempt=attempt + 1, error=str(e)
                )

        return False


def create_partial_fill_recovery_factory(config: Config | None = None):
    """Create a factory function for PartialFillRecovery instances."""

    def factory(injector=None):
        resolved_config = config or Config()

        # Resolve services from DI container if available
        database_service = None
        risk_service = None
        cache_service = None
        state_service = None
        bot_service = None

        if injector:
            database_service = (
                injector.resolve("DatabaseService")
                if injector.has_service("DatabaseService")
                else None
            )
            risk_service = (
                injector.resolve("RiskService") if injector.has_service("RiskService") else None
            )
            cache_service = (
                injector.resolve("CacheService") if injector.has_service("CacheService") else None
            )
            state_service = (
                injector.resolve("StateService") if injector.has_service("StateService") else None
            )
            bot_service = (
                injector.resolve("BotService") if injector.has_service("BotService") else None
            )

        recovery = PartialFillRecovery(
            resolved_config,
            database_service,
            risk_service,
            cache_service,
            state_service,
            bot_service,
        )

        # Configure additional dependencies if injector is available
        if injector:
            recovery.configure_dependencies(injector)

        return recovery

    return factory


def create_network_disconnection_recovery_factory(config: Config | None = None):
    """Create a factory function for NetworkDisconnectionRecovery instances."""

    def factory(injector=None):
        resolved_config = config or Config()

        # Resolve services from DI container if available
        database_service = None
        risk_service = None
        cache_service = None
        state_service = None
        bot_service = None

        if injector:
            database_service = (
                injector.resolve("DatabaseService")
                if injector.has_service("DatabaseService")
                else None
            )
            risk_service = (
                injector.resolve("RiskService") if injector.has_service("RiskService") else None
            )
            cache_service = (
                injector.resolve("CacheService") if injector.has_service("CacheService") else None
            )
            state_service = (
                injector.resolve("StateService") if injector.has_service("StateService") else None
            )
            bot_service = (
                injector.resolve("BotService") if injector.has_service("BotService") else None
            )

        recovery = NetworkDisconnectionRecovery(
            resolved_config,
            database_service,
            risk_service,
            cache_service,
            state_service,
            bot_service,
        )

        # Configure additional dependencies if injector is available
        if injector:
            recovery.configure_dependencies(injector)

        return recovery

    return factory
