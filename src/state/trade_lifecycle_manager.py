"""
Trade Lifecycle Management for the T-Bot trading system (P-023).

This module manages the complete lifecycle of trades from signal generation
through execution, settlement, and performance attribution. It provides:

- Trade state transitions and workflow management
- Trade history tracking and audit trails
- Performance metrics and attribution analysis
- Trade quality scoring and validation
- Position lifecycle coordination

The TradeLifecycleManager ensures complete visibility into trade execution
and provides the foundation for performance analysis and optimization.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.base.component import BaseComponent
from src.core.caching import CacheKeys, cache_invalidate, cached, get_cache_manager
from src.core.config.main import Config
from src.core.exceptions import StateConsistencyError
from src.core.types import ExecutionResult, OrderRequest, OrderSide, OrderType

# Service layer imports only - no direct database access
from .services import StatePersistenceServiceProtocol
from .services.trade_lifecycle_service import (
    TradeContext,
    TradeHistoryRecord,
    TradeLifecycleServiceProtocol,
    TradeLifecycleState,
)
from .utils_imports import time_execution


class TradeEvent(str, Enum):
    """Trade event enumeration."""

    SIGNAL_RECEIVED = "signal_received"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"  # Added to match interface
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"  # Added to match interface
    SETTLEMENT_COMPLETE = "settlement_complete"
    ATTRIBUTION_COMPLETE = "attribution_complete"


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis for trades."""

    trade_id: str = ""
    bot_id: str = ""
    strategy_name: str = ""

    # Attribution components
    strategy_alpha: Decimal = Decimal("0")  # Strategy-specific return
    market_beta: Decimal = Decimal("0")  # Market-related return
    execution_alpha: Decimal = Decimal("0")  # Execution efficiency
    fees_impact: Decimal = Decimal("0")  # Fee impact
    slippage_impact: Decimal = Decimal("0")  # Slippage impact

    # Risk metrics
    var_contribution: Decimal = Decimal("0")
    volatility_contribution: Decimal = Decimal("0")

    # Quality factors
    timing_quality: Decimal = Decimal("0.0")
    execution_quality: Decimal = Decimal("0.0")
    strategy_quality: Decimal = Decimal("0.0")

    # Period
    attribution_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradeLifecycleManager(BaseComponent):
    """
    Comprehensive trade lifecycle management system.

    Features:
    - Complete trade state machine with all lifecycle phases
    - Real-time trade tracking and monitoring
    - Performance attribution and quality scoring
    - Historical analysis and reporting
    - Integration with execution and risk systems
    """

    def __init__(
        self,
        config: Config,
        persistence_service: StatePersistenceServiceProtocol,
        lifecycle_service: TradeLifecycleServiceProtocol,
        cache_service: Any | None = None,
    ):
        """
        Initialize the trade lifecycle manager.

        Args:
            config: Application configuration
            persistence_service: Service layer for persistence operations (required)
            lifecycle_service: Service layer for trade lifecycle business logic (required)
            cache_service: Cache service for performance optimization
        """
        super().__init__(
            name="TradeLifecycleManager",
            config=config.__dict__ if hasattr(config, "__dict__") else {},
        )
        self.config = config

        # Service layer components (required - no fallback to direct DB access)
        self._persistence_service = persistence_service
        self._lifecycle_service = lifecycle_service

        # Optional services
        self.cache_service = cache_service

        # Active trades tracking
        self.active_trades: dict[str, TradeContext] = {}
        self.trade_history: list[TradeHistoryRecord] = []

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "average_execution_time": Decimal("0.0"),
            "average_quality_score": Decimal("0.0"),
            "total_pnl": Decimal("0"),
            "total_fees": Decimal("0"),
        }

        # State machine configuration
        self.valid_transitions = {
            TradeLifecycleState.SIGNAL_GENERATED: [TradeLifecycleState.PRE_TRADE_VALIDATION],
            TradeLifecycleState.PRE_TRADE_VALIDATION: [
                TradeLifecycleState.ORDER_CREATED,
                TradeLifecycleState.REJECTED,
            ],
            TradeLifecycleState.ORDER_CREATED: [
                TradeLifecycleState.ORDER_SUBMITTED,
                TradeLifecycleState.CANCELLED,
            ],
            TradeLifecycleState.ORDER_SUBMITTED: [
                TradeLifecycleState.PARTIALLY_FILLED,
                TradeLifecycleState.FULLY_FILLED,
                TradeLifecycleState.CANCELLED,
                TradeLifecycleState.REJECTED,
            ],
            TradeLifecycleState.PARTIALLY_FILLED: [
                TradeLifecycleState.FULLY_FILLED,
                TradeLifecycleState.CANCELLED,
            ],
            TradeLifecycleState.FULLY_FILLED: [TradeLifecycleState.SETTLED],
            TradeLifecycleState.SETTLED: [TradeLifecycleState.ATTRIBUTED],
            # Terminal states
            TradeLifecycleState.CANCELLED: [],
            TradeLifecycleState.REJECTED: [],
            TradeLifecycleState.ATTRIBUTED: [],
        }

        self.logger.info("TradeLifecycleManager initialized")

    async def initialize(self) -> None:
        """Initialize the trade lifecycle manager with service dependencies."""
        try:
            # Initialize persistence service (required)
            if hasattr(self._persistence_service, "start"):
                await self._persistence_service.start()
            self.logger.info("Persistence service initialized")

            # Initialize cache service if available
            if not self.cache_service:
                # Fall back to cache manager if no cache service provided
                try:
                    self.cache_service = get_cache_manager(config=self.config)
                    self.logger.info("Cache manager initialized as fallback")
                except Exception as e:
                    self.logger.warning(
                        f"Cache initialization failed: {e} - operating without cache"
                    )
                    self.cache_service = None

            # Load active trades from persistence layer
            await self._load_active_trades()

            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("TradeLifecycleManager initialization completed")

        except Exception as e:
            self.logger.error(f"TradeLifecycleManager initialization failed: {e}")
            raise StateConsistencyError(f"Failed to initialize TradeLifecycleManager: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup trade lifecycle manager resources."""
        try:
            self.logger.info("Starting TradeLifecycleManager cleanup")

            # Cancel monitoring task if running
            if (
                hasattr(self, "_monitoring_task")
                and self._monitoring_task
                and not self._monitoring_task.done()
            ):
                self._monitoring_task.cancel()
                try:
                    await asyncio.wait_for(self._monitoring_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    self.logger.warning(f"Monitoring task cleanup error: {e}")

            # Clear task reference
            if hasattr(self, "_monitoring_task"):
                self._monitoring_task = None

            # Cleanup persistence service
            if self._persistence_service and hasattr(self._persistence_service, "stop"):
                try:
                    await self._persistence_service.stop()
                    self.logger.info("Persistence service stopped")
                except Exception as e:
                    self.logger.warning(f"Persistence service stop error: {e}")

            self.logger.info("TradeLifecycleManager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during TradeLifecycleManager cleanup: {e}")
            raise
        finally:
            # Ensure task reference is cleared even if cleanup fails
            if hasattr(self, "_monitoring_task"):
                self._monitoring_task = None

    @time_execution
    @cache_invalidate(patterns=["trade_lifecycle:*"], namespace="state")
    async def start_trade_lifecycle(
        self, bot_id: str, strategy_name: str, order_request: OrderRequest
    ) -> str:
        """
        Start a new trade lifecycle.

        Args:
            bot_id: Bot identifier
            strategy_name: Strategy that generated the signal
            order_request: Order request details

        Returns:
            Trade ID

        Raises:
            StateConsistencyError: If lifecycle start fails
        """
        try:
            # Create trade context through service layer
            trade_context = await self._lifecycle_service.create_trade_context(
                bot_id=bot_id,
                strategy_name=strategy_name,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
            )

            # Store in active trades
            self.active_trades[trade_context.trade_id] = trade_context

            # Cache trade context for persistence
            await self._cache_trade_context(trade_context)

            # Log trade start
            await self._log_trade_event(
                trade_context.trade_id,
                TradeEvent.SIGNAL_RECEIVED,
                {"order_request": order_request.model_dump()},
            )

            self.logger.info(
                "Trade lifecycle started",
                trade_id=trade_context.trade_id,
                bot_id=bot_id,
                strategy=strategy_name,
                symbol=order_request.symbol,
                side=order_request.side.value,
            )

            return trade_context.trade_id

        except Exception as e:
            self.logger.error(f"Failed to start trade lifecycle: {e}")
            raise StateConsistencyError(f"Trade lifecycle start failed: {e}") from e

    async def transition_trade_state(
        self,
        trade_id: str,
        new_state: TradeLifecycleState,
        event_data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Transition a trade to a new state.

        Args:
            trade_id: Trade identifier
            new_state: Target state
            event_data: Additional event data

        Returns:
            True if transition successful

        Raises:
            StateConsistencyError: If transition is invalid
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateConsistencyError(f"Trade {trade_id} not found")

            # Validate transition through service layer
            current_state = trade_context.current_state
            transition_valid = await self._lifecycle_service.validate_trade_transition(
                current_state, new_state
            )
            if not transition_valid:
                raise StateConsistencyError(
                    f"Invalid transition from {current_state.value} to {new_state.value}"
                )

            # Update trade context
            trade_context.previous_state = current_state
            trade_context.current_state = new_state

            # Update timestamps based on state
            current_time = datetime.now(timezone.utc)
            if new_state == TradeLifecycleState.ORDER_SUBMITTED:
                trade_context.order_submission_timestamp = current_time
            elif new_state == TradeLifecycleState.PARTIALLY_FILLED:
                if not trade_context.first_fill_timestamp:
                    trade_context.first_fill_timestamp = current_time
            elif new_state == TradeLifecycleState.FULLY_FILLED:
                trade_context.final_fill_timestamp = current_time
            elif new_state == TradeLifecycleState.SETTLED:
                trade_context.settlement_timestamp = current_time

            # Cache updated context
            await self._cache_trade_context(trade_context)

            # Log state transition
            await self._log_trade_event(trade_id, self._state_to_event(new_state), event_data or {})

            # Handle terminal states
            if new_state in [
                TradeLifecycleState.CANCELLED,
                TradeLifecycleState.REJECTED,
                TradeLifecycleState.ATTRIBUTED,
            ]:
                await self._finalize_trade(trade_id)

            self.logger.info(
                "Trade state transition",
                trade_id=trade_id,
                from_state=current_state.value,
                to_state=new_state.value,
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to transition trade state: {e}", trade_id=trade_id)
            raise StateConsistencyError(f"Trade state transition failed: {e}") from e

    async def update_trade_execution(
        self, trade_id: str, execution_result: ExecutionResult
    ) -> None:
        """
        Update trade with execution results.

        Args:
            trade_id: Trade identifier
            execution_result: Execution result details

        Raises:
            StateConsistencyError: If update fails
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateConsistencyError(f"Trade {trade_id} not found")

            # Update execution details
            trade_context.execution_results.append(execution_result)
            trade_context.order_id = execution_result.instruction_id
            trade_context.exchange_order_id = getattr(execution_result, "exchange_order_id", None)

            # Update fill quantities
            if execution_result.filled_quantity > 0:
                trade_context.filled_quantity += execution_result.filled_quantity
                trade_context.remaining_quantity = (
                    trade_context.original_quantity - trade_context.filled_quantity
                )

                # Update average fill price
                if trade_context.filled_quantity > 0:
                    total_value = Decimal("0")
                    total_quantity = Decimal("0")

                    for result in trade_context.execution_results:
                        if result.filled_quantity > 0:
                            fill_value = result.filled_quantity * result.average_price
                            total_value += fill_value
                            total_quantity += result.filled_quantity

                    if total_quantity > 0:
                        trade_context.average_fill_price = total_value / total_quantity

            # Update fees
            trade_context.fees_paid += getattr(execution_result, "total_fees", Decimal("0"))

            # Determine new state
            if trade_context.remaining_quantity <= 0:
                await self.transition_trade_state(
                    trade_id,
                    TradeLifecycleState.FULLY_FILLED,
                    {"execution_result": execution_result.model_dump()},
                )
            elif trade_context.filled_quantity > 0:
                await self.transition_trade_state(
                    trade_id,
                    TradeLifecycleState.PARTIALLY_FILLED,
                    {"execution_result": execution_result.model_dump()},
                )

            self.logger.info(
                "Trade execution updated",
                trade_id=trade_id,
                filled_quantity=str(execution_result.filled_quantity),
                total_filled=str(trade_context.filled_quantity),
            )

        except Exception as e:
            self.logger.error(f"Failed to update trade execution: {e}", trade_id=trade_id)
            raise StateConsistencyError(f"Trade execution update failed: {e}") from e

    async def update_trade_event(
        self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]
    ) -> None:
        """
        Update trade event (implements ITradeLifecycleManager protocol).

        Args:
            trade_id: Trade identifier
            event: Trade event
            event_data: Additional event data

        Raises:
            StateConsistencyError: If update fails
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateConsistencyError(f"Trade {trade_id} not found")

            # Log the event
            await self._log_trade_event(trade_id, event, event_data)

            # Map events to state transitions
            event_to_state = {
                TradeEvent.ORDER_ACCEPTED: TradeLifecycleState.ORDER_SUBMITTED,
                TradeEvent.ORDER_SUBMITTED: TradeLifecycleState.ORDER_SUBMITTED,
                TradeEvent.PARTIAL_FILL: TradeLifecycleState.PARTIALLY_FILLED,
                TradeEvent.COMPLETE_FILL: TradeLifecycleState.FULLY_FILLED,
                TradeEvent.ORDER_CANCELLED: TradeLifecycleState.CANCELLED,
                TradeEvent.ORDER_REJECTED: TradeLifecycleState.REJECTED,
                TradeEvent.ORDER_EXPIRED: TradeLifecycleState.CANCELLED,
                TradeEvent.SETTLEMENT_COMPLETE: TradeLifecycleState.SETTLED,
                TradeEvent.ATTRIBUTION_COMPLETE: TradeLifecycleState.ATTRIBUTED,
            }

            # Transition state if mapping exists
            if event in event_to_state:
                new_state = event_to_state[event]
                await self.transition_trade_state(trade_id, new_state, event_data)

            # Handle specific event data updates
            if event in [TradeEvent.PARTIAL_FILL, TradeEvent.COMPLETE_FILL]:
                # Update fill information if provided
                if "filled_quantity" in event_data:
                    trade_context.filled_quantity = Decimal(str(event_data["filled_quantity"]))
                    trade_context.remaining_quantity = (
                        trade_context.original_quantity - trade_context.filled_quantity
                    )
                if "average_price" in event_data:
                    trade_context.average_fill_price = Decimal(str(event_data["average_price"]))
                if "fees" in event_data:
                    trade_context.fees_paid += Decimal(str(event_data["fees"]))

            # Cache updated context
            await self._cache_trade_context(trade_context)

            self.logger.info(
                "Trade event updated",
                trade_id=trade_id,
                event=event.value,
                current_state=trade_context.current_state.value,
            )

        except Exception as e:
            self.logger.error(f"Failed to update trade event: {e}", trade_id=trade_id)
            raise StateConsistencyError(f"Trade event update failed: {e}") from e

    async def calculate_trade_performance(self, trade_id: str) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Performance metrics dictionary

        Raises:
            StateConsistencyError: If calculation fails
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                # Try to load from history
                for record in self.trade_history:
                    if record.trade_id == trade_id:
                        return self._history_record_to_performance(record)
                raise StateConsistencyError(f"Trade {trade_id} not found")

            performance = {
                "trade_id": trade_id,
                "symbol": trade_context.symbol,
                "side": trade_context.side.value,
                "filled_quantity": str(trade_context.filled_quantity),
                "average_fill_price": str(trade_context.average_fill_price),
                "fees_paid": str(trade_context.fees_paid),
                "realized_pnl": str(trade_context.realized_pnl),
                "unrealized_pnl": str(trade_context.unrealized_pnl),
                "net_pnl": str(trade_context.realized_pnl - trade_context.fees_paid),
                "quality_score": trade_context.quality_score,
                "execution_quality": trade_context.execution_quality.copy(),
            }

            # Calculate timing metrics
            if trade_context.order_submission_timestamp and trade_context.signal_timestamp:
                signal_to_submission = (
                    trade_context.order_submission_timestamp - trade_context.signal_timestamp
                ).total_seconds()
                performance["signal_to_submission_seconds"] = Decimal(str(signal_to_submission))

            if trade_context.final_fill_timestamp and trade_context.order_submission_timestamp:
                execution_duration = (
                    trade_context.final_fill_timestamp - trade_context.order_submission_timestamp
                ).total_seconds()
                performance["execution_duration_seconds"] = Decimal(str(execution_duration))

            # Calculate slippage (if market order with expected price)
            if (
                trade_context.requested_price
                and trade_context.average_fill_price > 0
                and trade_context.order_type == OrderType.MARKET
            ):
                if trade_context.side == OrderSide.BUY:
                    slippage = (
                        trade_context.average_fill_price - trade_context.requested_price
                    ) / trade_context.requested_price
                else:
                    slippage = (
                        trade_context.requested_price - trade_context.average_fill_price
                    ) / trade_context.requested_price

                performance["slippage_percentage"] = str(slippage * Decimal("100"))
                performance["slippage_bps"] = str(slippage * Decimal("10000"))

            return performance

        except Exception as e:
            self.logger.error(f"Failed to calculate trade performance: {e}", trade_id=trade_id)
            raise StateConsistencyError(f"Trade performance calculation failed: {e}") from e

    @cached(
        ttl=60,
        namespace="state",
        data_type="orders",
        key_generator=lambda self, **kwargs: (
            f"trade_history:{kwargs.get('bot_id')}:{kwargs.get('strategy_name')}:"
            f"{kwargs.get('symbol')}:{kwargs.get('limit', 100)}"
        ),
    )
    async def get_trade_history(
        self,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get trade history with optional filters.

        Args:
            bot_id: Filter by bot ID
            strategy_name: Filter by strategy
            symbol: Filter by symbol
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum records to return

        Returns:
            List of trade history records
        """
        try:
            filtered_trades = []

            for record in self.trade_history:
                # Apply filters
                if bot_id and record.bot_id != bot_id:
                    continue
                if strategy_name and record.strategy_name != strategy_name:
                    continue
                if symbol and record.symbol != symbol:
                    continue
                if start_date and record.signal_time < start_date:
                    continue
                if end_date and record.signal_time > end_date:
                    continue

                filtered_trades.append(
                    {
                        "trade_id": record.trade_id,
                        "bot_id": record.bot_id,
                        "strategy_name": record.strategy_name,
                        "symbol": record.symbol,
                        "side": record.side.value,
                        "quantity": str(record.quantity),
                        "average_price": str(record.average_price),
                        "pnl": str(record.pnl),
                        "net_pnl": str(record.net_pnl),
                        "fees": str(record.fees),
                        "quality_score": record.quality_score,
                        "execution_time_seconds": record.execution_time_seconds,
                        "slippage_bps": record.slippage_bps,
                        "signal_time": record.signal_time.isoformat(),
                        "execution_time": record.execution_time.isoformat(),
                        "settlement_time": record.settlement_time.isoformat(),
                    }
                )

            # Sort by signal time (newest first) and limit
            filtered_trades.sort(key=lambda x: x["signal_time"], reverse=True)
            return filtered_trades[:limit]

        except Exception as e:
            self.logger.error(f"Failed to get trade history: {e}")
            return []

    async def get_performance_attribution(
        self, bot_id: str, period_days: int = 30
    ) -> dict[str, Any]:
        """
        Get performance attribution analysis for a bot.

        Args:
            bot_id: Bot identifier
            period_days: Analysis period in days

        Returns:
            Performance attribution analysis
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=period_days)

            # Get trades for the period
            trades = await self.get_trade_history(
                bot_id=bot_id, start_date=start_date, end_date=end_date
            )

            if not trades:
                return {
                    "bot_id": bot_id,
                    "period_days": period_days,
                    "total_trades": 0,
                    "attribution": {},
                }

            # Calculate attribution components
            total_pnl = sum(Decimal(str(t["pnl"])) for t in trades)
            total_fees = sum(Decimal(str(t["fees"])) for t in trades)
            net_pnl = total_pnl - total_fees

            # Simplified attribution (can be enhanced)
            attribution = {
                "total_pnl": str(total_pnl),
                "gross_pnl": str(total_pnl),
                "fees_impact": str(-total_fees),
                "net_pnl": str(net_pnl),
                "trade_count": len(trades),
                "win_rate": len([t for t in trades if Decimal(str(t["pnl"])) > 0]) / len(trades),
                "average_trade_pnl": str(total_pnl / len(trades)),
                "average_quality_score": sum(t["quality_score"] for t in trades) / len(trades),
            }

            return {
                "bot_id": bot_id,
                "period_days": period_days,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "total_trades": len(trades),
                "attribution": attribution,
            }

        except Exception as e:
            self.logger.error(f"Failed to get performance attribution: {e}")
            return {"error": str(e)}

    # Private helper methods

    async def _cache_trade_context(self, trade_context: TradeContext) -> None:
        """Cache trade context using cache service abstraction or persistence service."""
        try:
            # Try persistence service first (more reliable)
            if self._persistence_service:
                # Create a pseudo-state for the trade context
                from .state_service import StateType as _StateType

                context_data = {
                    "trade_id": trade_context.trade_id,
                    "bot_id": trade_context.bot_id,
                    "strategy_name": trade_context.strategy_name,
                    "symbol": trade_context.symbol,
                    "side": trade_context.side.value,
                    "current_state": trade_context.current_state.value,
                    "original_quantity": str(trade_context.original_quantity),
                    "filled_quantity": str(trade_context.filled_quantity),
                    "average_fill_price": str(trade_context.average_fill_price),
                    "fees_paid": str(trade_context.fees_paid),
                    "signal_timestamp": trade_context.signal_timestamp.isoformat(),
                }

                # Create metadata using centralized utility
                from src.utils.state_utils import create_state_metadata

                metadata = create_state_metadata(
                    state_id=trade_context.trade_id,
                    state_type=_StateType.TRADE_STATE,
                    source_component="TradeLifecycleManager",
                    state_data=context_data
                )

                # Save through persistence service
                await self._persistence_service.save_state(
                    _StateType.TRADE_STATE, trade_context.trade_id, context_data, metadata
                )

            # Fall back to cache service if available
            elif self.cache_service:
                cache_key = CacheKeys.trade_lifecycle(trade_context.trade_id)
                context_data = {
                    "trade_id": trade_context.trade_id,
                    "bot_id": trade_context.bot_id,
                    "strategy_name": trade_context.strategy_name,
                    "symbol": trade_context.symbol,
                    "side": trade_context.side.value,
                    "current_state": trade_context.current_state.value,
                    "original_quantity": str(trade_context.original_quantity),
                    "filled_quantity": str(trade_context.filled_quantity),
                    "average_fill_price": str(trade_context.average_fill_price),
                    "fees_paid": str(trade_context.fees_paid),
                    "signal_timestamp": trade_context.signal_timestamp.isoformat(),
                }

                await self.cache_service.set(
                    cache_key,
                    context_data,
                    namespace="state",
                    ttl=self._get_state_ttl(),
                    data_type="state",
                )
            # If neither service is available, context is only stored in memory

        except Exception as e:
            self.logger.warning(f"Failed to cache trade context: {e}")

    async def _log_trade_event(
        self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]
    ) -> None:
        """Log a trade event through service layer."""
        try:
            event_record = {
                "trade_id": trade_id,
                "event": event.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event_data,
            }

            # Use persistence service if available
            if self._persistence_service:
                from .state_service import StateType as _StateType

                # Create event state for logging
                event_state_id = f"{trade_id}_{event.value}_{datetime.now().timestamp()}"
                metadata = create_state_metadata(
                    state_id=event_state_id,
                    state_type=_StateType.TRADE_STATE,  # Use trade state type for events
                    source_component="TradeLifecycleManager",
                    state_data=event_record
                )

                # Save event through service layer
                await self._persistence_service.save_state(
                    _StateType.TRADE_STATE, event_state_id, event_record, metadata
                )
            else:
                # Fall back to logging only
                self.logger.debug(f"Trade event logged: {event_record}")

        except Exception as e:
            self.logger.warning(f"Failed to log trade event: {e}")

    def _state_to_event(self, state: TradeLifecycleState) -> TradeEvent:
        """Convert state to corresponding event."""
        state_to_event_map = {
            TradeLifecycleState.SIGNAL_GENERATED: TradeEvent.SIGNAL_RECEIVED,
            TradeLifecycleState.PRE_TRADE_VALIDATION: TradeEvent.VALIDATION_PASSED,
            TradeLifecycleState.ORDER_SUBMITTED: TradeEvent.ORDER_SUBMITTED,
            TradeLifecycleState.PARTIALLY_FILLED: TradeEvent.PARTIAL_FILL,
            TradeLifecycleState.FULLY_FILLED: TradeEvent.COMPLETE_FILL,
            TradeLifecycleState.CANCELLED: TradeEvent.ORDER_CANCELLED,
            TradeLifecycleState.REJECTED: TradeEvent.ORDER_REJECTED,
            TradeLifecycleState.SETTLED: TradeEvent.SETTLEMENT_COMPLETE,
            TradeLifecycleState.ATTRIBUTED: TradeEvent.ATTRIBUTION_COMPLETE,
        }
        return state_to_event_map.get(state, TradeEvent.SIGNAL_RECEIVED)

    async def _finalize_trade(self, trade_id: str) -> None:
        """Finalize a trade and move to history."""
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                return

            # Create history record through service layer
            history_record = await self._lifecycle_service.create_history_record(trade_context)

            # Execution time calculation is now handled by the service layer

            # Persist history record through service layer
            if self._persistence_service:
                from .state_service import StateType as _StateType

                # Convert history record to dict for persistence
                history_data = {
                    "trade_id": history_record.trade_id,
                    "bot_id": history_record.bot_id,
                    "strategy_name": history_record.strategy_name,
                    "symbol": history_record.symbol,
                    "side": history_record.side.value,
                    "quantity": str(history_record.quantity),
                    "average_price": str(history_record.average_price),
                    "pnl": str(history_record.pnl),
                    "fees": str(history_record.fees),
                    "net_pnl": str(history_record.net_pnl),
                    "quality_score": history_record.quality_score,
                    "execution_time_seconds": history_record.execution_time_seconds,
                    "signal_time": history_record.signal_time.isoformat(),
                    "execution_time": history_record.execution_time.isoformat(),
                    "settlement_time": history_record.settlement_time.isoformat(),
                    "status": "finalized",
                }

                metadata = create_state_metadata(
                    state_id=f"{trade_id}_history",
                    state_type=_StateType.TRADE_STATE,
                    source_component="TradeLifecycleManager",
                    state_data=history_data
                )

                await self._persistence_service.save_state(
                    _StateType.TRADE_STATE, f"{trade_id}_history", history_data, metadata
                )

            # Add to in-memory history
            self.trade_history.append(history_record)

            # Remove from active trades
            del self.active_trades[trade_id]

            # Update performance metrics
            self._update_performance_metrics(trade_context)

            self.logger.info("Trade finalized and moved to history", trade_id=trade_id)

        except Exception as e:
            self.logger.error(f"Failed to finalize trade: {e}", trade_id=trade_id)

    def _update_performance_metrics(self, trade_context: TradeContext) -> None:
        """Update global performance metrics."""
        try:
            self.performance_metrics["total_trades"] += 1

            if trade_context.current_state == TradeLifecycleState.ATTRIBUTED:
                self.performance_metrics["successful_trades"] += 1
            else:
                self.performance_metrics["failed_trades"] += 1

            self.performance_metrics["total_pnl"] += trade_context.realized_pnl
            self.performance_metrics["total_fees"] += trade_context.fees_paid

            if trade_context.quality_score:
                # Update average quality score
                total_trades = self.performance_metrics["total_trades"]
                current_avg = self.performance_metrics["average_quality_score"]
                new_avg = (
                    current_avg * (total_trades - 1) + trade_context.quality_score
                ) / total_trades
                self.performance_metrics["average_quality_score"] = new_avg

        except Exception as e:
            self.logger.warning(f"Failed to update performance metrics: {e}")

    def _history_record_to_performance(self, record: TradeHistoryRecord) -> dict[str, Any]:
        """Convert history record to performance dict."""
        return {
            "trade_id": record.trade_id,
            "symbol": record.symbol,
            "side": record.side.value,
            "filled_quantity": str(record.quantity),
            "average_fill_price": str(record.average_price),
            "fees_paid": str(record.fees),
            "realized_pnl": str(record.pnl),
            "net_pnl": str(record.net_pnl),
            "quality_score": record.quality_score,
            "execution_duration_seconds": record.execution_time_seconds,
            "slippage_bps": record.slippage_bps,
        }

    async def _load_active_trades(self) -> None:
        """Load active trades from persistence layer."""
        try:
            from .state_service import StateType as _StateType

            # Load all trade states from persistence
            states = await self._persistence_service.list_states(
                _StateType.TRADE_STATE,
                limit=1000,  # Reasonable limit for active trades
            )

            # Reconstruct trade contexts from persisted data
            loaded_count = 0
            for state_info in states:
                try:
                    trade_data = state_info.get("data", {})
                    trade_id = trade_data.get("trade_id")

                    if trade_id and trade_data.get("current_state") not in [
                        "cancelled",
                        "rejected",
                        "attributed",
                    ]:
                        # Reconstruct trade context (simplified)
                        context = TradeContext(
                            trade_id=trade_id,
                            bot_id=trade_data.get("bot_id", ""),
                            strategy_name=trade_data.get("strategy_name", ""),
                            symbol=trade_data.get("symbol", ""),
                            side=OrderSide(trade_data.get("side", "BUY")),
                            original_quantity=Decimal(
                                str(trade_data.get("original_quantity", "0"))
                            ),
                            filled_quantity=Decimal(str(trade_data.get("filled_quantity", "0"))),
                        )

                        self.active_trades[trade_id] = context
                        loaded_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to reconstruct trade context: {e}")

            self.logger.info(f"Loaded {loaded_count} active trades from persistence")

        except Exception as e:
            self.logger.warning(f"Failed to load active trades: {e}")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Monitor active trades for timeouts, stuck states, etc.
                current_time = datetime.now(timezone.utc)

                for trade_id, trade_context in list(self.active_trades.items()):
                    # Check for stale trades (older than 1 hour without updates)
                    if (current_time - trade_context.signal_timestamp).total_seconds() > self._get_staleness_threshold():
                        self.logger.warning(f"Stale trade detected: {trade_id}")
                        # Could trigger alerts or automatic cleanup

                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def create_trade_state(self, trade: Any) -> None:
        """
        Create a new trade state.

        Args:
            trade: Trade object to create state for
        """
        try:
            # Create trade context from trade object
            trade_id = getattr(trade, "trade_id", str(uuid4()))

            # Create trade context based on trade attributes
            context = TradeContext(
                trade_id=trade_id,
                bot_id=getattr(trade, "bot_id", "unknown"),
                strategy_name=getattr(trade, "strategy_name", "unknown"),
                symbol=getattr(trade, "symbol", ""),
                side=getattr(trade, "side", OrderSide.BUY),
                order_type=getattr(trade, "order_type", OrderType.MARKET),
                original_quantity=getattr(trade, "quantity", Decimal("0")),
                remaining_quantity=getattr(trade, "quantity", Decimal("0")),
                requested_price=getattr(trade, "price", None),
            )

            # Store in active trades
            self.active_trades[trade_id] = context

            # Cache trade context
            await self._cache_trade_context(context)

            self.logger.info(f"Created trade state for trade {trade_id}")

        except Exception as e:
            self.logger.error(f"Failed to create trade state: {e}")
            raise StateConsistencyError(f"Trade state creation failed: {e}") from e

    async def validate_trade_state(self, trade: Any) -> bool:
        """
        Validate trade state.

        Args:
            trade: Trade to validate

        Returns:
            True if trade state is valid
        """
        try:
            if not trade:
                return False

            # Basic trade validation
            quantity = getattr(trade, "quantity", 0)
            if quantity <= 0:
                return False

            price = getattr(trade, "price", 0)
            if price is not None and price <= 0:
                return False

            symbol = getattr(trade, "symbol", "")
            if not symbol:
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Trade state validation error: {e}")
            return False

    async def calculate_trade_pnl(self, trade: Any) -> Decimal:
        """
        Calculate trade PnL.

        Args:
            trade: Trade to calculate PnL for

        Returns:
            Calculated PnL
        """
        try:
            # Get current price (placeholder - would get from market data)
            current_price = Decimal("51000.0")  # Default current price for calculation
            entry_price = getattr(trade, "price", Decimal("50000.0"))
            quantity = getattr(trade, "quantity", Decimal("1.0"))
            side = getattr(trade, "side", OrderSide.BUY)

            # Calculate PnL based on side
            if side == OrderSide.BUY:
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            return pnl

        except Exception as e:
            self.logger.warning(f"Trade PnL calculation error: {e}")
            return Decimal("0")

    async def assess_trade_risk(self, trade: Any) -> str:
        """
        Assess trade risk level.

        Args:
            trade: Trade to assess

        Returns:
            Risk assessment ('NORMAL', 'HIGH_LOSS', 'HIGH_GAIN')
        """
        try:
            # Get PnL and risk thresholds
            pnl = getattr(trade, "pnl", Decimal("0"))

            # Default thresholds (would be configurable)
            max_loss_threshold = getattr(self, "max_loss_threshold", Decimal("5000.0"))
            max_gain_threshold = getattr(self, "max_gain_threshold", Decimal("10000.0"))

            if pnl < -max_loss_threshold:
                return "HIGH_LOSS"
            elif pnl > max_gain_threshold:
                return "HIGH_GAIN"
            else:
                return "NORMAL"

        except Exception as e:
            self.logger.warning(f"Trade risk assessment error: {e}")
            return "NORMAL"

    async def close_trade(self, trade_id: str, final_pnl: Decimal) -> None:
        """
        Close a trade.

        Args:
            trade_id: Trade ID to close
            final_pnl: Final PnL for the trade
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateConsistencyError(f"Trade {trade_id} not found")

            # Update final PnL
            trade_context.realized_pnl = final_pnl

            # Transition to settled state
            await self.transition_trade_state(trade_id, TradeLifecycleState.SETTLED)

            self.logger.info(f"Closed trade {trade_id} with PnL {final_pnl}")

        except Exception as e:
            self.logger.error(f"Failed to close trade {trade_id}: {e}")
            raise StateConsistencyError(f"Trade closure failed: {e}") from e

    async def update_trade_state(self, trade_id: str, trade_data: Any) -> None:
        """
        Update trade state.

        Args:
            trade_id: Trade ID to update
            trade_data: New trade data
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateConsistencyError(f"Trade {trade_id} not found")

            # Update trade context with new data
            if hasattr(trade_data, "current_price"):
                trade_context.average_fill_price = trade_data.current_price

            if hasattr(trade_data, "pnl"):
                trade_context.realized_pnl = trade_data.pnl

            if hasattr(trade_data, "filled_quantity"):
                trade_context.filled_quantity = trade_data.filled_quantity
                trade_context.remaining_quantity = (
                    trade_context.original_quantity - trade_context.filled_quantity
                )

            # Cache updated context
            await self._cache_trade_context(trade_context)

            self.logger.info(f"Updated trade state for {trade_id}")

        except Exception as e:
            self.logger.error(f"Failed to update trade state {trade_id}: {e}")
            raise StateConsistencyError(f"Trade state update failed: {e}") from e

    def _get_state_ttl(self) -> int:
        """Get the TTL for state caching from configuration or use default."""
        from .utils_imports import DEFAULT_CLEANUP_INTERVAL
        try:
            if hasattr(self.config, "state_management") and hasattr(self.config.state_management, "state_ttl_seconds"):
                return self.config.state_management.state_ttl_seconds
            return DEFAULT_CLEANUP_INTERVAL
        except Exception:
            return DEFAULT_CLEANUP_INTERVAL

    def _get_staleness_threshold(self) -> int:
        """Get the trade staleness threshold from configuration or use default."""
        from .utils_imports import DEFAULT_TRADE_STALENESS_THRESHOLD
        try:
            if hasattr(self.config, "state_management") and hasattr(self.config.state_management, "trade_staleness_threshold"):
                return self.config.state_management.trade_staleness_threshold
            return DEFAULT_TRADE_STALENESS_THRESHOLD
        except Exception:
            return DEFAULT_TRADE_STALENESS_THRESHOLD
