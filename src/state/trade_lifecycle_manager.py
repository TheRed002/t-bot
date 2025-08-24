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

from src.base import BaseComponent

# Database service abstractions - use service pattern instead of direct imports
# Caching imports
from src.core.caching import CacheKeys, cache_invalidate, cached, get_cache_manager

# Core framework imports
from src.core.config.main import Config
from src.core.exceptions import StateError
from src.core.types import (
    ExecutionResult,
    OrderRequest,
    OrderSide,
    OrderType,
)
from src.database.service import DatabaseService

# Import utilities through centralized import handler
from .utils_imports import time_execution


class TradeLifecycleState(Enum):
    """Trade lifecycle state enumeration."""

    SIGNAL_GENERATED = "signal_generated"
    PRE_TRADE_VALIDATION = "pre_trade_validation"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    PARTIALLY_FILLED = "partially_filled"
    FULLY_FILLED = "fully_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    SETTLED = "settled"
    ATTRIBUTED = "attributed"


class TradeEvent(Enum):
    """Trade event enumeration."""

    SIGNAL_RECEIVED = "signal_received"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    ORDER_SUBMITTED = "order_submitted"
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    SETTLEMENT_COMPLETE = "settlement_complete"
    ATTRIBUTION_COMPLETE = "attribution_complete"


@dataclass
class TradeContext:
    """Complete context for a trade throughout its lifecycle."""

    trade_id: str = field(default_factory=lambda: str(uuid4()))
    bot_id: str = ""
    strategy_name: str = ""

    # Trade identification
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET

    # State tracking
    current_state: TradeLifecycleState = TradeLifecycleState.SIGNAL_GENERATED
    previous_state: TradeLifecycleState | None = None

    # Order details
    original_quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")

    # Pricing
    requested_price: Decimal | None = None
    average_fill_price: Decimal = Decimal("0")

    # Execution tracking
    order_id: str | None = None
    exchange_order_id: str | None = None
    execution_results: list[ExecutionResult] = field(default_factory=list)

    # Timing
    signal_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_submission_timestamp: datetime | None = None
    first_fill_timestamp: datetime | None = None
    final_fill_timestamp: datetime | None = None
    settlement_timestamp: datetime | None = None

    # Performance metrics
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")

    # Quality metrics
    quality_score: float | None = None
    execution_quality: dict[str, float] = field(default_factory=dict)

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeHistoryRecord:
    """Historical trade record for analysis."""

    trade_id: str = ""
    bot_id: str = ""
    strategy_name: str = ""

    # Trade summary
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")

    # Performance
    pnl: Decimal = Decimal("0")
    return_percentage: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    # Quality metrics
    quality_score: float = 0.0
    execution_time_seconds: float = 0.0
    slippage_bps: float = 0.0

    # Timestamps
    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
    timing_quality: float = 0.0
    execution_quality: float = 0.0
    strategy_quality: float = 0.0

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
        database_service: DatabaseService | None = None,
        cache_service: Any | None = None,
    ):
        """
        Initialize the trade lifecycle manager.

        Args:
            config: Application configuration
            database_service: Database service for data persistence
            cache_service: Cache service for performance optimization
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config

        # Injected services (with fallback handling)
        self.database_service = database_service
        self.cache_service = cache_service

        # Active trades tracking
        self.active_trades: dict[str, TradeContext] = {}
        self.trade_history: list[TradeHistoryRecord] = []

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "average_execution_time": 0.0,
            "average_quality_score": 0.0,
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
            # Initialize database service if available
            if self.database_service and not self.database_service.is_running:
                await self.database_service.start()
                self.logger.info("Database service initialized")
            elif not self.database_service:
                self.logger.warning(
                    "Database service not available - persistence operations will be limited"
                )

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
            raise StateError(f"Failed to initialize TradeLifecycleManager: {e}") from e

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
            StateError: If lifecycle start fails
        """
        try:
            # Create trade context
            trade_context = TradeContext(
                bot_id=bot_id,
                strategy_name=strategy_name,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                original_quantity=order_request.quantity,
                remaining_quantity=order_request.quantity,
                requested_price=order_request.price,
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
            raise StateError(f"Trade lifecycle start failed: {e}") from e

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
            StateError: If transition is invalid
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateError(f"Trade {trade_id} not found")

            # Validate transition
            current_state = trade_context.current_state
            if new_state not in self.valid_transitions.get(current_state, []):
                raise StateError(
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
            raise StateError(f"Trade state transition failed: {e}") from e

    async def update_trade_execution(
        self, trade_id: str, execution_result: ExecutionResult
    ) -> None:
        """
        Update trade with execution results.

        Args:
            trade_id: Trade identifier
            execution_result: Execution result details

        Raises:
            StateError: If update fails
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                raise StateError(f"Trade {trade_id} not found")

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
                filled_quantity=float(execution_result.filled_quantity),
                total_filled=float(trade_context.filled_quantity),
            )

        except Exception as e:
            self.logger.error(f"Failed to update trade execution: {e}", trade_id=trade_id)
            raise StateError(f"Trade execution update failed: {e}") from e

    async def calculate_trade_performance(self, trade_id: str) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Performance metrics dictionary

        Raises:
            StateError: If calculation fails
        """
        try:
            trade_context = self.active_trades.get(trade_id)
            if not trade_context:
                # Try to load from history
                for record in self.trade_history:
                    if record.trade_id == trade_id:
                        return self._history_record_to_performance(record)
                raise StateError(f"Trade {trade_id} not found")

            performance = {
                "trade_id": trade_id,
                "symbol": trade_context.symbol,
                "side": trade_context.side.value,
                "filled_quantity": float(trade_context.filled_quantity),
                "average_fill_price": float(trade_context.average_fill_price),
                "fees_paid": float(trade_context.fees_paid),
                "realized_pnl": float(trade_context.realized_pnl),
                "unrealized_pnl": float(trade_context.unrealized_pnl),
                "net_pnl": float(trade_context.realized_pnl - trade_context.fees_paid),
                "quality_score": trade_context.quality_score,
                "execution_quality": trade_context.execution_quality.copy(),
            }

            # Calculate timing metrics
            if trade_context.order_submission_timestamp and trade_context.signal_timestamp:
                signal_to_submission = (
                    trade_context.order_submission_timestamp - trade_context.signal_timestamp
                ).total_seconds()
                performance["signal_to_submission_seconds"] = signal_to_submission

            if trade_context.final_fill_timestamp and trade_context.order_submission_timestamp:
                execution_duration = (
                    trade_context.final_fill_timestamp - trade_context.order_submission_timestamp
                ).total_seconds()
                performance["execution_duration_seconds"] = execution_duration

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

                performance["slippage_percentage"] = float(slippage * 100)
                performance["slippage_bps"] = float(slippage * 10000)

            return performance

        except Exception as e:
            self.logger.error(f"Failed to calculate trade performance: {e}", trade_id=trade_id)
            raise StateError(f"Trade performance calculation failed: {e}") from e

    @cached(
        ttl=60,
        namespace="state",
        data_type="orders",
        key_generator=lambda self, bot_id=None, strategy_name=None, symbol=None, start_date=None, end_date=None, limit=100: f"trade_history:{bot_id}:{strategy_name}:{symbol}:{limit}",
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
                        "quantity": float(record.quantity),
                        "average_price": float(record.average_price),
                        "pnl": float(record.pnl),
                        "net_pnl": float(record.net_pnl),
                        "fees": float(record.fees),
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
                "total_pnl": float(total_pnl),
                "gross_pnl": float(total_pnl),
                "fees_impact": float(-total_fees),
                "net_pnl": float(net_pnl),
                "trade_count": len(trades),
                "win_rate": len([t for t in trades if t["pnl"] > 0]) / len(trades),
                "average_trade_pnl": float(total_pnl / len(trades)),
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
        """Cache trade context using cache service abstraction."""
        if not self.cache_service:
            # No cache service available - skip caching
            return

        try:
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
                ttl=3600,  # 1 hour TTL
                data_type="state",
            )

        except Exception as e:
            self.logger.warning(f"Failed to cache trade context: {e}")

    async def _log_trade_event(
        self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]
    ) -> None:
        """Log a trade event."""
        try:
            # Log to database (simplified)
            event_record = {
                "trade_id": trade_id,
                "event": event.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event_data,
            }

            # In a full implementation, this would write to a trade_events table
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

            # Create history record
            history_record = TradeHistoryRecord(
                trade_id=trade_id,
                bot_id=trade_context.bot_id,
                strategy_name=trade_context.strategy_name,
                symbol=trade_context.symbol,
                side=trade_context.side,
                quantity=trade_context.filled_quantity,
                average_price=trade_context.average_fill_price,
                pnl=trade_context.realized_pnl,
                fees=trade_context.fees_paid,
                net_pnl=trade_context.realized_pnl - trade_context.fees_paid,
                quality_score=trade_context.quality_score or 0.0,
                signal_time=trade_context.signal_timestamp,
                execution_time=trade_context.final_fill_timestamp or trade_context.signal_timestamp,
                settlement_time=trade_context.settlement_timestamp or datetime.now(timezone.utc),
            )

            # Calculate execution time
            if trade_context.final_fill_timestamp and trade_context.order_submission_timestamp:
                execution_duration = (
                    trade_context.final_fill_timestamp - trade_context.order_submission_timestamp
                ).total_seconds()
                history_record.execution_time_seconds = execution_duration

            # Add to history
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
            "filled_quantity": float(record.quantity),
            "average_fill_price": float(record.average_price),
            "fees_paid": float(record.fees),
            "realized_pnl": float(record.pnl),
            "net_pnl": float(record.net_pnl),
            "quality_score": record.quality_score,
            "execution_duration_seconds": record.execution_time_seconds,
            "slippage_bps": record.slippage_bps,
        }

    async def _load_active_trades(self) -> None:
        """Load active trades from persistence layer."""
        try:
            if not self.cache_service:
                self.logger.info("No cache service available - starting with empty active trades")
                return

            # In a full implementation, this would scan cache service for active trade contexts
            # For now, we'll use a simple pattern-based scan if supported by the cache service
            if hasattr(self.cache_service, "invalidate_pattern"):
                # Cache service supports pattern operations
                self.logger.info("Cache service available for trade context persistence")
            else:
                self.logger.info("Cache service available but limited pattern support")

            self.logger.info("Active trades loading completed")

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
                    if (current_time - trade_context.signal_timestamp).total_seconds() > 3600:
                        self.logger.warning(f"Stale trade detected: {trade_id}")
                        # Could trigger alerts or automatic cleanup

                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
