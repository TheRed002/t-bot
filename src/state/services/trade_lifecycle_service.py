"""
Trade Lifecycle Service - Service layer for trade lifecycle operations.

This service provides business logic for trade lifecycle management,
decoupled from infrastructure and presentation concerns.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from src.core.base.service import BaseService
from src.core.exceptions import BusinessRuleValidationError, StateError
from src.core.types import OrderSide, OrderType

# Import types directly to avoid circular dependencies
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal
from uuid import uuid4


# Define the enums and dataclasses here to avoid circular import
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


@dataclass
class TradeContext:
    """Complete context for a trade throughout its lifecycle."""
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    bot_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    current_state: TradeLifecycleState = TradeLifecycleState.SIGNAL_GENERATED
    previous_state: TradeLifecycleState | None = None
    original_quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    requested_price: Decimal | None = None
    average_fill_price: Decimal = Decimal("0")
    order_id: str | None = None
    exchange_order_id: str | None = None
    signal_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_submission_timestamp: datetime | None = None
    first_fill_timestamp: datetime | None = None
    final_fill_timestamp: datetime | None = None
    settlement_timestamp: datetime | None = None
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    quality_score: float | None = None
    execution_quality: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Additional field from original implementation
    execution_results: list[Any] = field(default_factory=list)


@dataclass
class TradeHistoryRecord:
    """Historical trade record for analysis."""
    trade_id: str = ""
    bot_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    pnl: Decimal = Decimal("0")
    return_percentage: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    quality_score: float = 0.0
    execution_time_seconds: float = 0.0
    slippage_bps: float = 0.0
    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradeLifecycleServiceProtocol(Protocol):
    """Protocol defining the trade lifecycle service interface."""

    async def create_trade_context(
        self, bot_id: str, strategy_name: str, symbol: str, side: OrderSide, 
        order_type: OrderType, quantity: Decimal, price: Decimal | None
    ) -> TradeContext: ...

    async def validate_trade_transition(
        self, current_state: TradeLifecycleState, new_state: TradeLifecycleState
    ) -> bool: ...

    async def calculate_trade_performance(
        self, context: TradeContext
    ) -> dict[str, Any]: ...

    async def create_history_record(
        self, context: TradeContext
    ) -> TradeHistoryRecord: ...


class TradeLifecycleService(BaseService):
    """
    Trade lifecycle service implementing core trade lifecycle business logic.

    This service handles all business rules, validations, and trade processing
    logic independent of infrastructure concerns.
    """

    def __init__(self, config: Any = None):
        """
        Initialize the trade lifecycle service.
        
        Args:
            config: Optional configuration object for business rules
        """
        super().__init__(name="TradeLifecycleService")

        # Valid state transitions for trade lifecycle
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

        self.logger.info("TradeLifecycleService initialized")

    async def create_trade_context(
        self, bot_id: str, strategy_name: str, symbol: str, side: OrderSide, 
        order_type: OrderType, quantity: Decimal, price: Decimal | None = None
    ) -> TradeContext:
        """
        Create a new trade context.

        Args:
            bot_id: Bot identifier
            strategy_name: Strategy name
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (optional)

        Returns:
            TradeContext instance

        Raises:
            BusinessRuleValidationError: If trade parameters are invalid
        """
        try:
            # Validate inputs
            if quantity <= 0:
                raise BusinessRuleValidationError("Trade quantity must be positive")
            
            if not symbol or not symbol.strip():
                raise BusinessRuleValidationError("Trade symbol is required")
            
            if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS] and (not price or price <= 0):
                raise BusinessRuleValidationError(f"{order_type.value} orders require a positive price")

            # Create trade context
            context = TradeContext(
                bot_id=bot_id,
                strategy_name=strategy_name,
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                original_quantity=quantity,
                remaining_quantity=quantity,
                requested_price=price,
                signal_timestamp=datetime.now(timezone.utc),
            )

            self.logger.debug(
                f"Created trade context {context.trade_id}",
                extra={
                    "bot_id": bot_id,
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                }
            )

            return context

        except Exception as e:
            self.logger.error(f"Failed to create trade context: {e}")
            raise BusinessRuleValidationError(f"Trade context creation failed: {e}") from e

    async def validate_trade_transition(
        self, current_state: TradeLifecycleState, new_state: TradeLifecycleState
    ) -> bool:
        """
        Validate if a state transition is allowed.

        Args:
            current_state: Current trade state
            new_state: Proposed new state

        Returns:
            True if transition is valid
        """
        try:
            allowed_states = self.valid_transitions.get(current_state, [])
            is_valid = new_state in allowed_states
            
            if not is_valid:
                self.logger.warning(
                    f"Invalid state transition from {current_state.value} to {new_state.value}"
                )
            
            return is_valid

        except Exception as e:
            self.logger.error(f"State transition validation error: {e}")
            return False

    async def calculate_trade_performance(self, context: TradeContext) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a trade.

        Args:
            context: Trade context

        Returns:
            Performance metrics dictionary
        """
        try:
            performance = {
                "trade_id": context.trade_id,
                "symbol": context.symbol,
                "side": context.side.value,
                "filled_quantity": str(context.filled_quantity),
                "average_fill_price": str(context.average_fill_price),
                "fees_paid": str(context.fees_paid),
                "realized_pnl": str(context.realized_pnl),
                "unrealized_pnl": str(context.unrealized_pnl),
                "net_pnl": str(context.realized_pnl - context.fees_paid),
                "quality_score": context.quality_score,
                "execution_quality": context.execution_quality.copy(),
            }

            # Calculate timing metrics
            if context.order_submission_timestamp and context.signal_timestamp:
                signal_to_submission = (
                    context.order_submission_timestamp - context.signal_timestamp
                ).total_seconds()
                performance["signal_to_submission_seconds"] = signal_to_submission

            if context.final_fill_timestamp and context.order_submission_timestamp:
                execution_duration = (
                    context.final_fill_timestamp - context.order_submission_timestamp
                ).total_seconds()
                performance["execution_duration_seconds"] = execution_duration

            # Calculate slippage for market orders
            if (
                context.requested_price
                and context.average_fill_price > 0
                and context.order_type == OrderType.MARKET
            ):
                if context.side == OrderSide.BUY:
                    slippage = (
                        context.average_fill_price - context.requested_price
                    ) / context.requested_price
                else:
                    slippage = (
                        context.requested_price - context.average_fill_price
                    ) / context.requested_price

                performance["slippage_percentage"] = str(slippage * 100)
                performance["slippage_bps"] = str(slippage * 10000)

            return performance

        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            raise StateError(f"Failed to calculate trade performance: {e}") from e

    async def create_history_record(self, context: TradeContext) -> TradeHistoryRecord:
        """
        Create a history record from a trade context.

        Args:
            context: Trade context

        Returns:
            TradeHistoryRecord instance
        """
        try:
            # Calculate execution time
            execution_time_seconds = 0.0
            if context.final_fill_timestamp and context.order_submission_timestamp:
                execution_duration = (
                    context.final_fill_timestamp - context.order_submission_timestamp
                ).total_seconds()
                execution_time_seconds = execution_duration

            # Calculate slippage in basis points
            slippage_bps = 0.0
            if (
                context.requested_price 
                and context.average_fill_price > 0
                and context.order_type == OrderType.MARKET
            ):
                if context.side == OrderSide.BUY:
                    slippage = (
                        context.average_fill_price - context.requested_price
                    ) / context.requested_price
                else:
                    slippage = (
                        context.requested_price - context.average_fill_price
                    ) / context.requested_price
                slippage_bps = slippage * 10000

            # Create history record
            record = TradeHistoryRecord(
                trade_id=context.trade_id,
                bot_id=context.bot_id,
                strategy_name=context.strategy_name,
                symbol=context.symbol,
                side=context.side,
                quantity=context.filled_quantity,
                average_price=context.average_fill_price,
                pnl=context.realized_pnl,
                fees=context.fees_paid,
                net_pnl=context.realized_pnl - context.fees_paid,
                quality_score=context.quality_score or 0.0,
                execution_time_seconds=execution_time_seconds,
                slippage_bps=slippage_bps,
                signal_time=context.signal_timestamp,
                execution_time=context.final_fill_timestamp or context.signal_timestamp,
                settlement_time=context.settlement_timestamp or datetime.now(timezone.utc),
            )

            return record

        except Exception as e:
            self.logger.error(f"Failed to create history record: {e}")
            raise StateError(f"History record creation failed: {e}") from e

    async def apply_business_rules(self, context: TradeContext) -> list[str]:
        """
        Apply business rules validation to a trade context.

        Args:
            context: Trade context to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        try:
            # Validate quantities
            if context.original_quantity <= 0:
                issues.append("Original quantity must be positive")
            
            if context.filled_quantity < 0:
                issues.append("Filled quantity cannot be negative")
            
            if context.filled_quantity > context.original_quantity:
                issues.append("Filled quantity cannot exceed original quantity")

            # Validate prices
            if context.requested_price is not None and context.requested_price <= 0:
                issues.append("Requested price must be positive")
            
            if context.average_fill_price < 0:
                issues.append("Average fill price cannot be negative")

            # Validate state consistency
            if context.current_state == TradeLifecycleState.FULLY_FILLED:
                if context.remaining_quantity > 0:
                    issues.append("Fully filled trades cannot have remaining quantity")

            # Validate timestamps
            if (
                context.order_submission_timestamp 
                and context.signal_timestamp
                and context.order_submission_timestamp < context.signal_timestamp
            ):
                issues.append("Order submission cannot be before signal generation")

            return issues

        except Exception as e:
            self.logger.error(f"Business rules validation error: {e}")
            return [f"Validation error: {e}"]