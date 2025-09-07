"""
Analytics Event System.

This module provides event-driven patterns to break circular dependencies
between analytics components and enable loose coupling.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.core.logging import get_logger
from src.core.types import Order, Position, Trade


class AnalyticsEventType(Enum):
    """Types of analytics events."""

    # Data update events
    POSITION_UPDATED = "position_updated"
    TRADE_EXECUTED = "trade_executed"
    ORDER_UPDATED = "order_updated"
    PRICE_UPDATED = "price_updated"
    BENCHMARK_UPDATED = "benchmark_updated"

    # Analytics calculation events
    PORTFOLIO_METRICS_CALCULATED = "portfolio_metrics_calculated"
    RISK_METRICS_CALCULATED = "risk_metrics_calculated"
    STRATEGY_METRICS_CALCULATED = "strategy_metrics_calculated"

    # Alert events
    ALERT_GENERATED = "alert_generated"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_RESOLVED = "alert_resolved"
    RISK_LIMIT_BREACHED = "risk_limit_breached"

    # System events
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    ERROR_OCCURRED = "error_occurred"
    HEALTH_CHECK_COMPLETED = "health_check_completed"

    # Export events
    DATA_EXPORTED = "data_exported"
    REPORT_GENERATED = "report_generated"


class AnalyticsEvent(BaseModel):
    """Base analytics event."""

    event_id: str
    event_type: AnalyticsEventType
    timestamp: datetime
    source_component: str
    data: dict[str, Any]
    metadata: dict[str, Any] = {}


class PositionUpdatedEvent(AnalyticsEvent):
    """Event fired when position is updated."""

    def __init__(self, position: Position, source_component: str, **kwargs):
        from src.utils.datetime_utils import get_current_utc_timestamp

        timestamp = get_current_utc_timestamp()
        super().__init__(
            event_id=f"position_updated_{position.symbol}_{timestamp}",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=timestamp,
            source_component=source_component,
            data={
                "position": position.model_dump(),
                "symbol": position.symbol,
                "exchange": position.exchange,
            },
            **kwargs,
        )


class TradeExecutedEvent(AnalyticsEvent):
    """Event fired when trade is executed."""

    def __init__(self, trade: Trade, source_component: str, **kwargs):
        super().__init__(
            event_id=f"trade_executed_{trade.trade_id}",
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=trade.timestamp,
            source_component=source_component,
            data={
                "trade": trade.model_dump(),
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "exchange": trade.exchange,
            },
            **kwargs,
        )


class OrderUpdatedEvent(AnalyticsEvent):
    """Event fired when order is updated."""

    def __init__(self, order: Order, source_component: str, **kwargs):
        super().__init__(
            event_id=f"order_updated_{order.order_id}",
            event_type=AnalyticsEventType.ORDER_UPDATED,
            timestamp=order.updated_at or order.created_at,
            source_component=source_component,
            data={
                "order": order.model_dump(),
                "order_id": order.order_id,
                "symbol": order.symbol,
                "exchange": order.exchange,
            },
            **kwargs,
        )


class PriceUpdatedEvent(AnalyticsEvent):
    """Event fired when price is updated."""

    def __init__(
        self, symbol: str, price: Decimal, timestamp: datetime, source_component: str, **kwargs
    ):
        super().__init__(
            event_id=f"price_updated_{symbol}_{timestamp}",
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=timestamp,
            source_component=source_component,
            data={
                "symbol": symbol,
                "price": str(price),
                "timestamp": timestamp.isoformat(),
            },
            **kwargs,
        )


class RiskLimitBreachedEvent(AnalyticsEvent):
    """Event fired when risk limit is breached."""

    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        breach_percentage: float,
        source_component: str,
        **kwargs,
    ):
        from src.utils.datetime_utils import get_current_utc_timestamp

        super().__init__(
            event_id=f"risk_limit_breached_{limit_type}_{current_value}",
            event_type=AnalyticsEventType.RISK_LIMIT_BREACHED,
            timestamp=get_current_utc_timestamp(),
            source_component=source_component,
            data={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "breach_percentage": breach_percentage,
            },
            **kwargs,
        )


class AnalyticsEventHandler(ABC):
    """Abstract base class for analytics event handlers."""

    @abstractmethod
    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle an analytics event."""
        pass

    @abstractmethod
    def can_handle(self, event_type: AnalyticsEventType) -> bool:
        """Check if this handler can handle the event type."""
        pass


class AnalyticsEventBus:
    """Event bus for analytics events."""

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[AnalyticsEventType, list[AnalyticsEventHandler]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._running = False
        self._event_history: list[AnalyticsEvent] = []
        self._max_history = 1000
        self.logger = get_logger(__name__)

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    def register_handler(
        self, event_type: AnalyticsEventType, handler: AnalyticsEventHandler
    ) -> None:
        """Register an event handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)

    def unregister_handler(
        self, event_type: AnalyticsEventType, handler: AnalyticsEventHandler
    ) -> None:
        """Unregister an event handler."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                if not self._handlers[event_type]:
                    del self._handlers[event_type]
            except ValueError:
                pass  # Handler not found

    async def publish_event(self, event: AnalyticsEvent) -> None:
        """Publish an event to the bus."""
        await self._event_queue.put(event)

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

    def get_event_history(
        self, event_type: AnalyticsEventType | None = None, limit: int = 100
    ) -> list[AnalyticsEvent]:
        """Get event history."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Get handlers for this event type
                handlers = self._handlers.get(event.event_type, [])

                # Process handlers concurrently
                if handlers:
                    tasks = [handler.handle_event(event) for handler in handlers]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Mark task as done
                self._event_queue.task_done()

            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue processing events
                self.logger.error(f"Error processing event: {e}")


class PortfolioEventHandler(AnalyticsEventHandler):
    """Event handler for portfolio analytics."""

    def __init__(self, portfolio_service: Any) -> None:
        """Initialize with portfolio service."""
        self.portfolio_service = portfolio_service
        self.logger = get_logger(__name__)

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle portfolio-related events."""
        try:
            if event.event_type == AnalyticsEventType.POSITION_UPDATED:
                if hasattr(self.portfolio_service, "handle_position_update"):
                    await self.portfolio_service.handle_position_update(event.data)

            elif event.event_type == AnalyticsEventType.PRICE_UPDATED:
                if hasattr(self.portfolio_service, "handle_price_update"):
                    await self.portfolio_service.handle_price_update(event.data)

        except Exception as e:
            self.logger.error(f"Error in portfolio event handler: {e}")

    def can_handle(self, event_type: AnalyticsEventType) -> bool:
        """Check if this handler can handle the event type."""
        return event_type in {
            AnalyticsEventType.POSITION_UPDATED,
            AnalyticsEventType.PRICE_UPDATED,
            AnalyticsEventType.BENCHMARK_UPDATED,
        }


class RiskEventHandler(AnalyticsEventHandler):
    """Event handler for risk monitoring."""

    def __init__(self, risk_service: Any) -> None:
        """Initialize with risk service."""
        self.risk_service = risk_service
        self.logger = get_logger(__name__)

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle risk-related events."""
        try:
            if event.event_type == AnalyticsEventType.POSITION_UPDATED:
                if hasattr(self.risk_service, "handle_position_update"):
                    await self.risk_service.handle_position_update(event.data)

            elif event.event_type == AnalyticsEventType.PRICE_UPDATED:
                if hasattr(self.risk_service, "handle_price_update"):
                    await self.risk_service.handle_price_update(event.data)

        except Exception as e:
            self.logger.error(f"Error in risk event handler: {e}")

    def can_handle(self, event_type: AnalyticsEventType) -> bool:
        """Check if this handler can handle the event type."""
        return event_type in {
            AnalyticsEventType.POSITION_UPDATED,
            AnalyticsEventType.PRICE_UPDATED,
            AnalyticsEventType.TRADE_EXECUTED,
        }


class AlertEventHandler(AnalyticsEventHandler):
    """Event handler for alert management."""

    def __init__(self, alert_service: Any) -> None:
        """Initialize with alert service."""
        self.alert_service = alert_service
        self.logger = get_logger(__name__)

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle alert-related events."""
        try:
            if event.event_type == AnalyticsEventType.RISK_LIMIT_BREACHED:
                if hasattr(self.alert_service, "handle_risk_limit_breach"):
                    await self.alert_service.handle_risk_limit_breach(event.data)

        except Exception as e:
            self.logger.error(f"Error in alert event handler: {e}")

    def can_handle(self, event_type: AnalyticsEventType) -> bool:
        """Check if this handler can handle the event type."""
        return event_type in {
            AnalyticsEventType.RISK_LIMIT_BREACHED,
            AnalyticsEventType.ERROR_OCCURRED,
        }


# Global event bus instance
_event_bus: AnalyticsEventBus | None = None


def get_event_bus() -> AnalyticsEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = AnalyticsEventBus()
    return _event_bus


async def publish_position_updated(position: Position, source_component: str) -> None:
    """Publish position updated event."""
    event = PositionUpdatedEvent(position, source_component)
    await get_event_bus().publish_event(event)


async def publish_trade_executed(trade: Trade, source_component: str) -> None:
    """Publish trade executed event."""
    event = TradeExecutedEvent(trade, source_component)
    await get_event_bus().publish_event(event)


async def publish_order_updated(order: Order, source_component: str) -> None:
    """Publish order updated event."""
    event = OrderUpdatedEvent(order, source_component)
    await get_event_bus().publish_event(event)


async def publish_price_updated(
    symbol: str, price: Decimal, timestamp: datetime, source_component: str
) -> None:
    """Publish price updated event."""
    event = PriceUpdatedEvent(symbol, price, timestamp, source_component)
    await get_event_bus().publish_event(event)


async def publish_risk_limit_breached(
    limit_type: str,
    current_value: float,
    limit_value: float,
    breach_percentage: float,
    source_component: str,
) -> None:
    """Publish risk limit breached event."""
    event = RiskLimitBreachedEvent(
        limit_type, current_value, limit_value, breach_percentage, source_component
    )
    await get_event_bus().publish_event(event)
