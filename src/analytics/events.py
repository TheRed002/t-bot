"""
Analytics Events - Simple event handling for analytics components.

Provides basic event notifications without complex orchestration.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.core.event_constants import (
    AlertEvents,
    MarketDataEvents,
    MetricEvents,
    OrderEvents,
    PositionEvents,
    RiskEvents,
    SystemEvents,
    TradeEvents,
)
# Robust logger function with fallback for test suite compatibility
def get_logger_safe(name):
    """Get logger with robust fallback for test environments."""
    try:
        from src.core.logging import get_logger
        return get_logger(name)
    except ImportError:
        import logging
        return logging.getLogger(name)
from src.core.types import Order, Position, Trade


class AnalyticsEventType(Enum):
    """Simple analytics event types using core event constants."""

    POSITION_UPDATED = PositionEvents.MODIFIED
    TRADE_EXECUTED = TradeEvents.EXECUTED
    ORDER_UPDATED = OrderEvents.FILLED
    PRICE_UPDATED = MarketDataEvents.PRICE_UPDATE
    BENCHMARK_UPDATED = "benchmark.updated"
    PORTFOLIO_METRICS_CALCULATED = "portfolio.metrics_calculated"
    RISK_METRICS_CALCULATED = "risk.metrics_calculated"
    STRATEGY_METRICS_CALCULATED = "strategy.metrics_calculated"
    ALERT_GENERATED = AlertEvents.FIRED
    ALERT_ACKNOWLEDGED = AlertEvents.ACKNOWLEDGED
    ALERT_RESOLVED = AlertEvents.RESOLVED
    RISK_LIMIT_BREACHED = RiskEvents.LIMIT_EXCEEDED
    SERVICE_STARTED = SystemEvents.STARTUP
    SERVICE_STOPPED = SystemEvents.SHUTDOWN
    ERROR_OCCURRED = SystemEvents.COMPONENT_ERROR
    HEALTH_CHECK_COMPLETED = SystemEvents.HEALTH_CHECK_FAILED  # Reuse closest available constant
    DATA_EXPORTED = MetricEvents.EXPORTED
    REPORT_GENERATED = MetricEvents.AGGREGATED  # Use available metric event for report generation
    RISK_ALERT = RiskEvents.THRESHOLD_BREACH  # Use centralized risk event


class AnalyticsEvent(BaseModel):
    """Analytics event aligned with core event patterns."""

    event_type: AnalyticsEventType
    timestamp: datetime
    source: str
    data: dict[str, Any]
    processing_mode: str = "stream"  # Align with core default
    message_pattern: str = "pub_sub"  # Consistent with core messaging
    data_format: str = "analytics_event_v1"  # Versioned format for consistency
    boundary_crossed: bool = True  # Flag for module boundary crossing


# Simple event handler base class
class EventHandler:
    """Base event handler."""

    def __init__(self, service):
        self.service = service
        self.logger = get_logger_safe(self.__class__.__name__)

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle analytics event."""
        pass


class PortfolioEventHandler(EventHandler):
    """Handle portfolio-related events."""

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle portfolio events."""
        if not self.service:
            return

        try:
            if event.event_type == AnalyticsEventType.POSITION_UPDATED:
                position_data = event.data.get("position")
                if position_data and hasattr(self.service, "update_position"):
                    # Simple direct call
                    position = Position(**position_data)
                    self.service.update_position(position)
        except Exception as e:
            self.logger.error(f"Error handling portfolio event: {e}")


class RiskEventHandler(EventHandler):
    """Handle risk-related events."""

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle risk events."""
        if not self.service:
            return

        try:
            if event.event_type == AnalyticsEventType.POSITION_UPDATED:
                position_data = event.data.get("position")
                if position_data and hasattr(self.service, "update_position"):
                    position = Position(**position_data)
                    self.service.update_position(position)
        except Exception as e:
            self.logger.error(f"Error handling risk event: {e}")


class AlertEventHandler(EventHandler):
    """Handle alert-related events."""

    async def handle_event(self, event: AnalyticsEvent) -> None:
        """Handle alert events."""
        if not self.service:
            return

        try:
            if event.event_type == AnalyticsEventType.RISK_ALERT:
                alert_data = event.data
                if hasattr(self.service, "handle_alert"):
                    self.service.handle_alert(alert_data)
        except Exception as e:
            self.logger.error(f"Error handling alert event: {e}")


# Simple event bus - no complex orchestration
class SimpleEventBus:
    """Simple event bus for analytics."""

    def __init__(self):
        self.handlers: dict[AnalyticsEventType, list[EventHandler]] = {}
        self.logger = get_logger_safe(self.__class__.__name__)

    def register_handler(self, event_type: AnalyticsEventType, handler: EventHandler) -> None:
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event: AnalyticsEvent) -> None:
        """Publish event to handlers using consistent patterns aligned with core."""
        try:
            # Apply consistent data transformation aligned with core patterns
            self._apply_core_alignment(event)

            handlers = self.handlers.get(event.event_type, [])

            # Process handlers using consistent pub/sub pattern
            if event.processing_mode == "batch":
                await self._process_handlers_batch(handlers, event)
            else:
                await self._process_handlers_stream(handlers, event)

        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_type.value}: {e}")
            self._propagate_event_error_consistently(e, "publish", event.event_type.value)

    async def start(self) -> None:
        """Start event bus with core-aligned initialization."""
        self.logger.info("Analytics event bus starting with core alignment")

    async def stop(self) -> None:
        """Stop event bus with core-aligned cleanup."""
        self.logger.info("Analytics event bus stopping")

    def _apply_core_alignment(self, event: AnalyticsEvent) -> None:
        """Apply core module alignment to event data."""
        if isinstance(event.data, dict):
            # Ensure consistent metadata aligned with core patterns
            event.data.update({
                "processing_mode": event.processing_mode,
                "message_pattern": event.message_pattern,
                "data_format": event.data_format,
                "boundary_crossed": event.boundary_crossed,
                "source_module": "analytics",
            })

    async def _process_handlers_stream(self, handlers: list, event: AnalyticsEvent) -> None:
        """Process handlers in stream mode aligned with core patterns."""
        import asyncio
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_handle(handler, event))
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_handlers_batch(self, handlers: list, event: AnalyticsEvent) -> None:
        """Process handlers in batch mode aligned with core patterns."""
        for handler in handlers:
            await self._safe_handle(handler, event)

    async def _safe_handle(self, handler, event: AnalyticsEvent) -> None:
        """Safely handle event with error isolation."""
        try:
            await handler.handle_event(event)
        except Exception as e:
            self.logger.error(f"Handler failed for event {event.event_type.value}: {e}")

    def _propagate_event_error_consistently(self, error: Exception, operation: str, event_type: str) -> None:
        """Propagate event errors consistently with core patterns aligned with database module."""
        from datetime import timezone

        from src.error_handling.propagation_utils import (
            ProcessingStage,
            PropagationMethod,
            add_propagation_step,
        )

        error_metadata = {
            "error_type": type(error).__name__,
            "operation": operation,
            "event_type": event_type,
            "component": "AnalyticsEventBus",
            "processing_mode": "stream",
            "message_pattern": "pub_sub",
            "boundary_crossed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add propagation step for consistency with database module patterns
        try:
            error_data = add_propagation_step(
                error_metadata,
                source_module="analytics",
                target_module="error_handling",
                method=PropagationMethod.DIRECT_CALL,
                stage=ProcessingStage.ERROR_PROPAGATION
            )
            self.logger.error(f"Analytics event error in {operation} for {event_type}: {error}", extra=error_data)
        except Exception as propagation_error:
            # Fallback if propagation utilities fail
            self.logger.error(f"Analytics event error in {operation} for {event_type}: {error}", extra=error_metadata)
            self.logger.warning(f"Error propagation failed: {propagation_error}")


# Global event bus instance
_event_bus = SimpleEventBus()


# Aliases for backward compatibility
AnalyticsEventBus = SimpleEventBus
AnalyticsEventHandler = EventHandler

# Event aliases for compatibility (if tests expect specific event classes)
PositionUpdatedEvent = AnalyticsEvent
OrderUpdatedEvent = AnalyticsEvent
PriceUpdatedEvent = AnalyticsEvent
TradeExecutedEvent = AnalyticsEvent
RiskAlertEvent = AnalyticsEvent
RiskLimitBreachedEvent = AnalyticsEvent


def get_event_bus() -> SimpleEventBus:
    """Get global event bus."""
    return _event_bus


# Simple publish functions
async def publish_position_updated(position: Position, source: str) -> None:
    """Publish position updated event with core-aligned patterns."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.POSITION_UPDATED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"position": position.model_dump()},
        processing_mode="stream",  # Real-time position updates
        message_pattern="pub_sub",  # Consistent with core
        data_format="analytics_position_event_v1",
        boundary_crossed=True,
    )
    await _event_bus.publish(event)


async def publish_trade_executed(trade: Trade, source: str) -> None:
    """Publish trade executed event with core-aligned patterns."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.TRADE_EXECUTED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"trade": trade.model_dump()},
        processing_mode="stream",  # Real-time trade updates
        message_pattern="pub_sub",  # Consistent with core
        data_format="analytics_trade_event_v1",
        boundary_crossed=True,
    )
    await _event_bus.publish(event)


async def publish_order_updated(order: Order, source: str) -> None:
    """Publish order updated event with core-aligned patterns."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.ORDER_UPDATED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"order": order.model_dump()},
        processing_mode="stream",  # Real-time order updates
        message_pattern="pub_sub",  # Consistent with core
        data_format="analytics_order_event_v1",
        boundary_crossed=True,
    )
    await _event_bus.publish(event)


async def publish_price_updated(
    symbol: str, price: Decimal, timestamp: datetime, source: str
) -> None:
    """Publish price updated event."""
    event = AnalyticsEvent(
        event_type=AnalyticsEventType.PRICE_UPDATED,
        timestamp=timestamp,
        source=source,
        data={"symbol": symbol, "price": str(price), "timestamp": timestamp.isoformat()},
    )
    await _event_bus.publish(event)


async def publish_risk_limit_breached(
    symbol: str, limit_type: str, current_value: Decimal, limit_value: Decimal, source: str
) -> None:
    """Publish risk limit breached event."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.RISK_LIMIT_BREACHED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={
            "symbol": symbol,
            "limit_type": limit_type,
            "current_value": str(current_value),
            "limit_value": str(limit_value),
            "severity": "HIGH",
        },
    )
    await _event_bus.publish(event)
