"""
Analytics Events - Simple event handling for analytics components.

Provides basic event notifications without complex orchestration.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.core.logging import get_logger
from src.core.types import Order, Position, Trade


class AnalyticsEventType(Enum):
    """Simple analytics event types."""

    POSITION_UPDATED = "position.modified"
    TRADE_EXECUTED = "trade.executed"
    ORDER_UPDATED = "order.filled"
    PRICE_UPDATED = "market.price_update"
    BENCHMARK_UPDATED = "benchmark.updated"
    PORTFOLIO_METRICS_CALCULATED = "portfolio.metrics_calculated"
    RISK_METRICS_CALCULATED = "risk.metrics_calculated"
    STRATEGY_METRICS_CALCULATED = "strategy.metrics_calculated"
    ALERT_GENERATED = "alert.fired"
    ALERT_ACKNOWLEDGED = "alert.acknowledged"
    ALERT_RESOLVED = "alert.resolved"
    RISK_LIMIT_BREACHED = "risk.limit_exceeded"
    SERVICE_STARTED = "system.startup"
    SERVICE_STOPPED = "system.shutdown"
    ERROR_OCCURRED = "system.error"
    HEALTH_CHECK_COMPLETED = "system.health_check"
    DATA_EXPORTED = "metric.exported"
    REPORT_GENERATED = "report.generated"
    RISK_ALERT = "risk.alert"


class AnalyticsEvent(BaseModel):
    """Simple analytics event."""

    event_type: AnalyticsEventType
    timestamp: datetime
    source: str
    data: dict[str, Any]


# Simple event handler base class
class EventHandler:
    """Base event handler."""

    def __init__(self, service):
        self.service = service
        self.logger = get_logger(self.__class__.__name__)

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
        self.logger = get_logger(self.__class__.__name__)

    def register_handler(self, event_type: AnalyticsEventType, handler: EventHandler) -> None:
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event: AnalyticsEvent) -> None:
        """Publish event to handlers."""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler.handle_event(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")

    async def start(self) -> None:
        """Start event bus."""
        pass

    async def stop(self) -> None:
        """Stop event bus."""
        pass


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
    """Publish position updated event."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.POSITION_UPDATED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"position": position.model_dump()},
    )
    await _event_bus.publish(event)


async def publish_trade_executed(trade: Trade, source: str) -> None:
    """Publish trade executed event."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.TRADE_EXECUTED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"trade": trade.model_dump()},
    )
    await _event_bus.publish(event)


async def publish_order_updated(order: Order, source: str) -> None:
    """Publish order updated event."""
    from src.utils.datetime_utils import get_current_utc_timestamp

    event = AnalyticsEvent(
        event_type=AnalyticsEventType.ORDER_UPDATED,
        timestamp=get_current_utc_timestamp(),
        source=source,
        data={"order": order.model_dump()},
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
