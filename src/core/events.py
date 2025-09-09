"""
Event system for bot management coordination and monitoring.

This module provides event-driven architecture components for coordinating
bot operations and enabling real-time monitoring across the trading system.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.logging import get_logger

logger = get_logger(__name__)


class BotEventType(Enum):
    """Bot event types for coordination and monitoring."""

    # Lifecycle events
    BOT_CREATED = "bot_created"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    BOT_PAUSED = "bot_paused"
    BOT_RESUMED = "bot_resumed"
    BOT_ERROR = "bot_error"
    BOT_DELETED = "bot_deleted"

    # Status events
    BOT_STARTING = "bot_starting"
    BOT_STOPPING = "bot_stopping"
    BOT_INITIALIZING = "bot_initializing"

    # Performance events
    BOT_METRICS_UPDATE = "bot_metrics_update"
    BOT_PERFORMANCE_ALERT = "bot_performance_alert"

    # Risk events
    BOT_RISK_ALERT = "bot_risk_alert"
    BOT_RISK_LIMIT_EXCEEDED = "bot_risk_limit_exceeded"
    BOT_EMERGENCY_STOP = "bot_emergency_stop"

    # Trading events
    BOT_TRADE_EXECUTED = "bot_trade_executed"
    BOT_ORDER_PLACED = "bot_order_placed"
    BOT_ORDER_FILLED = "bot_order_filled"
    BOT_POSITION_UPDATE = "bot_position_update"


@dataclass
class BotEvent:
    """Bot event data structure."""

    event_type: BotEventType
    bot_id: str
    timestamp: datetime | None = None
    event_id: str | None = None
    data: dict[str, Any] | None = None
    source: str | None = None
    priority: str = "normal"  # low, normal, high, critical

    def __post_init__(self):
        """Initialize derived fields."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.event_id is None:
            self.event_id = str(uuid4())
        if self.data is None:
            self.data = {}


class EventHandler:
    """Base class for event handlers."""

    def __init__(self, handler_name: str):
        self.handler_name = handler_name
        self.logger = get_logger(f"EventHandler.{handler_name}")

    async def handle(self, event: BotEvent) -> None:
        """Handle an event. Override in subclasses."""
        raise NotImplementedError


class EventPublisher:
    """Event publisher for bot management coordination."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._handlers: dict[BotEventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._event_history: list[BotEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: BotEventType, handler: EventHandler) -> None:
        """Subscribe a handler to specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        self.logger.info(
            f"Handler {handler.handler_name} subscribed to {event_type.value}"
        )

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to all event types."""
        self._global_handlers.append(handler)
        self.logger.info(f"Handler {handler.handler_name} subscribed to all events")

    def unsubscribe(self, event_type: BotEventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def publish(self, event: BotEvent) -> None:
        """Publish an event to all subscribed handlers."""
        try:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            self.logger.debug(
                f"Publishing event {event.event_type.value}",
                bot_id=event.bot_id,
                event_id=event.event_id
            )

            # Get handlers for this event type
            handlers = []
            if event.event_type in self._handlers:
                handlers.extend(self._handlers[event.event_type])
            handlers.extend(self._global_handlers)

            # Execute handlers concurrently
            if handlers:
                tasks = [self._safe_handle(handler, event) for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_type.value}: {e}")

    async def _safe_handle(self, handler: EventHandler, event: BotEvent) -> None:
        """Safely execute handler with error isolation."""
        try:
            await handler.handle(event)
        except Exception as e:
            self.logger.error(
                f"Handler {handler.handler_name} failed to process "
                f"event {event.event_type.value}: {e}",
                bot_id=event.bot_id
            )

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: BotEventType | None = None,
        bot_id: str | None = None
    ) -> list[BotEvent]:
        """Get recent events with optional filtering."""
        events = self._event_history

        # Filter by event type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by bot ID
        if bot_id:
            events = [e for e in events if e.bot_id == bot_id]

        # Return most recent events
        return events[-limit:] if events else []


class AnalyticsEventHandler(EventHandler):
    """Event handler for analytics integration."""

    def __init__(self, analytics_service=None):
        super().__init__("AnalyticsEventHandler")
        self.analytics_service = analytics_service

    async def handle(self, event: BotEvent) -> None:
        """Handle events for analytics recording."""
        if not self.analytics_service:
            return

        try:
            # Record different types of events in analytics
            if event.event_type in [
                BotEventType.BOT_CREATED,
                BotEventType.BOT_STARTED,
                BotEventType.BOT_STOPPED
            ]:
                if hasattr(self.analytics_service, "record_bot_lifecycle_event"):
                    await self.analytics_service.record_bot_lifecycle_event(
                        event_type=event.event_type.value,
                        bot_id=event.bot_id,
                        timestamp=event.timestamp,
                        data=event.data
                    )

            elif event.event_type == BotEventType.BOT_METRICS_UPDATE:
                if hasattr(self.analytics_service, "record_bot_metrics"):
                    await self.analytics_service.record_bot_metrics(
                        bot_id=event.bot_id,
                        metrics=event.data,
                        timestamp=event.timestamp
                    )

            elif event.event_type in [
                BotEventType.BOT_RISK_ALERT,
                BotEventType.BOT_RISK_LIMIT_EXCEEDED
            ]:
                if hasattr(self.analytics_service, "record_risk_event"):
                    await self.analytics_service.record_risk_event(
                        event_type=event.event_type.value,
                        bot_id=event.bot_id,
                        risk_data=event.data,
                        timestamp=event.timestamp
                    )

        except Exception as e:
            self.logger.warning(f"Failed to record analytics event: {e}")


class RiskMonitoringEventHandler(EventHandler):
    """Event handler for risk monitoring."""

    def __init__(self, risk_service=None):
        super().__init__("RiskMonitoringEventHandler")
        self.risk_service = risk_service

    async def handle(self, event: BotEvent) -> None:
        """Handle events for risk monitoring."""
        if not self.risk_service:
            return

        try:
            # Monitor risk-related events
            if event.event_type == BotEventType.BOT_STARTED:
                # Validate initial risk on bot start
                if hasattr(self.risk_service, "validate_bot_startup"):
                    await self.risk_service.validate_bot_startup(event.bot_id)

            elif event.event_type == BotEventType.BOT_TRADE_EXECUTED:
                # Check risk after each trade
                if hasattr(self.risk_service, "post_trade_risk_check"):
                    await self.risk_service.post_trade_risk_check(
                        bot_id=event.bot_id,
                        trade_data=event.data
                    )

            elif event.event_type == BotEventType.BOT_RISK_ALERT:
                # Handle risk alerts
                if hasattr(self.risk_service, "handle_risk_alert"):
                    await self.risk_service.handle_risk_alert(
                        bot_id=event.bot_id,
                        alert_data=event.data
                    )

        except Exception as e:
            self.logger.warning(f"Risk monitoring event handling failed: {e}")


# Global event publisher instance
_global_publisher = None


def get_event_publisher() -> EventPublisher:
    """Get the global event publisher instance."""
    global _global_publisher
    if _global_publisher is None:
        _global_publisher = EventPublisher()
    return _global_publisher


def setup_bot_management_events(analytics_service=None, risk_service=None) -> EventPublisher:
    """Setup event handlers for bot management."""
    publisher = get_event_publisher()

    # Setup analytics event handler
    if analytics_service:
        analytics_handler = AnalyticsEventHandler(analytics_service)
        publisher.subscribe_all(analytics_handler)

    # Setup risk monitoring event handler
    if risk_service:
        risk_handler = RiskMonitoringEventHandler(risk_service)
        publisher.subscribe_all(risk_handler)

    return publisher
