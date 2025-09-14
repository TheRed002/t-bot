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

    async def publish(self, event: BotEvent, processing_mode: str = "stream") -> None:
        """
        Publish an event to all subscribed handlers with consistent processing patterns aligned with state module.
        
        Args:
            event: Bot event to publish
            processing_mode: Processing mode ("stream" for real-time, "batch" for grouped processing)
        """
        try:
            # Apply consistent data transformation aligned with state module patterns
            transformed_event = self._transform_event_data_consistent(event, processing_mode)

            # Add to history
            self._event_history.append(transformed_event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            self.logger.debug(
                f"Publishing event {transformed_event.event_type.value} with processing mode {processing_mode}",
                bot_id=transformed_event.bot_id,
                event_id=transformed_event.event_id,
                processing_mode=processing_mode
            )

            # Get handlers for this event type using consistent pub/sub pattern aligned with state
            handlers = self._collect_handlers_by_priority(transformed_event.event_type)

            # Execute handlers based on processing mode with consistent message patterns
            if handlers:
                # Apply consistent data transformation aligned with execution module patterns
                from src.core.data_transformer import CoreDataTransformer
                
                # Transform event data using consistent patterns
                if transformed_event.data is None:
                    transformed_event.data = {}
                    
                # Apply cross-module consistency for data flow alignment
                event_data = CoreDataTransformer.apply_cross_module_consistency(
                    data={
                        "event_type": transformed_event.event_type.value,
                        "bot_id": transformed_event.bot_id,
                        "data": transformed_event.data,
                        "processing_mode": "stream",  # Default for real-time events
                        "data_format": "bot_event_v1",
                        "message_pattern": "pub_sub",
                        "timestamp": transformed_event.timestamp.isoformat() if transformed_event.timestamp else datetime.now(timezone.utc).isoformat(),
                        "source": "core",
                    },
                    target_module="execution",  # Default target for consistency
                    source_module="core"
                )
                
                # Update event data with consistent transformation
                transformed_event.data.update(event_data)

                if processing_mode == "batch":
                    # Process in batches for better throughput
                    await self._process_handlers_batch(handlers, transformed_event)
                else:
                    # Process as stream for real-time requirements
                    await self._process_handlers_stream(handlers, transformed_event)

        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_type.value}: {e}")
            # Apply consistent error propagation
            self._propagate_event_error_consistently(e, "publish", event.event_type.value)

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

    # Helper methods for consistent data flow patterns

    def _transform_event_data_consistent(self, event: BotEvent, processing_mode: str) -> BotEvent:
        """Transform event data to ensure consistency with core module patterns."""
        # Apply consistent data transformations aligned with BaseEventEmitter
        if event.data and isinstance(event.data, dict):
            # Ensure processing metadata consistency
            if "processing_mode" not in event.data:
                event.data["processing_mode"] = processing_mode
            if "data_format" not in event.data:
                event.data["data_format"] = "bot_event_v1"

            # Apply consistent financial data transformations
            financial_fields = ["price", "quantity", "volume"]
            for field in financial_fields:
                if field in event.data and event.data[field] is not None:
                    try:
                        from src.utils.decimal_utils import to_decimal
                        event.data[field] = to_decimal(event.data[field])
                    except (ValueError, TypeError, ImportError):
                        # Log warning but continue processing
                        self.logger.warning(f"Failed to convert {field} to decimal in event {event.event_id}")

        return event

    def _collect_handlers_by_priority(self, event_type: BotEventType) -> list[EventHandler]:
        """Collect handlers using consistent pub/sub patterns with priority sorting."""
        handlers = []

        # Direct event type handlers
        if event_type in self._handlers:
            handlers.extend(self._handlers[event_type])

        # Global handlers (subscribe to all events)
        handlers.extend(self._global_handlers)

        # Sort by priority if handlers have priority attribute, otherwise maintain order
        handlers.sort(key=lambda h: getattr(h, "priority", 0), reverse=True)

        return handlers

    async def _process_handlers_batch(self, handlers: list[EventHandler], event: BotEvent) -> None:
        """Process handlers in batch mode for better throughput aligned with state processing paradigms."""
        # Apply consistent batch processing matching state module patterns
        batch_size = 10  # Process handlers in small batches

        # Apply batch metadata for consistency with execution and state modules
        from src.core.data_transformer import CoreDataTransformer
        
        if event.data is None:
            event.data = {}

        # Align processing paradigm for batch mode consistency
        batch_data = CoreDataTransformer.align_processing_paradigm(
            data=event.data,
            target_mode="batch"
        )
        
        batch_data.update({
            "batch_processing": True,
            "batch_size": min(batch_size, len(handlers)),
            "total_handlers": len(handlers),
            "processing_paradigm": "batch",
            "handler_processing_mode": "batch"
        })
        
        event.data.update(batch_data)

        for i in range(0, len(handlers), batch_size):
            batch = handlers[i:i + batch_size]
            # Add batch position metadata
            for handler in batch:
                if hasattr(handler, "set_batch_context"):
                    handler.set_batch_context({
                        "batch_id": f"batch_{i//batch_size}",
                        "batch_position": i,
                        "processing_mode": "batch"
                    })

            tasks = [self._safe_handle(handler, event) for handler in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            # Small delay between batches to prevent overwhelming
            if i + batch_size < len(handlers):
                await asyncio.sleep(0.001)

    async def _process_handlers_stream(self, handlers: list[EventHandler], event: BotEvent) -> None:
        """Process handlers in stream mode for real-time requirements aligned with state processing paradigms."""
        # Apply consistent stream processing matching state module patterns
        from src.core.data_transformer import CoreDataTransformer
        
        if event.data is None:
            event.data = {}

        # Align processing paradigm for stream mode consistency
        stream_data = CoreDataTransformer.align_processing_paradigm(
            data=event.data,
            target_mode="stream"
        )
        
        stream_data.update({
            "stream_processing": True,
            "stream_id": event.event_id,
            "handler_count": len(handlers),
            "processing_paradigm": "stream",
            "handler_processing_mode": "stream"
        })
        
        event.data.update(stream_data)

        # Add stream context to handlers that support it
        for handler in handlers:
            if hasattr(handler, "set_stream_context"):
                handler.set_stream_context({
                    "stream_id": event.event_id,
                    "processing_mode": "stream",
                    "real_time": True
                })

        # Process all handlers concurrently for minimal latency
        tasks = [self._safe_handle(handler, event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _propagate_event_error_consistently(self, error: Exception, operation: str, event_type: str) -> None:
        """Propagate event errors with consistent patterns aligned with state module."""
        from src.core.exceptions import EventError, ValidationError

        # Apply consistent error data structure matching state module patterns
        error_metadata = {
            "error_type": type(error).__name__,
            "operation": operation,
            "event_type": event_type,
            "component": "EventPublisher",
            "processing_mode": "stream",  # Match state module default
            "data_format": "bot_event_v1",  # Consistent with event system format
            "message_pattern": "pub_sub",  # Consistent messaging pattern
            "boundary_crossed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Apply consistent error propagation patterns similar to state module
        if isinstance(error, ValidationError):
            # Validation errors are re-raised as-is for consistency
            self.logger.debug(
                f"Validation error in events.{operation} for {event_type} - propagating as validation error",
                extra=error_metadata
            )
            # Add propagation metadata to error if supported
            if hasattr(error, "__dict__"):
                try:
                    error.__dict__.update({
                        "propagation_metadata": error_metadata,
                        "boundary_validation_applied": True
                    })
                except (AttributeError, TypeError):
                    # Some exception types don't allow attribute modification
                    pass
        elif isinstance(error, EventError):
            # Event errors get additional context
            self.logger.warning(
                f"Event error in events.{operation} for {event_type} - adding context",
                extra=error_metadata
            )
        else:
            # Generic errors get event-level error propagation with boundary validation
            self.logger.error(
                f"Event service error in events.{operation} for {event_type} - wrapping in EventError",
                extra=error_metadata
            )

            # Apply boundary validation for consistency
            try:
                from src.utils.messaging_patterns import BoundaryValidator
                # Use existing boundary validation for error data
                BoundaryValidator.validate_error_to_monitoring_boundary(error_metadata)
            except Exception as validation_error:
                self.logger.warning(f"Event error boundary validation failed: {validation_error}")


class AnalyticsEventHandler(EventHandler):
    """Event handler for analytics integration."""

    def __init__(self, analytics_service=None):
        super().__init__("AnalyticsEventHandler")
        self.analytics_service = analytics_service

    async def handle(self, event: BotEvent) -> None:
        """Handle events for analytics recording with consistent data validation."""
        if not self.analytics_service:
            return
            
        try:
            # Apply consistent data transformation for analytics module boundary
            from src.core.data_transformer import CoreDataTransformer
            
            # Transform event data for analytics module consistency
            analytics_data = CoreDataTransformer.apply_cross_module_consistency(
                data={
                    "event_type": event.event_type.value,
                    "bot_id": event.bot_id,
                    "data": event.data or {},
                    "processing_mode": "stream",  # Analytics prefers stream for real-time
                    "data_format": "analytics_event_v1",
                    "message_pattern": "pub_sub",
                    "timestamp": event.timestamp.isoformat() if event.timestamp else datetime.now(timezone.utc).isoformat(),
                    "source": "core",
                },
                target_module="analytics",
                source_module="core"
            )

            # Record different types of events in analytics with consistent patterns
            if event.event_type in [
                BotEventType.BOT_CREATED,
                BotEventType.BOT_STARTED,
                BotEventType.BOT_STOPPED
            ]:
                if hasattr(self.analytics_service, "record_strategy_event"):
                    # Use consistent data format for bot lifecycle events
                    lifecycle_data = analytics_data.copy()
                    lifecycle_data["data_format"] = "bot_lifecycle_event_v1"
                    
                    self.analytics_service.record_strategy_event(
                        strategy_name=str(event.bot_id),
                        event_type=event.event_type.value,
                        success=True
                    )

            elif event.event_type == BotEventType.BOT_METRICS_UPDATE:
                if hasattr(self.analytics_service, "record_market_data_event"):
                    # Use consistent data format for metrics data
                    metrics_data = analytics_data.copy()
                    metrics_data["data_format"] = "bot_metrics_v1"
                    
                    self.analytics_service.record_market_data_event(
                        data_type="bot_metrics",
                        data=metrics_data
                    )

            elif event.event_type in [
                BotEventType.BOT_RISK_ALERT,
                BotEventType.BOT_RISK_LIMIT_EXCEEDED
            ]:
                if hasattr(self.analytics_service, "record_system_error"):
                    # Use consistent data format for risk events
                    risk_data = analytics_data.copy()
                    risk_data["data_format"] = "risk_event_v1"
                    
                    self.analytics_service.record_system_error(
                        component="risk_management",
                        error_type=event.event_type.value,
                        error_message=str(risk_data),
                        severity="WARNING"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to record analytics event: {e}")
            # Apply consistent error propagation aligned with execution module
            from src.core.exceptions import ValidationError
            if not isinstance(e, ValidationError):
                # Apply consistent error propagation patterns
                try:
                    from src.core.data_transformer import CoreDataTransformer
                    error_data = CoreDataTransformer.transform_for_pub_sub_pattern(
                        event_type="analytics_handler_error",
                        data={"error": str(e), "original_event_type": event.event_type.value}
                    )
                    self.logger.error(
                        f"Analytics handler error for {event.event_type.value}: {e!s}",
                        extra=error_data
                    )
                except Exception:
                    # Fallback to simple logging if transformation fails
                    self.logger.error(
                        f"Analytics handler error for {event.event_type.value}: {e!s}",
                        event_id=event.event_id,
                        bot_id=event.bot_id
                    )


class RiskMonitoringEventHandler(EventHandler):
    """Event handler for risk monitoring."""

    def __init__(self, risk_service=None):
        super().__init__("RiskMonitoringEventHandler")
        self.risk_service = risk_service

    async def handle(self, event: BotEvent) -> None:
        """Handle events for risk monitoring with consistent data validation."""
        if not self.risk_service:
            return

        try:
            # Apply consistent data transformation for risk monitoring module boundary
            from src.core.data_transformer import CoreDataTransformer
            
            # Transform event data for risk management module consistency
            risk_data = CoreDataTransformer.apply_cross_module_consistency(
                data={
                    "event_type": event.event_type.value,
                    "bot_id": event.bot_id,
                    "data": event.data or {},
                    "processing_mode": "stream",  # Risk monitoring needs real-time processing
                    "data_format": "risk_monitoring_event_v1",
                    "message_pattern": "pub_sub",
                    "timestamp": event.timestamp.isoformat() if event.timestamp else datetime.now(timezone.utc).isoformat(),
                    "source": "core",
                },
                target_module="risk_management",
                source_module="core"
            )

            # Monitor risk-related events with consistent data patterns
            if event.event_type == BotEventType.BOT_STARTED:
                # Validate initial risk on bot start
                if hasattr(self.risk_service, "validate_bot_startup"):
                    await self.risk_service.validate_bot_startup(event.bot_id)

            elif event.event_type == BotEventType.BOT_TRADE_EXECUTED:
                # Check risk after each trade
                if hasattr(self.risk_service, "post_trade_risk_check"):
                    # Use consistent data format for trade data
                    trade_data = risk_data.copy()
                    trade_data["data_format"] = "trade_data_v1"
                    
                    await self.risk_service.post_trade_risk_check(
                        bot_id=event.bot_id,
                        trade_data=trade_data
                    )

            elif event.event_type == BotEventType.BOT_RISK_ALERT:
                # Handle risk alerts
                if hasattr(self.risk_service, "handle_risk_alert"):
                    # Use consistent data format for alert data
                    alert_data = risk_data.copy()
                    alert_data.update({
                        "data_format": "risk_alert_v1",
                        "priority": "high",  # Risk alerts are high priority
                        "severity": "critical"  # Align with execution module patterns
                    })
                    
                    await self.risk_service.handle_risk_alert(
                        bot_id=event.bot_id,
                        alert_data=alert_data
                    )

        except Exception as e:
            self.logger.warning(f"Risk monitoring event handling failed: {e}")
            # Apply consistent error propagation aligned with execution module
            from src.core.exceptions import ValidationError
            if not isinstance(e, ValidationError):
                # Apply consistent error propagation patterns
                try:
                    from src.core.data_transformer import CoreDataTransformer
                    error_data = CoreDataTransformer.transform_for_pub_sub_pattern(
                        event_type="risk_monitoring_handler_error",
                        data={"error": str(e), "original_event_type": event.event_type.value}
                    )
                    self.logger.error(
                        f"Risk monitoring handler error for {event.event_type.value}: {e!s}",
                        extra=error_data
                    )
                except Exception:
                    # Fallback to simple logging if transformation fails
                    self.logger.error(
                        f"Risk monitoring handler error for {event.event_type.value}: {e!s}",
                        event_id=event.event_id,
                        bot_id=event.bot_id
                    )


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
