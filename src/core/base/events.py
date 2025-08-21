"""
Base event system implementation for the observer pattern.

This module provides the foundation for event-driven communication
in the trading bot system, implementing publish-subscribe patterns,
event routing, and asynchronous event handling.
"""

import asyncio
import re
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from re import Pattern
from typing import (
    Any,
    TypeVar,
)

from src.core.base.component import BaseComponent
from src.core.base.interfaces import EventEmitter, HealthStatus
from src.core.exceptions import EventError, EventHandlerError
from src.core.types.base import ConfigDict

# Type variables for event system
T = TypeVar("T")  # Event data type
E = TypeVar("E")  # Event type


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EventMetadata:
    """Metadata for event tracking."""

    event_id: str
    event_type: str
    timestamp: datetime
    priority: EventPriority
    source: str | None = None
    correlation_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventContext:
    """Context for event processing."""

    metadata: EventMetadata
    data: Any
    processed_by: set[str] = field(default_factory=set)
    processing_errors: list[Exception] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


class EventHandler:
    """Wrapper for event handler functions."""

    def __init__(
        self,
        handler: Callable[[Any], None] | Callable[[Any], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL,
        once: bool = False,
        filter_func: Callable[[Any], bool] | None = None,
        max_retries: int = 3,
        name: str | None = None,
    ):
        self.handler = handler
        self.priority = priority
        self.once = once
        self.filter_func = filter_func
        self.max_retries = max_retries
        self.name = name or f"{handler.__module__}.{handler.__name__}"
        self.call_count = 0
        self.error_count = 0
        self.last_called: datetime | None = None
        self.is_async = asyncio.iscoroutinefunction(handler)

    async def __call__(self, event_context: EventContext) -> bool:
        """
        Execute handler with event context.

        Returns:
            bool: True if handler executed successfully
        """
        try:
            # Apply filter if present
            if self.filter_func and not self.filter_func(event_context.data):
                return True  # Filtered out, but not an error

            self.call_count += 1
            self.last_called = datetime.utcnow()

            if self.is_async:
                await self.handler(event_context.data)
            else:
                self.handler(event_context.data)

            return True

        except Exception as e:
            self.error_count += 1
            event_context.processing_errors.append(e)
            return False


class BaseEventEmitter(BaseComponent, EventEmitter):
    r"""
    Base event emitter implementing the observer pattern.

    Provides:
    - Event publishing and subscription
    - Priority-based event handling
    - Asynchronous and synchronous handlers
    - Event filtering and routing
    - Error handling and retries
    - Event history and metrics
    - Weak reference management
    - Pattern-based subscriptions

    Example:
        ```python
        emitter = BaseEventEmitter(name="TradingEvents")


        # Subscribe to events
        @emitter.on("order.created")
        async def handle_order_created(data):
            print(f"Order created: {data}")


        # Subscribe with pattern
        emitter.on_pattern(r"order\.*", handle_order_events)

        # Emit events
        await emitter.emit("order.created", {"id": 123, "symbol": "BTCUSDT"})
        ```
    """

    def __init__(
        self,
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize base event emitter.

        Args:
            name: Event emitter name
            config: Event emitter configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name, config, correlation_id)

        # Event handlers storage
        self._handlers: dict[str, list[EventHandler]] = {}
        self._pattern_handlers: list[tuple[Pattern, EventHandler]] = []
        self._global_handlers: list[EventHandler] = []

        # Weak reference tracking for automatic cleanup
        self._handler_refs: set[weakref.ReferenceType] = set()

        # Event processing
        self._event_queue: asyncio.Queue[EventContext] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._max_queue_size = 10000
        self._batch_size = 100
        self._processing_interval = 0.01  # seconds

        # Event history and metrics
        self._event_history: list[EventContext] = []
        self._max_history_size = 1000
        self._event_metrics: dict[str, Any] = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "events_by_type": {},
            "handlers_by_event": {},
            "processing_times": [],
            "queue_size": 0,
            "last_event_time": None,
        }

        # Configuration
        self._enable_history = True
        self._enable_metrics = True
        self._enable_async_processing = True
        self._error_on_no_handlers = False
        self._default_priority = EventPriority.NORMAL

        # Threading
        self._lock = threading.RLock()

        self._logger.debug("Event emitter initialized", emitter=self._name)

    @property
    def event_metrics(self) -> dict[str, Any]:
        """Get event processing metrics."""
        metrics = self._event_metrics.copy()
        metrics["queue_size"] = self._event_queue.qsize()
        metrics["active_handlers"] = sum(len(handlers) for handlers in self._handlers.values())
        metrics["pattern_handlers"] = len(self._pattern_handlers)
        metrics["global_handlers"] = len(self._global_handlers)
        return metrics

    # Event Subscription
    def on(
        self,
        event: str,
        callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[Any], bool] | None = None,
        max_retries: int = 3,
    ) -> EventHandler:
        """
        Subscribe to event.

        Args:
            event: Event name to subscribe to
            callback: Handler function
            priority: Handler priority
            filter_func: Optional filter function
            max_retries: Maximum retry attempts on failure

        Returns:
            EventHandler wrapper for the callback
        """
        with self._lock:
            handler = EventHandler(
                handler=callback,
                priority=priority,
                filter_func=filter_func,
                max_retries=max_retries,
            )

            if event not in self._handlers:
                self._handlers[event] = []

            self._handlers[event].append(handler)

            # Sort handlers by priority (highest first)
            self._handlers[event].sort(key=lambda h: h.priority.value, reverse=True)

            self._logger.debug(
                "Event handler registered",
                emitter=self._name,
                event=event,
                handler=handler.name,
                priority=priority.name,
            )

            return handler

    def off(self, event: str, callback: Callable | EventHandler | None = None) -> None:
        """
        Unsubscribe from event.

        Args:
            event: Event name
            callback: Specific handler to remove, or None to remove all
        """
        with self._lock:
            if event not in self._handlers:
                return

            if callback is None:
                # Remove all handlers for this event
                removed_count = len(self._handlers[event])
                self._handlers[event].clear()

                self._logger.debug(
                    "All handlers removed for event",
                    emitter=self._name,
                    event=event,
                    count=removed_count,
                )
            else:
                # Remove specific handler
                if isinstance(callback, EventHandler):
                    target_handler = callback.handler
                else:
                    target_handler = callback

                original_count = len(self._handlers[event])
                self._handlers[event] = [
                    h for h in self._handlers[event] if h.handler != target_handler
                ]

                removed_count = original_count - len(self._handlers[event])

                self._logger.debug(
                    "Handler removed from event",
                    emitter=self._name,
                    event=event,
                    count=removed_count,
                )

    def once(
        self,
        event: str,
        callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[Any], bool] | None = None,
    ) -> EventHandler:
        """
        Subscribe to event for single execution.

        Args:
            event: Event name
            callback: Handler function
            priority: Handler priority
            filter_func: Optional filter function

        Returns:
            EventHandler wrapper
        """
        handler = EventHandler(
            handler=callback,
            priority=priority,
            once=True,
            filter_func=filter_func,
        )

        with self._lock:
            if event not in self._handlers:
                self._handlers[event] = []

            self._handlers[event].append(handler)
            self._handlers[event].sort(key=lambda h: h.priority.value, reverse=True)

        self._logger.debug(
            "One-time handler registered",
            emitter=self._name,
            event=event,
            handler=handler.name,
        )

        return handler

    def on_pattern(
        self,
        pattern: str | Pattern,
        callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[Any], bool] | None = None,
    ) -> EventHandler:
        """
        Subscribe to events matching a pattern.

        Args:
            pattern: Regex pattern to match event names
            callback: Handler function
            priority: Handler priority
            filter_func: Optional filter function

        Returns:
            EventHandler wrapper
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        handler = EventHandler(
            handler=callback,
            priority=priority,
            filter_func=filter_func,
        )

        with self._lock:
            self._pattern_handlers.append((pattern, handler))
            # Sort by priority
            self._pattern_handlers.sort(key=lambda x: x[1].priority.value, reverse=True)

        self._logger.debug(
            "Pattern handler registered",
            emitter=self._name,
            pattern=pattern.pattern,
            handler=handler.name,
        )

        return handler

    def on_global(
        self,
        callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[Any], bool] | None = None,
    ) -> EventHandler:
        """
        Subscribe to all events globally.

        Args:
            callback: Handler function
            priority: Handler priority
            filter_func: Optional filter function

        Returns:
            EventHandler wrapper
        """
        handler = EventHandler(
            handler=callback,
            priority=priority,
            filter_func=filter_func,
        )

        with self._lock:
            self._global_handlers.append(handler)
            self._global_handlers.sort(key=lambda h: h.priority.value, reverse=True)

        self._logger.debug(
            "Global handler registered",
            emitter=self._name,
            handler=handler.name,
        )

        return handler

    def remove_all_listeners(self, event: str | None = None) -> None:
        """
        Remove all listeners for event or all events.

        Args:
            event: Specific event name, or None for all events
        """
        with self._lock:
            if event is None:
                # Remove all listeners
                total_removed = sum(len(handlers) for handlers in self._handlers.values())
                total_removed += len(self._pattern_handlers)
                total_removed += len(self._global_handlers)

                self._handlers.clear()
                self._pattern_handlers.clear()
                self._global_handlers.clear()

                self._logger.info(
                    "All event handlers removed",
                    emitter=self._name,
                    count=total_removed,
                )
            else:
                # Remove listeners for specific event
                removed_count = len(self._handlers.get(event, []))
                self._handlers.pop(event, None)

                self._logger.info(
                    "Event handlers removed",
                    emitter=self._name,
                    event=event,
                    count=removed_count,
                )

    # Event Emission
    def emit(
        self,
        event: str,
        data: Any = None,
        priority: EventPriority | None = None,
        source: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit event synchronously.

        Args:
            event: Event name
            data: Event data
            priority: Event priority
            source: Event source identifier
            tags: Additional event tags
        """
        if self._enable_async_processing:
            # Queue for async processing
            asyncio.create_task(self.emit_async(event, data, priority, source, tags))
        else:
            # Process synchronously
            asyncio.create_task(self._emit_sync(event, data, priority, source, tags))

    async def emit_async(
        self,
        event: str,
        data: Any = None,
        priority: EventPriority | None = None,
        source: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit event asynchronously.

        Args:
            event: Event name
            data: Event data
            priority: Event priority
            source: Event source identifier
            tags: Additional event tags
        """
        await self._emit_sync(event, data, priority, source, tags)

    async def _emit_sync(
        self,
        event: str,
        data: Any = None,
        priority: EventPriority | None = None,
        source: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Internal synchronous emission implementation."""
        start_time = datetime.utcnow()

        # Create event context
        metadata = EventMetadata(
            event_id=f"{event}_{start_time.timestamp()}",
            event_type=event,
            timestamp=start_time,
            priority=priority or self._default_priority,
            source=source or self._name,
            correlation_id=self._correlation_id,
            tags=tags or {},
        )

        event_context = EventContext(metadata=metadata, data=data)

        try:
            self._logger.debug(
                "Emitting event",
                emitter=self._name,
                event=event,
                priority=metadata.priority.name,
                has_data=data is not None,
            )

            # Collect all matching handlers
            handlers = self._collect_handlers(event)

            if not handlers and self._error_on_no_handlers:
                raise EventError(f"No handlers found for event '{event}'")

            # Execute handlers
            success_count = 0
            for handler in handlers:
                try:
                    success = await self._execute_handler(handler, event_context)
                    if success:
                        success_count += 1
                        event_context.processed_by.add(handler.name)

                    # Remove one-time handlers
                    if handler.once:
                        self._remove_handler(handler)

                except Exception as e:
                    self._logger.error(
                        "Handler execution failed",
                        emitter=self._name,
                        event=event,
                        handler=handler.name,
                        error=str(e),
                    )
                    event_context.processing_errors.append(e)

            # Record metrics
            if self._enable_metrics:
                self._record_event_metrics(event, start_time, len(handlers), success_count)

            # Store in history
            if self._enable_history:
                self._add_to_history(event_context)

            self._logger.debug(
                "Event processing completed",
                emitter=self._name,
                event=event,
                handlers_executed=success_count,
                total_handlers=len(handlers),
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            self._logger.error(
                "Event emission failed",
                emitter=self._name,
                event=event,
                error=str(e),
                error_type=type(e).__name__,
            )

            if self._enable_metrics:
                self._event_metrics["failed_events"] += 1

            raise EventError(f"Failed to emit event '{event}': {e}")

    # Handler Management
    def _collect_handlers(self, event: str) -> list[EventHandler]:
        """Collect all handlers that should process this event."""
        handlers = []

        with self._lock:
            # Direct event handlers
            handlers.extend(self._handlers.get(event, []))

            # Pattern handlers
            for pattern, handler in self._pattern_handlers:
                if pattern.match(event):
                    handlers.append(handler)

            # Global handlers
            handlers.extend(self._global_handlers)

        # Sort by priority
        handlers.sort(key=lambda h: h.priority.value, reverse=True)
        return handlers

    async def _execute_handler(self, handler: EventHandler, event_context: EventContext) -> bool:
        """Execute single event handler with retry logic."""
        for attempt in range(handler.max_retries + 1):
            try:
                success = await handler(event_context)
                if success:
                    return True

                # If handler returned False, retry if attempts remain
                if attempt < handler.max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return False

            except Exception as e:
                if attempt < handler.max_retries:
                    self._logger.warning(
                        "Handler execution failed, retrying",
                        emitter=self._name,
                        handler=handler.name,
                        attempt=attempt + 1,
                        max_retries=handler.max_retries,
                        error=str(e),
                    )
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    # Final failure
                    raise EventHandlerError(
                        f"Handler {handler.name} failed after {handler.max_retries} retries"
                    )

        return False

    def _remove_handler(self, target_handler: EventHandler) -> None:
        """Remove handler from all collections."""
        with self._lock:
            # Remove from direct handlers
            for event, handlers in self._handlers.items():
                self._handlers[event] = [h for h in handlers if h != target_handler]

            # Remove from pattern handlers
            self._pattern_handlers = [
                (pattern, handler)
                for pattern, handler in self._pattern_handlers
                if handler != target_handler
            ]

            # Remove from global handlers
            self._global_handlers = [h for h in self._global_handlers if h != target_handler]

    # History and Metrics
    def _add_to_history(self, event_context: EventContext) -> None:
        """Add event to history."""
        self._event_history.append(event_context)

        # Maintain history size limit
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

    def _record_event_metrics(
        self,
        event: str,
        start_time: datetime,
        handlers_count: int,
        success_count: int,
    ) -> None:
        """Record event processing metrics."""
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        self._event_metrics["total_events"] += 1
        self._event_metrics["last_event_time"] = datetime.utcnow()

        if success_count > 0:
            self._event_metrics["successful_events"] += 1
        else:
            self._event_metrics["failed_events"] += 1

        # Per-event type metrics
        if event not in self._event_metrics["events_by_type"]:
            self._event_metrics["events_by_type"][event] = 0
        self._event_metrics["events_by_type"][event] += 1

        # Handler count tracking
        if event not in self._event_metrics["handlers_by_event"]:
            self._event_metrics["handlers_by_event"][event] = []
        self._event_metrics["handlers_by_event"][event].append(handlers_count)

        # Processing time tracking
        self._event_metrics["processing_times"].append(processing_time)

        # Keep only last 1000 processing times
        if len(self._event_metrics["processing_times"]) > 1000:
            self._event_metrics["processing_times"].pop(0)

    # Information and Management
    def get_event_history(
        self,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[EventContext]:
        """
        Get event processing history.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of event contexts
        """
        history = self._event_history.copy()

        if event_type:
            history = [ctx for ctx in history if ctx.metadata.event_type == event_type]

        if limit:
            history = history[-limit:]

        return history

    def get_handler_info(self) -> dict[str, Any]:
        """Get information about registered handlers."""
        with self._lock:
            return {
                "direct_handlers": {
                    event: [
                        {
                            "name": handler.name,
                            "priority": handler.priority.name,
                            "call_count": handler.call_count,
                            "error_count": handler.error_count,
                            "last_called": (
                                handler.last_called.isoformat() if handler.last_called else None
                            ),
                            "is_async": handler.is_async,
                        }
                        for handler in handlers
                    ]
                    for event, handlers in self._handlers.items()
                },
                "pattern_handlers": [
                    {
                        "pattern": pattern.pattern,
                        "handler": {
                            "name": handler.name,
                            "priority": handler.priority.name,
                            "call_count": handler.call_count,
                            "error_count": handler.error_count,
                        },
                    }
                    for pattern, handler in self._pattern_handlers
                ],
                "global_handlers": [
                    {
                        "name": handler.name,
                        "priority": handler.priority.name,
                        "call_count": handler.call_count,
                        "error_count": handler.error_count,
                    }
                    for handler in self._global_handlers
                ],
            }

    def get_events_summary(self) -> dict[str, Any]:
        """Get summary of event types and frequencies."""
        events_by_type = self._event_metrics["events_by_type"].copy()

        # Calculate statistics
        if events_by_type:
            total_events = sum(events_by_type.values())
            most_frequent = max(events_by_type.items(), key=lambda x: x[1])

            return {
                "total_events": total_events,
                "event_types_count": len(events_by_type),
                "most_frequent_event": {
                    "type": most_frequent[0],
                    "count": most_frequent[1],
                    "percentage": (most_frequent[1] / total_events) * 100,
                },
                "events_by_type": events_by_type,
            }

        return {
            "total_events": 0,
            "event_types_count": 0,
            "events_by_type": {},
        }

    # Health Check
    async def _health_check_internal(self) -> HealthStatus:
        """Event emitter health check."""
        try:
            # Check queue size
            if hasattr(self._event_queue, "qsize"):
                queue_size = self._event_queue.qsize()
                if queue_size > self._max_queue_size * 0.8:
                    return HealthStatus.DEGRADED

            # Check handler error rates
            for handlers in self._handlers.values():
                for handler in handlers:
                    if handler.call_count > 10:  # Only check handlers with significant calls
                        error_rate = handler.error_count / handler.call_count
                        if error_rate > 0.1:  # More than 10% errors
                            return HealthStatus.DEGRADED

            # Check processing times
            if self._event_metrics["processing_times"]:
                recent_times = self._event_metrics["processing_times"][-100:]
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time > 1.0:  # More than 1 second average
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(
                "Event emitter health check failed",
                emitter=self._name,
                error=str(e),
            )
            return HealthStatus.UNHEALTHY

    # Configuration
    def configure_processing(
        self,
        enable_async: bool = True,
        enable_history: bool = True,
        enable_metrics: bool = True,
        max_history_size: int = 1000,
        error_on_no_handlers: bool = False,
    ) -> None:
        """
        Configure event processing settings.

        Args:
            enable_async: Enable asynchronous processing
            enable_history: Enable event history tracking
            enable_metrics: Enable metrics collection
            max_history_size: Maximum history entries to keep
            error_on_no_handlers: Raise error if no handlers found
        """
        self._enable_async_processing = enable_async
        self._enable_history = enable_history
        self._enable_metrics = enable_metrics
        self._max_history_size = max_history_size
        self._error_on_no_handlers = error_on_no_handlers

        # Clear history if disabled
        if not enable_history:
            self._event_history.clear()

        self._logger.info(
            "Event processing configured",
            emitter=self._name,
            async_processing=enable_async,
            history=enable_history,
            metrics=enable_metrics,
            max_history_size=max_history_size,
        )

    # Metrics
    def get_metrics(self) -> dict[str, Any]:
        """Get combined component and event metrics."""
        metrics = super().get_metrics()
        metrics.update(self.event_metrics)
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        super().reset_metrics()
        self._event_metrics = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "events_by_type": {},
            "handlers_by_event": {},
            "processing_times": [],
            "queue_size": 0,
            "last_event_time": None,
        }

        # Reset handler metrics
        with self._lock:
            for handlers in self._handlers.values():
                for handler in handlers:
                    handler.call_count = 0
                    handler.error_count = 0
                    handler.last_called = None

    # Cleanup
    async def _do_stop(self) -> None:
        """Clean up event emitter resources."""
        # Stop processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Clear all handlers
        self.remove_all_listeners()

        # Clear history and metrics
        self._event_history.clear()
        self.reset_metrics()

        self._logger.debug("Event emitter stopped and cleaned up", emitter=self._name)
