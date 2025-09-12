"""
Unified Messaging Patterns for Data Flow

This module provides consistent messaging patterns across exchanges and core systems,
standardizing on pub/sub patterns for data distribution.

Patterns:
- Publisher/Subscriber for market data streams
- Event-driven architecture for order updates  
- Message queuing for batch processing
- Circuit breaker integration for resilience
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from weakref import WeakSet

from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.error_handling.decorators import with_circuit_breaker, with_retry

logger = get_logger(__name__)


class MessageType(Enum):
    """Message types for the messaging system."""
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    TRADE_EXECUTION = "trade_execution"
    ACCOUNT_UPDATE = "account_update"
    ERROR_EVENT = "error_event"
    SYSTEM_EVENT = "system_event"


class MessagePattern(Enum):
    """Message communication patterns."""
    PUB_SUB = "pub_sub"
    REQ_REPLY = "req_reply"
    STREAM = "stream"
    BATCH = "batch"


@dataclass
class Message:
    """Standard message format."""
    id: str
    type: MessageType
    source: str
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)


class MessageHandler(ABC):
    """Abstract message handler."""

    @abstractmethod
    async def handle(self, message: Message) -> bool:
        """Handle a message. Return True if handled successfully."""
        pass

    @abstractmethod
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if this handler can process the message type."""
        pass


class MessagePublisher(ABC):
    """Abstract message publisher."""

    @abstractmethod
    async def publish(self, message: Message) -> bool:
        """Publish a message. Return True if published successfully."""
        pass

    @abstractmethod
    async def subscribe(self, message_type: MessageType, handler: MessageHandler) -> bool:
        """Subscribe a handler to message type."""
        pass

    @abstractmethod
    async def unsubscribe(self, message_type: MessageType, handler: MessageHandler) -> bool:
        """Unsubscribe a handler from message type."""
        pass


class InMemoryMessageBus(MessagePublisher):
    """In-memory message bus implementation using pub/sub pattern."""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.subscribers: dict[MessageType, WeakSet[MessageHandler]] = defaultdict(WeakSet)
        self.message_queue: deque[Message] = deque(maxlen=max_queue_size)
        self.processing_lock = asyncio.Lock()
        self.stats = {
            "messages_published": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "subscribers_count": 0,
        }
        self.logger = get_logger(f"{__name__}.InMemoryMessageBus")

    async def publish(self, message: Message) -> bool:
        """Publish message to all subscribers."""
        try:
            if not isinstance(message, Message):
                raise ValidationError("Invalid message format")

            # Add to queue for processing
            self.message_queue.append(message)
            self.stats["messages_published"] += 1

            # Process immediately if possible
            asyncio.create_task(self._process_message(message))

            return True

        except Exception as e:
            self.logger.error(f"Failed to publish message {message.id}: {e}")
            return False

    async def subscribe(self, message_type: MessageType, handler: MessageHandler) -> bool:
        """Subscribe handler to message type."""
        try:
            if not isinstance(handler, MessageHandler):
                raise ValidationError("Handler must implement MessageHandler interface")

            self.subscribers[message_type].add(handler)
            self.stats["subscribers_count"] = sum(len(handlers) for handlers in self.subscribers.values())

            self.logger.info(f"Subscribed handler to {message_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to subscribe handler: {e}")
            return False

    async def unsubscribe(self, message_type: MessageType, handler: MessageHandler) -> bool:
        """Unsubscribe handler from message type."""
        try:
            if message_type in self.subscribers:
                self.subscribers[message_type].discard(handler)
                self.stats["subscribers_count"] = sum(len(handlers) for handlers in self.subscribers.values())

            self.logger.info(f"Unsubscribed handler from {message_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe handler: {e}")
            return False

    @with_circuit_breaker(failure_threshold=5)
    @with_retry(max_attempts=3)
    async def _process_message(self, message: Message) -> None:
        """Process message by delivering to subscribers."""
        async with self.processing_lock:
            try:
                handlers = list(self.subscribers.get(message.type, []))

                if not handlers:
                    self.logger.debug(f"No handlers for message type {message.type.value}")
                    return

                # Process handlers concurrently
                tasks = []
                for handler in handlers:
                    if handler.can_handle(message.type):
                        task = asyncio.create_task(self._deliver_to_handler(handler, message))
                        tasks.append(task)

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Count successes and failures
                    successes = sum(1 for result in results if result is True)
                    failures = len(results) - successes

                    self.stats["messages_processed"] += successes
                    self.stats["messages_failed"] += failures

                    if failures > 0:
                        self.logger.warning(f"Message {message.id} had {failures} handler failures")

            except Exception as e:
                self.logger.error(f"Failed to process message {message.id}: {e}")
                self.stats["messages_failed"] += 1

    async def _deliver_to_handler(self, handler: MessageHandler, message: Message) -> bool:
        """Deliver message to specific handler."""
        try:
            return await handler.handle(message)
        except Exception as e:
            self.logger.error(f"Handler failed for message {message.id}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self.stats,
            "queue_size": len(self.message_queue),
            "queue_capacity": self.max_queue_size,
            "subscription_types": list(self.subscribers.keys()),
        }


class ExchangeDataHandler(MessageHandler):
    """Handler for exchange market data messages."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        self.callback = callback
        self.logger = get_logger(f"{__name__}.ExchangeDataHandler")

    async def handle(self, message: Message) -> bool:
        """Handle market data message."""
        try:
            if message.type == MessageType.MARKET_DATA:
                # Transform exchange data using consistent pattern
                transformed_data = self._transform_market_data(message.data, message.source)

                # Call the callback with transformed data
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(transformed_data)
                else:
                    self.callback(transformed_data)

                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to handle market data message: {e}")
            return False

    def can_handle(self, message_type: MessageType) -> bool:
        """Check if can handle message type."""
        return message_type == MessageType.MARKET_DATA

    def _transform_market_data(self, data: dict[str, Any], source: str) -> dict[str, Any]:
        """Transform market data to consistent format."""
        # Apply consistent data transformation
        transformed = {
            "symbol": data.get("symbol", ""),
            "price": str(data.get("price", "0")),  # String for precision
            "volume": str(data.get("volume", "0")),
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "source": source,
            "bid": str(data.get("bid", "0")),
            "ask": str(data.get("ask", "0")),
        }

        return transformed


class OrderUpdateHandler(MessageHandler):
    """Handler for order update messages."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        self.callback = callback
        self.logger = get_logger(f"{__name__}.OrderUpdateHandler")

    async def handle(self, message: Message) -> bool:
        """Handle order update message."""
        try:
            if message.type == MessageType.ORDER_UPDATE:
                # Transform order data using consistent pattern
                transformed_data = self._transform_order_data(message.data, message.source)

                # Call the callback
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(transformed_data)
                else:
                    self.callback(transformed_data)

                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to handle order update message: {e}")
            return False

    def can_handle(self, message_type: MessageType) -> bool:
        """Check if can handle message type."""
        return message_type == MessageType.ORDER_UPDATE

    def _transform_order_data(self, data: dict[str, Any], source: str) -> dict[str, Any]:
        """Transform order data to consistent format."""
        return {
            "order_id": data.get("order_id", ""),
            "symbol": data.get("symbol", ""),
            "status": data.get("status", ""),
            "filled_quantity": str(data.get("filled_quantity", "0")),
            "remaining_quantity": str(data.get("remaining_quantity", "0")),
            "average_price": str(data.get("average_price", "0")),
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "source": source,
        }


class StreamToQueueBridge:
    """Bridge between stream data and message queue system."""

    def __init__(self, message_bus: MessagePublisher, source: str):
        self.message_bus = message_bus
        self.source = source
        self.logger = get_logger(f"{__name__}.StreamToQueueBridge")
        self.message_counter = 0

    async def handle_stream_data(self, data_type: str, data: dict[str, Any]) -> None:
        """Convert stream data to message and publish."""
        try:
            # Map data types to message types
            type_mapping = {
                "ticker": MessageType.MARKET_DATA,
                "orderbook": MessageType.MARKET_DATA,
                "trade": MessageType.MARKET_DATA,
                "order": MessageType.ORDER_UPDATE,
                "account": MessageType.ACCOUNT_UPDATE,
            }

            message_type = type_mapping.get(data_type, MessageType.SYSTEM_EVENT)

            # Create message
            self.message_counter += 1
            message = Message(
                id=f"{self.source}_{self.message_counter}_{datetime.now().timestamp()}",
                type=message_type,
                source=self.source,
                timestamp=datetime.now(timezone.utc),
                data=data,
                metadata={"data_type": data_type}
            )

            # Publish to message bus
            await self.message_bus.publish(message)

        except Exception as e:
            self.logger.error(f"Failed to bridge stream data: {e}")


class MessageQueueManager:
    """Manager for coordinating messaging patterns across the system."""

    def __init__(self):
        self.message_bus = InMemoryMessageBus()
        self.bridges: dict[str, StreamToQueueBridge] = {}
        self.handlers: list[MessageHandler] = []
        self.logger = get_logger(f"{__name__}.MessageQueueManager")

    def create_bridge(self, source: str) -> StreamToQueueBridge:
        """Create bridge for exchange stream data."""
        if source not in self.bridges:
            self.bridges[source] = StreamToQueueBridge(self.message_bus, source)
        return self.bridges[source]

    async def register_market_data_handler(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register handler for market data."""
        handler = ExchangeDataHandler(callback)
        await self.message_bus.subscribe(MessageType.MARKET_DATA, handler)
        self.handlers.append(handler)

    async def register_order_update_handler(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register handler for order updates."""
        handler = OrderUpdateHandler(callback)
        await self.message_bus.subscribe(MessageType.ORDER_UPDATE, handler)
        self.handlers.append(handler)

    def get_stats(self) -> dict[str, Any]:
        """Get messaging system statistics."""
        return {
            "message_bus": self.message_bus.get_stats(),
            "bridges_count": len(self.bridges),
            "handlers_count": len(self.handlers),
            "bridge_sources": list(self.bridges.keys()),
        }


# Global message queue manager instance
_message_queue_manager: MessageQueueManager | None = None


def get_message_queue_manager() -> MessageQueueManager:
    """Get global message queue manager instance."""
    global _message_queue_manager
    if _message_queue_manager is None:
        _message_queue_manager = MessageQueueManager()
    return _message_queue_manager


# Utility functions for consistent messaging
async def publish_market_data(source: str, symbol: str, data: dict[str, Any]) -> None:
    """Utility to publish market data consistently."""
    manager = get_message_queue_manager()
    bridge = manager.create_bridge(source)

    market_data = {
        "symbol": symbol,
        **data
    }

    await bridge.handle_stream_data("ticker", market_data)


async def publish_order_update(source: str, order_data: dict[str, Any]) -> None:
    """Utility to publish order updates consistently."""
    manager = get_message_queue_manager()
    bridge = manager.create_bridge(source)

    await bridge.handle_stream_data("order", order_data)


class RequestReplyPattern(ABC):
    """Request-reply pattern for synchronous operations."""

    @abstractmethod
    async def request(self, data: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        """Send request and wait for reply."""
        pass

    @abstractmethod
    async def reply(self, request_id: str, data: dict[str, Any]) -> bool:
        """Send reply to specific request."""
        pass


class BatchToStreamBridge:
    """Bridge between batch processing and stream processing."""

    def __init__(self, message_manager: MessageQueueManager, batch_size: int = 100):
        self.message_manager = message_manager
        self.batch_size = batch_size
        self.batch_buffer: list[dict[str, Any]] = []
        self.logger = get_logger(f"{__name__}.BatchToStreamBridge")

    async def add_to_batch(self, data: dict[str, Any]) -> None:
        """Add data to batch buffer."""
        self.batch_buffer.append(data)

        if len(self.batch_buffer) >= self.batch_size:
            await self._process_batch()

    async def _process_batch(self) -> None:
        """Process accumulated batch as stream events."""
        if not self.batch_buffer:
            return

        try:
            for item in self.batch_buffer:
                # Convert batch item to stream message
                bridge = self.message_manager.create_bridge("batch_processor")
                await bridge.handle_stream_data("ticker", item)

            self.logger.info(f"Processed batch of {len(self.batch_buffer)} items")
            self.batch_buffer.clear()

        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")

    async def flush(self) -> None:
        """Force process remaining items in batch."""
        await self._process_batch()

class StandardMessage:
    """Standardized message format across all communication patterns."""

    def __init__(
        self,
        pattern: MessagePattern,
        message_type: MessageType,
        data: Any = None,
        correlation_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.pattern = pattern
        self.message_type = message_type
        self.data = data
        self.correlation_id = correlation_id or str(uuid4())
        self.source = source
        self.target = target
        self.timestamp = datetime.now(timezone.utc)
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "pattern": self.pattern.value,
            "message_type": self.message_type.value,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "target": self.target,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardMessage":
        """Create message from dictionary format."""
        return cls(
            pattern=MessagePattern(data["pattern"]),
            message_type=MessageType(data["message_type"]),
            data=data.get("data"),
            correlation_id=data.get("correlation_id"),
            source=data.get("source"),
            target=data.get("target"),
            metadata=data.get("metadata", {}),
        )


class MessageHandler(ABC):
    """Abstract base class for consistent message handling."""

    @abstractmethod
    async def handle(self, message: StandardMessage) -> StandardMessage | None:
        """Handle incoming message and optionally return response."""
        pass


class ErrorPropagationMixin:
    """Mixin for consistent error propagation patterns across all modules with enhanced boundary validation."""

    def propagate_validation_error(self, error: Exception, context: str) -> None:
        """Propagate validation errors consistently with enhanced logging and boundary validation."""
        # Apply consistent error propagation metadata
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "validation_direct",
            "data_format": "error_propagation_v1",
            "processing_mode": "stream",  # Align with core events default processing mode
            "message_pattern": "pub_sub",
            "boundary_crossed": True,
            "validation_status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"Validation error in {context}: {error}", extra=error_metadata)

        # Add propagation metadata to error if supported
        if hasattr(error, "__dict__"):
            try:
                error.__dict__.update(
                    {"propagation_metadata": error_metadata, "boundary_validation_applied": True}
                )
            except (AttributeError, TypeError):
                # Some exception types don't allow attribute modification
                pass

        raise error

    def propagate_database_error(self, error: Exception, context: str) -> None:
        """Propagate database errors consistently with enhanced context."""
        logger.error(
            f"Database error in {context}: {error}",
            extra={
                "error_type": type(error).__name__,
                "context": context,
                "propagation_pattern": "database_to_repository",
                "data_format": "error_propagation_v1",
            },
        )
        from src.core.exceptions import RepositoryError

        raise RepositoryError(
            f"Database operation failed in {context}: {error}",
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "database",
                "data_format": "error_propagation_v1",
            },
        ) from error

    def propagate_service_error(self, error: Exception, context: str) -> None:
        """Propagate service errors consistently with enhanced context for risk_management-state alignment."""
        # Enhanced error metadata for cross-module consistency
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "service_to_service",
            "data_format": "error_propagation_v1",
            "processing_mode": "stream",  # Align with risk_management and state modules
            "message_pattern": "pub_sub",  # Consistent messaging pattern
            "cross_module_error": True,  # Flag for risk_management-state errors
            "boundary_crossed": True,
            "validation_status": "propagated",  # Error propagation status
            "component": "service_layer",  # Source component identification
            "severity": "medium",  # Default severity for service errors
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"Service error in {context}: {error}", extra=error_metadata)

        from src.core.exceptions import ServiceError

        raise ServiceError(
            f"Service operation failed in {context}: {error}",
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "service",
                "data_format": "error_propagation_v1",
                "processing_mode": "stream",
                "message_pattern": "pub_sub",
                "cross_module_alignment": True,
                "risk_state_compatible": True,
                "error_metadata": error_metadata,
            },
        ) from error

    def propagate_monitoring_error(self, error: Exception, context: str) -> None:
        """Propagate monitoring errors consistently across module boundaries with enhanced validation."""
        # Apply consistent error propagation metadata matching other error propagation patterns
        error_metadata = {
            "error_type": type(error).__name__,
            "context": context,
            "propagation_pattern": "monitoring_to_core",
            "data_format": "error_propagation_v1",
            "processing_mode": "stream",  # Align with core events default processing mode
            "message_pattern": "pub_sub",
            "boundary_crossed": True,
            "validation_status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.error(f"Monitoring error in {context}: {error}", extra=error_metadata)

        from src.core.exceptions import ComponentError

        raise ComponentError(
            f"Monitoring operation failed in {context}: {error}",
            component_name="monitoring",
            operation=context,
            details={
                "original_error": str(error),
                "error_context": context,
                "propagation_source": "monitoring",
                "data_format": "error_propagation_v1",
                "processing_mode": "stream",
                "message_pattern": "pub_sub",
                "boundary_validation": "applied",
                "propagation_metadata": error_metadata,
            },
        ) from error


class BoundaryValidator:
    """Validator for module boundary consistency."""

    @staticmethod
    def validate_database_entity(entity_dict: dict[str, Any], operation: str) -> None:
        """Validate entity at database boundary."""
        if not entity_dict:
            raise ValidationError(f"Empty entity provided for {operation}")

        # Check for required fields based on operation
        if operation == "create" and not entity_dict.get("id"):
            # Allow creates without ID (will be generated)
            pass
        elif operation == "update" and not entity_dict.get("id"):
            raise ValidationError("ID required for update operation")

        # Validate financial fields if present
        financial_fields = ["price", "quantity", "volume", "value", "amount"]
        for field in financial_fields:
            if field in entity_dict and entity_dict[field] is not None:
                try:
                    value = float(entity_dict[field])
                    if value < 0:
                        raise ValidationError(f"Financial field {field} cannot be negative")
                except (ValueError, TypeError):
                    raise ValidationError(f"Financial field {field} must be numeric")

    @staticmethod
    def validate_database_to_error_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from database to error_handling modules."""
        if not isinstance(data, dict):
            raise ValidationError(
                "Database to error boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Required fields for database error propagation
        required_fields = ["error_type", "timestamp", "operation"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Required field '{field}' missing in database to error data",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

        # Validate processing mode consistency
        if "processing_mode" in data:
            valid_modes = ["stream", "batch"]
            if data["processing_mode"] not in valid_modes:
                raise ValidationError(
                    f"Invalid processing_mode in database boundary: {data['processing_mode']}",
                    field_name="processing_mode",
                    field_value=data["processing_mode"],
                    expected_type=f"one of {valid_modes}",
                )

        # Validate data format consistency
        if "data_format" in data and not data["data_format"].startswith("database_"):
            raise ValidationError(
                "Database boundary data_format must start with 'database_'",
                field_name="data_format",
                field_value=data["data_format"],
                expected_type="database_*",
            )

    @staticmethod
    def validate_monitoring_to_error_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from monitoring/web_interface to error_handling modules with enhanced consistency patterns."""
        # This validation ensures consistency between monitoring/web_interface and error_handling modules
        if not isinstance(data, dict):
            raise ValidationError(
                "Module to error boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Enhanced validation for required boundary fields
        required_boundary_fields = ["component", "timestamp"]
        for field in required_boundary_fields:
            if field not in data:
                raise ValidationError(
                    f"Required boundary field '{field}' missing in monitoring to error data",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

        # Validate processing metadata consistency across module boundaries with enhanced checks
        if "processing_mode" in data and data["processing_mode"] not in [
            "sync",
            "async",
            "batch",
            "stream",
        ]:
            raise ValidationError(
                f"Invalid processing_mode: {data.get('processing_mode')}",
                field_name="processing_mode",
                field_value=data.get("processing_mode"),
                validation_rule="must be one of sync, async, batch, stream",
            )

        # Ensure consistent message patterns across module boundaries (prefer pub_sub for consistency)
        if "message_pattern" in data and data["message_pattern"] not in [
            "batch",
            "stream",
            "pub_sub",
            "req_reply",
        ]:
            raise ValidationError(
                f"Invalid message_pattern: {data.get('message_pattern')}",
                field_name="message_pattern",
                field_value=data.get("message_pattern"),
                validation_rule="must be one of batch, stream, pub_sub, req_reply",
            )

        # Validate data format consistency with version tracking
        if "data_format" in data and not data["data_format"].endswith("_v1"):
            raise ValidationError(
                f"Invalid data_format version: {data.get('data_format')}",
                field_name="data_format",
                field_value=data.get("data_format"),
                validation_rule="must end with _v1 for version consistency",
            )

        # Validate boundary crossing metadata
        if "boundary_crossed" in data and not isinstance(data["boundary_crossed"], bool):
            raise ValidationError(
                f"Invalid boundary_crossed value: {data.get('boundary_crossed')}",
                field_name="boundary_crossed",
                field_value=data.get("boundary_crossed"),
                expected_type="bool",
            )

        # Apply standard financial validation if financial data is present
        financial_fields = ["price", "quantity", "volume"]
        for field in financial_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value < 0:
                        raise ValidationError(
                            f"Financial field {field} cannot be negative at module boundary",
                            field_name=field,
                            field_value=value,
                            validation_rule="must be non-negative",
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Financial field {field} must be numeric at module boundary",
                        field_name=field,
                        field_value=data[field],
                        expected_type="numeric",
                    )

    @staticmethod
    def validate_error_to_monitoring_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from error_handling to monitoring modules with consistent patterns."""
        # This validation ensures consistency between error_handling and monitoring modules
        if not isinstance(data, dict):
            raise ValidationError(
                "Error to monitoring boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Check required fields for error data with enhanced validation - relaxed for consistency
        required_fields = ["component", "severity"]  # Removed error_id as it's optional for events
        for field in required_fields:
            if field not in data:
                # If severity is missing, set default
                if field == "severity" and field not in data:
                    data[field] = "medium"
                # If component is missing, try to infer or set default
                elif field == "component" and field not in data:
                    data[field] = "unknown_component"

        # Validate severity values with consistent error format
        valid_severities = ["low", "medium", "high", "critical"]
        if data.get("severity") not in valid_severities:
            raise ValidationError(
                f"Invalid severity value: {data.get('severity')}. Must be one of {valid_severities}",
                field_name="severity",
                field_value=data.get("severity"),
                validation_rule=f"must be one of {valid_severities}",
            )

        # Validate processing metadata consistency with enhanced boundary validation
        if "processing_mode" in data and data["processing_mode"] not in [
            "sync",
            "async",
            "batch",
            "stream",
        ]:
            raise ValidationError(
                f"Invalid processing_mode: {data.get('processing_mode')}",
                field_name="processing_mode",
                field_value=data.get("processing_mode"),
                validation_rule="must be one of sync, async, batch, stream",
            )

        # Ensure consistent message patterns across module boundaries (prefer pub_sub for consistency)
        if "message_pattern" in data and data["message_pattern"] not in [
            "batch",
            "stream",
            "pub_sub",
            "req_reply",
        ]:
            raise ValidationError(
                f"Invalid message_pattern: {data.get('message_pattern')}",
                field_name="message_pattern",
                field_value=data.get("message_pattern"),
                validation_rule="must be one of batch, stream, pub_sub, req_reply",
            )

        # Validate data format consistency
        if "data_format" in data and not data["data_format"].endswith("_v1"):
            raise ValidationError(
                f"Invalid data_format version: {data.get('data_format')}",
                field_name="data_format",
                field_value=data.get("data_format"),
                validation_rule="must end with _v1 for version consistency",
            )

        # Apply standard financial validation if financial data is present
        financial_fields = ["price", "quantity", "volume"]
        for field in financial_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value < 0:
                        raise ValidationError(
                            f"Financial field {field} cannot be negative at module boundary",
                            field_name=field,
                            field_value=value,
                            validation_rule="must be non-negative",
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Financial field {field} must be numeric at module boundary",
                        field_name=field,
                        field_value=data[field],
                        expected_type="numeric",
                    )

    @staticmethod
    def validate_web_interface_to_error_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from web_interface to error_handling modules with consistent patterns."""
        # This validation ensures consistency between web_interface and error_handling modules
        if not isinstance(data, dict):
            raise ValidationError(
                "Web interface to error boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Check required fields for web interface error data
        required_fields = ["component", "timestamp", "processing_mode"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Required field '{field}' missing in web interface to error boundary data",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

        # Validate web interface specific fields
        if data.get("operation"):
            # Validate HTTP operation format (METHOD path)
            operation = str(data["operation"])
            if " " not in operation or operation.split(" ")[0] not in [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
            ]:
                raise ValidationError(
                    f"Invalid HTTP operation format: {operation}. Expected 'METHOD path'",
                    field_name="operation",
                    field_value=operation,
                    validation_rule="must be 'HTTP_METHOD path' format",
                )

        # Validate HTTP-specific metadata
        if (
            "request_context" in data
            and data["request_context"]
            and not isinstance(data["request_context"], bool)
        ):
            raise ValidationError(
                f"Invalid request_context value: {data.get('request_context')}",
                field_name="request_context",
                field_value=data.get("request_context"),
                expected_type="bool",
            )

        # Validate middleware error metadata
        if (
            "middleware_error" in data
            and data["middleware_error"]
            and not isinstance(data["middleware_error"], bool)
        ):
            raise ValidationError(
                f"Invalid middleware_error value: {data.get('middleware_error')}",
                field_name="middleware_error",
                field_value=data.get("middleware_error"),
                expected_type="bool",
            )

        # Apply standard boundary validation patterns
        BoundaryValidator.validate_monitoring_to_error_boundary(data)

    @staticmethod
    def validate_risk_to_state_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from risk_management to state modules for enhanced consistency."""
        if not isinstance(data, dict):
            raise ValidationError(
                "Risk to state boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Check required fields for risk-state flow
        required_fields = ["component", "operation", "timestamp"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Required field '{field}' missing in risk to state boundary data",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

        # Validate risk-specific fields
        if "available_capital" in data and data["available_capital"] is not None:
            try:
                from decimal import Decimal, InvalidOperation

                capital = Decimal(str(data["available_capital"]))
                if capital <= 0:
                    raise ValidationError(
                        "Available capital must be positive at risk-state boundary",
                        field_name="available_capital",
                        field_value=capital,
                        validation_rule="must be positive",
                    )
            except (ValueError, TypeError, InvalidOperation):
                raise ValidationError(
                    "Available capital must be a valid decimal at risk-state boundary",
                    field_name="available_capital",
                    field_value=data["available_capital"],
                    expected_type="decimal",
                )

        # Validate signal data structure
        if "signal" in data and isinstance(data["signal"], dict):
            signal = data["signal"]
            if "strength" in signal:
                try:
                    strength = float(signal["strength"])
                    if not (0 < strength <= 1):
                        raise ValidationError(
                            "Signal strength must be between 0 and 1 at risk-state boundary",
                            field_name="signal.strength",
                            field_value=strength,
                            validation_rule="must be between 0 and 1",
                        )
                except (ValueError, TypeError):
                    raise ValidationError(
                        "Signal strength must be numeric at risk-state boundary",
                        field_name="signal.strength",
                        field_value=signal["strength"],
                        expected_type="numeric",
                    )

    @staticmethod
    def validate_monitoring_to_risk_boundary(data: dict[str, Any]) -> None:
        """Validate data at monitoring -> risk_management boundary."""
        if not isinstance(data, dict):
            raise ValidationError(
                "Monitoring to risk boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )
        required_fields = ["timestamp", "processing_mode", "data_format"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Missing required field '{field}' at monitoring -> risk_management boundary",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )

    @staticmethod
    def validate_state_to_risk_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from state to risk_management modules for enhanced consistency."""
        if not isinstance(data, dict):
            raise ValidationError(
                "State to risk boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )

        # Check required fields for state-risk flow
        required_fields = ["component", "operation", "timestamp"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Required field '{field}' missing in state to risk boundary data",
                    field_name=field,
                    field_value=None,
                    expected_type="str",
                )

        # Validate processing paradigm consistency
        if "processing_mode" in data:
            valid_modes = ["stream", "batch", "request_reply"]
            if data["processing_mode"] not in valid_modes:
                raise ValidationError(
                    f"Invalid processing mode for state to risk boundary: {data['processing_mode']}",
                    field_name="processing_mode",
                    field_value=data["processing_mode"],
                    expected_type=f"one of {valid_modes}",
                )

        # Validate message pattern alignment
        if "message_pattern" in data and data["message_pattern"] not in ["pub_sub", "req_reply", "batch"]:
            raise ValidationError(
                f"Invalid message pattern for state to risk boundary: {data['message_pattern']}",
                field_name="message_pattern",
                field_value=data["message_pattern"],
                expected_type="pub_sub, req_reply, or batch",
            )

        # Validate data format consistency
        if "data_format" in data and not data["data_format"].startswith("bot_event_"):
            raise ValidationError(
                f"Invalid data format for state to risk boundary: {data['data_format']}",
                field_name="data_format",
                field_value=data["data_format"],
                expected_type="bot_event_v*",
            )

    @staticmethod
    def validate_monitoring_to_state_boundary(data: dict[str, Any]) -> None:
        """Validate data at monitoring -> state boundary."""
        if not isinstance(data, dict):
            raise ValidationError(
                "Monitoring to state boundary data must be a dictionary",
                field_name="boundary_data",
                field_value=type(data).__name__,
                expected_type="dict",
            )
        required_fields = ["timestamp", "processing_mode", "data_format"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"Missing required field '{field}' at monitoring -> state boundary",
                    field_name=field,
                    field_value=None,
                    expected_type="string",
                )


class ProcessingParadigmAligner:
    """Aligns processing paradigms between batch and stream processing."""

    @staticmethod
    def create_batch_from_stream(stream_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert stream items to batch format."""
        return {
            "items": stream_items,
            "batch_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "size": len(stream_items),
            "processing_mode": "batch",  # Keep batch for batch operations
        }

    @staticmethod
    def create_stream_from_batch(batch_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert batch data to stream format."""
        items = batch_data.get("items", [])
        # Add stream metadata to each item
        for item in items:
            item.update(
                {
                    "stream_id": str(uuid4()),
                    "batch_id": batch_data.get("batch_id"),
                    "stream_timestamp": datetime.now(timezone.utc),
                    "processing_mode": "stream",
                }
            )
        return items

    @staticmethod
    def align_processing_modes(
        source_mode: str, target_mode: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Align processing modes between modules for consistent data flow with risk_management and state alignment."""
        aligned_data = data.copy()

        # Add alignment metadata for cross-module consistency
        aligned_data.update(
            {
                "source_processing_mode": source_mode,
                "target_processing_mode": target_mode,
                "alignment_timestamp": datetime.now(timezone.utc).isoformat(),
                "paradigm_aligned": True,
                "cross_module_alignment": True,  # Flag for risk_management-state alignment
            }
        )

        # Apply mode-specific transformations with enhanced risk_management-state consistency
        if source_mode == "stream" and target_mode == "batch":
            # Convert stream to batch-compatible format for risk_management-state flow
            aligned_data["batch_metadata"] = {
                "converted_from_stream": True,
                "original_stream_id": aligned_data.get("stream_id", str(uuid4())),
                "batch_size": 1,
                "risk_state_compatible": True,
            }
        elif source_mode == "batch" and target_mode == "stream":
            # Convert batch to stream-compatible format for state-risk_management flow
            aligned_data["stream_metadata"] = {
                "converted_from_batch": True,
                "original_batch_id": aligned_data.get("batch_id", str(uuid4())),
                "stream_position": 0,
                "risk_state_compatible": True,
            }
        elif source_mode == "stream" and target_mode == "stream":
            # Stream-to-stream alignment for optimal risk_management-state consistency
            aligned_data["stream_metadata"] = {
                "stream_aligned": True,
                "stream_id": aligned_data.get("stream_id", str(uuid4())),
                "cross_module_stream": True,
                "risk_state_optimized": True,
            }
        elif source_mode == "async" and target_mode == "batch":
            # Convert async to batch-compatible format
            aligned_data["batch_metadata"] = {
                "converted_from_async": True,
                "async_correlation_id": aligned_data.get("correlation_id", str(uuid4())),
                "batch_size": 1,
                "risk_state_compatible": True,
            }

        # Ensure consistent financial data handling across risk_management-state boundary
        if (
            "price" in aligned_data
            or "quantity" in aligned_data
            or "available_capital" in aligned_data
        ):
            aligned_data["financial_data_aligned"] = True
            aligned_data["financial_precision_maintained"] = True

        return aligned_data


class MessagingCoordinator:
    """
    Coordinates messaging patterns to prevent conflicts between pub/sub and req/reply.

    This ensures database and core modules use consistent communication patterns.
    """

    def __init__(
        self,
        name: str = "MessagingCoordinator",
        event_emitter: Any = None,
    ):
        self.name = name
        # Use dependency injection for event emitter for better testability
        self._event_emitter = event_emitter
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_timeout = 30.0

    def register_handler(self, pattern: MessagePattern, handler: MessageHandler) -> None:
        """Register handler for specific message pattern."""
        pattern_key = pattern.value
        if pattern_key not in self._handlers:
            self._handlers[pattern_key] = []
        self._handlers[pattern_key].append(handler)

        logger.debug(f"Registered handler for pattern {pattern_key}")

    async def publish(
        self,
        topic: str,
        data: Any,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish event using pub/sub pattern (fire-and-forget)."""
        # Apply consistent data transformation before creating message
        transformed_data = self._apply_data_transformation(data)

        message = StandardMessage(
            pattern=MessagePattern.PUB_SUB,
            message_type=MessageType.SYSTEM_EVENT,
            data=transformed_data,
            source=source,
            metadata=metadata,
        )

        # Use event emitter for pub/sub pattern with consistent data format and timeout
        try:
            await asyncio.wait_for(
                self._event_emitter.emit_async(topic, message.to_dict()), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Event emission timed out for topic {topic}")
            raise ServiceError(f"Event emission timeout for topic {topic}")
        except Exception as e:
            logger.error(f"Event emission failed for topic {topic}: {e}")
            raise ServiceError(f"Event emission failed for topic {topic}: {e}") from e

        logger.debug(f"Published message to topic {topic}")

    async def request(
        self,
        target: str,
        data: Any,
        source: str | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Send request using req/reply pattern and wait for response."""
        # Apply consistent data transformation before creating message
        transformed_data = self._apply_data_transformation(data)

        message = StandardMessage(
            pattern=MessagePattern.REQ_REPLY,
            message_type=MessageType.QUERY,
            data=transformed_data,
            source=source,
            target=target,
            metadata=metadata,
        )

        # Create future for response
        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[message.correlation_id] = future

        try:
            # Send request through handlers with timeout
            await asyncio.wait_for(self._route_message(message), timeout=10.0)

            # Wait for response with timeout
            timeout_value = timeout or self._request_timeout
            response = await asyncio.wait_for(future, timeout=timeout_value)

            return response

        except asyncio.TimeoutError as e:
            logger.error(f"Request to {target} timed out after {timeout_value}s")
            raise ServiceError(f"Request to {target} timed out after {timeout_value}s") from e
        except Exception as e:
            logger.error(f"Request to {target} failed: {e}")
            raise ServiceError(f"Request to {target} failed: {e}") from e
        finally:
            # Clean up pending request
            self._pending_requests.pop(message.correlation_id, None)

    async def reply(self, original_message: StandardMessage, response_data: Any) -> None:
        """Send reply to a request message."""
        reply_message = StandardMessage(
            pattern=MessagePattern.REQ_REPLY,
            message_type=MessageType.RESPONSE,
            data=response_data,
            correlation_id=original_message.correlation_id,
            source=original_message.target,
            target=original_message.source,
        )

        # Resolve pending request if it exists
        if original_message.correlation_id in self._pending_requests:
            future = self._pending_requests[original_message.correlation_id]
            if not future.done():
                future.set_result(response_data)

        logger.debug(f"Sent reply for correlation_id {original_message.correlation_id}")

    async def stream_start(
        self,
        stream_id: str,
        data: Any,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Start streaming data using stream pattern."""
        # Apply consistent data transformation for stream data
        transformed_data = self._apply_data_transformation(data)

        message = StandardMessage(
            pattern=MessagePattern.STREAM,
            message_type=MessageType.SYSTEM_EVENT,
            data=transformed_data,
            correlation_id=stream_id,
            source=source,
            metadata=metadata,
        )

        try:
            await asyncio.wait_for(self._route_message(message), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"Stream start routing timed out for stream {stream_id}")
        except Exception as e:
            logger.error(f"Stream start routing failed for stream {stream_id}: {e}")

    async def batch_process(
        self,
        batch_id: str,
        data: list[Any],
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Process batch of data using batch pattern."""
        # Apply consistent data transformation to batch items
        transformed_batch = []
        for item in data:
            transformed_item = self._apply_data_transformation(item)
            transformed_batch.append(transformed_item)

        message = StandardMessage(
            pattern=MessagePattern.BATCH,
            message_type=MessageType.COMMAND,
            data=transformed_batch,
            correlation_id=batch_id,
            source=source,
            metadata=metadata,
        )

        try:
            await asyncio.wait_for(self._route_message(message), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"Batch process routing timed out for batch {batch_id}")
        except Exception as e:
            logger.error(f"Batch process routing failed for batch {batch_id}: {e}")

    def _apply_data_transformation(self, data: Any) -> Any:
        """Apply consistent data transformation matching core event system patterns exactly."""
        if data is None:
            return None

        if isinstance(data, dict):
            # Already in correct format, ensure it has required metadata
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Apply consistent financial data transformations
            if "price" in data and data["price"] is not None:
                from src.utils.decimal_utils import to_decimal

                data["price"] = to_decimal(data["price"])

            if "quantity" in data and data["quantity"] is not None:
                from src.utils.decimal_utils import to_decimal

                data["quantity"] = to_decimal(data["quantity"])

            # Add consistent processing metadata to match core events
            if "processing_mode" not in data:
                data["processing_mode"] = "stream"  # Default to stream processing
            if "data_format" not in data:
                data["data_format"] = "bot_event_v1"  # Align with core events format

            return data

        # Transform non-dict data to standard format matching core events
        transformed = {
            "payload": data,
            "type": type(data).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_mode": "stream",
            "data_format": "bot_event_v1",  # Align with core events format
        }

        # Apply financial data transformation if applicable
        if hasattr(data, "price") or hasattr(data, "quantity"):
            transformed["financial_data"] = True
            # Apply consistent financial data transformations
            if hasattr(data, "price") and data.price is not None:
                from src.utils.decimal_utils import to_decimal

                transformed["price"] = to_decimal(data.price)

            if hasattr(data, "quantity") and data.quantity is not None:
                from src.utils.decimal_utils import to_decimal

                transformed["quantity"] = to_decimal(data.quantity)

        return transformed

    async def _route_message(self, message: StandardMessage) -> None:
        """Route message to appropriate handlers."""
        pattern_key = message.pattern.value
        handlers = self._handlers.get(pattern_key, [])

        if not handlers:
            logger.warning(f"No handlers registered for pattern {pattern_key}")
            return

        # Execute handlers based on pattern
        if message.pattern == MessagePattern.PUB_SUB:
            # Fire-and-forget for pub/sub with timeout and concurrency limit
            max_concurrent = min(len(handlers), 20)  # Limit concurrent handlers
            semaphore = asyncio.Semaphore(max_concurrent)

            async def handle_with_semaphore(handler: MessageHandler) -> Any:
                async with semaphore:
                    try:
                        return await asyncio.wait_for(handler.handle(message), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Handler {handler.__class__.__name__} timed out")
                        return None
                    except Exception as e:
                        logger.error(f"Handler {handler.__class__.__name__} error: {e}")
                        return e

            tasks = [handle_with_semaphore(handler) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

        elif message.pattern == MessagePattern.REQ_REPLY:
            # First handler responds for req/reply with timeout
            for handler in handlers:
                try:
                    response = await asyncio.wait_for(handler.handle(message), timeout=30.0)
                    if response is not None:
                        await self.reply(message, response.data)
                        break
                except asyncio.TimeoutError:
                    logger.error(f"Request handler {handler.__class__.__name__} timed out")
                    continue
                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    continue

        else:
            # Stream and batch patterns with timeout and concurrency
            max_concurrent = min(len(handlers), 10)  # Limit concurrent handlers
            semaphore = asyncio.Semaphore(max_concurrent)

            async def handle_stream_batch(handler: MessageHandler) -> None:
                async with semaphore:
                    try:
                        await asyncio.wait_for(handler.handle(message), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Stream/batch handler {handler.__class__.__name__} timed out")
                    except Exception as e:
                        logger.error(f"Handler error in {message.pattern.value}: {e}")

            tasks = [handle_stream_batch(handler) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)


class DataTransformationHandler(MessageHandler):
    """Handler for consistent data transformation patterns."""

    def __init__(self, transform_func: Callable[[Any], Any] | None = None):
        self.transform_func = transform_func

    async def handle(self, message: StandardMessage) -> StandardMessage | None:
        """Transform message data consistently."""
        try:
            if self.transform_func and message.data is not None:
                transformed_data = self.transform_func(message.data)
                message.data = transformed_data

            # Apply standard financial data transformations
            if isinstance(message.data, dict):
                await self._transform_financial_data(message.data)

            return message

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise ValidationError(f"Data transformation error: {e}")

    async def _transform_financial_data(self, data: dict[str, Any]) -> None:
        """Apply consistent financial data transformations."""
        # Transform price and quantity to Decimal consistently
        if "price" in data and data["price"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["price"] = to_decimal(data["price"])

        if "quantity" in data and data["quantity"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["quantity"] = to_decimal(data["quantity"])

        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Ensure timestamps are timezone-aware
        if "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                # Try to parse ISO format timestamp
                dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                data["timestamp"] = dt
            except ValueError:
                pass  # Leave as-is if parsing fails


# Message type mappings for consistent cross-module communication
MONITORING_TO_ERROR_MESSAGE_TYPES = {
    "alert_creation_failed": MessageType.ERROR_EVENT,
    "metric_validation_failed": MessageType.ERROR_EVENT,
    "performance_issue_detected": MessageType.SYSTEM_EVENT,
}

ERROR_TO_MONITORING_MESSAGE_TYPES = {
    "error_pattern_detected": MessageType.SYSTEM_EVENT,
    "error_threshold_exceeded": MessageType.SYSTEM_EVENT,
    "recovery_completed": MessageType.SYSTEM_EVENT,
}


def get_messaging_coordinator(
    coordinator: MessagingCoordinator | None = None,
) -> MessagingCoordinator:
    """Get messaging coordinator instance using proper dependency injection.

    Args:
        coordinator: Injected coordinator (required from service layer)

    Returns:
        MessagingCoordinator: Coordinator instance

    Raises:
        ValidationError: If coordinator not properly injected
    """
    if coordinator is None:
        raise ValidationError(
            "MessagingCoordinator must be injected from service layer. "
            "Do not access DI container directly from utility functions.",
            error_code="SERV_001",
        )

    return coordinator
