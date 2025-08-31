"""
Standardized messaging patterns for consistent pub/sub and req/reply handling.

This module ensures all data flow between database and core modules follows
consistent messaging patterns, eliminating conflicts between different
communication paradigms.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.base.events import BaseEventEmitter
from src.core.exceptions import DataValidationError, ServiceError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class MessagePattern(Enum):
    """Standardized message patterns."""

    PUB_SUB = "pub_sub"  # Fire-and-forget, one-to-many
    REQ_REPLY = "req_reply"  # Request-response, one-to-one
    STREAM = "stream"  # Continuous data flow
    BATCH = "batch"  # Batch processing


class MessageType(Enum):
    """Standard message types."""

    COMMAND = "command"  # Action to be performed
    EVENT = "event"  # Something that happened
    QUERY = "query"  # Request for data
    RESPONSE = "response"  # Response to query/command
    ERROR = "error"  # Error notification


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


class MessagingCoordinator:
    """
    Coordinates messaging patterns to prevent conflicts between pub/sub and req/reply.

    This ensures database and core modules use consistent communication patterns.
    """

    def __init__(self, name: str = "MessagingCoordinator"):
        self.name = name
        self._event_emitter = BaseEventEmitter(name=f"{name}_EventEmitter")
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
            message_type=MessageType.EVENT,
            data=transformed_data,
            source=source,
            metadata=metadata,
        )

        # Use event emitter for pub/sub pattern with consistent data format
        await self._event_emitter.emit_async(topic, message.to_dict())

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
            # Send request through handlers
            await self._route_message(message)

            # Wait for response with timeout
            timeout_value = timeout or self._request_timeout
            response = await asyncio.wait_for(future, timeout=timeout_value)

            return response

        except asyncio.TimeoutError:
            raise ServiceError(f"Request to {target} timed out after {timeout_value}s")
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
            message_type=MessageType.EVENT,
            data=transformed_data,
            correlation_id=stream_id,
            source=source,
            metadata=metadata,
        )

        await self._route_message(message)

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

        await self._route_message(message)

    def _apply_data_transformation(self, data: Any) -> Any:
        """Apply consistent data transformation matching core event system patterns."""
        if data is None:
            return None

        if isinstance(data, dict):
            # Apply consistent financial data transformations
            if "price" in data and data["price"] is not None:
                from src.utils.decimal_utils import to_decimal
                data["price"] = to_decimal(data["price"])
            
            if "quantity" in data and data["quantity"] is not None:
                from src.utils.decimal_utils import to_decimal
                data["quantity"] = to_decimal(data["quantity"])
                
            # Ensure timestamp consistency
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()
                
            return data
        
        # For non-dict data, ensure consistent format
        return data

    async def _route_message(self, message: StandardMessage) -> None:
        """Route message to appropriate handlers."""
        pattern_key = message.pattern.value
        handlers = self._handlers.get(pattern_key, [])

        if not handlers:
            logger.warning(f"No handlers registered for pattern {pattern_key}")
            return

        # Execute handlers based on pattern
        if message.pattern == MessagePattern.PUB_SUB:
            # Fire-and-forget for pub/sub
            tasks = [handler.handle(message) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

        elif message.pattern == MessagePattern.REQ_REPLY:
            # First handler responds for req/reply
            for handler in handlers:
                try:
                    response = await handler.handle(message)
                    if response is not None:
                        await self.reply(message, response.data)
                        break
                except Exception as e:
                    logger.error(f"Handler error: {e}")

        else:
            # Stream and batch patterns
            for handler in handlers:
                try:
                    await handler.handle(message)
                except Exception as e:
                    logger.error(f"Handler error in {message.pattern.value}: {e}")


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


class ErrorPropagationMixin:
    """Mixin for consistent error propagation across monitoring and error_handling modules."""

    def propagate_validation_error(self, error: Exception, context: str) -> None:
        """Propagate validation errors consistently between modules."""
        if isinstance(error, (ValidationError, DataValidationError)):
            # Re-raise validation errors as-is for consistent error handling
            logger.error(f"Validation error in {context}: {error}")
            raise error

        # Wrap other errors in appropriate types
        if isinstance(error, ValueError):
            raise DataValidationError(
                f"Value error in {context}: {error}",
                field_name="unknown",
                field_value=str(error),
                expected_type="valid value",
            ) from error


class ProcessingParadigmAligner:
    """Aligns processing paradigms between monitoring (sync/async) and error_handling (async/stream)."""

    @staticmethod
    def align_processing_modes(source_mode: str, target_mode: str, data: dict[str, Any]) -> dict[str, Any]:
        """Align processing modes for consistent data flow."""
        aligned_data = data.copy()

        # Add processing mode metadata for consistency
        aligned_data.update(
            {
                "source_processing_mode": source_mode,
                "target_processing_mode": target_mode,
                "aligned_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Apply mode-specific transformations
        if source_mode == "sync" and target_mode == "async":
            aligned_data["sync_to_async_transition"] = True
        elif source_mode == "async" and target_mode == "stream":
            aligned_data["async_to_stream_transition"] = True
        elif source_mode == "batch" and target_mode == "stream":
            aligned_data["batch_to_stream_transition"] = True

        return aligned_data

    @staticmethod
    def create_batch_from_stream(stream_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert stream processing data to batch format for consistency."""
        return {
            "batch_id": str(uuid4())[:8],
            "batch_size": len(stream_data),
            "items": stream_data,
            "processing_mode": "batch",
            "data_format": "batch_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


class BoundaryValidator:
    """Validates data at module boundaries for consistency."""

    @staticmethod
    def validate_monitoring_to_error_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from monitoring to error_handling module."""
        # Required fields for monitoring -> error_handling communication
        required_fields = ["component", "error_type", "severity", "timestamp"]

        for field in required_fields:
            if field not in data or data[field] is None:
                raise DataValidationError(
                    f"Required field {field} missing at monitoring->error_handling boundary",
                    field_name=field,
                    field_value=data.get(field),
                    expected_type="non-None value",
                )

        # Apply consistent financial data transformations
        if "price" in data and data["price"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["price"] = to_decimal(data["price"])

        if "quantity" in data and data["quantity"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["quantity"] = to_decimal(data["quantity"])

    @staticmethod
    def validate_error_to_monitoring_boundary(data: dict[str, Any]) -> None:
        """Validate data flowing from error_handling to monitoring module."""
        # Required fields for error_handling -> monitoring communication
        required_fields = ["error_id", "severity", "recovery_success", "timestamp"]

        for field in required_fields:
            if field not in data or data[field] is None:
                raise DataValidationError(
                    f"Required field {field} missing at error_handling->monitoring boundary",
                    field_name=field,
                    field_value=data.get(field),
                    expected_type="non-None value",
                )

        # Apply consistent financial data transformations
        if "price" in data and data["price"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["price"] = to_decimal(data["price"])

        if "quantity" in data and data["quantity"] is not None:
            from src.utils.decimal_utils import to_decimal

            data["quantity"] = to_decimal(data["quantity"])


# Message type mappings for consistent cross-module communication
MONITORING_TO_ERROR_MESSAGE_TYPES = {
    "alert_creation_failed": MessageType.ERROR,
    "metric_validation_failed": MessageType.ERROR,
    "performance_issue_detected": MessageType.EVENT,
}

ERROR_TO_MONITORING_MESSAGE_TYPES = {
    "error_pattern_detected": MessageType.EVENT,
    "error_threshold_exceeded": MessageType.EVENT,
    "recovery_completed": MessageType.EVENT,
}


def get_messaging_coordinator(coordinator: MessagingCoordinator | None = None) -> MessagingCoordinator:
    """Get messaging coordinator instance using proper dependency injection.
    
    Args:
        coordinator: Injected coordinator (required from service layer)
        
    Returns:
        MessagingCoordinator: Coordinator instance
        
    Raises:
        ValidationError: If coordinator not properly injected
    """
    if coordinator is None:
        # Fallback to DI but warn about violation
        logger.warning("MessagingCoordinator not injected - violates clean architecture. Inject from service layer.")
        try:
            from src.core.dependency_injection import injector
            return injector.resolve("MessagingCoordinator")
        except Exception as e:
            raise ValidationError(
                "MessagingCoordinator must be injected from service layer. "
                "Do not access DI container directly from utility functions.",
                error_code="SERV_001"
            ) from e
    
    return coordinator
